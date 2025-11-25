import json
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _gini(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    sorted_vals, _ = torch.sort(values.flatten())
    n = sorted_vals.numel()
    index = torch.arange(1, n + 1, device=sorted_vals.device, dtype=sorted_vals.dtype)
    numerator = (2 * index - n - 1) * sorted_vals
    denom = sorted_vals.sum().clamp_min(1e-9) * n
    return _to_float(numerator.sum() / denom)


def analyze_routing(pi: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
    """
    pi: [B, T, K] routing weights (soft)
    attention_mask: [B, T] with 1 for valid tokens
    """
    eps = 1e-8
    if attention_mask is None:
        attention_mask = torch.ones(pi.shape[:2], device=pi.device, dtype=pi.dtype)
    mask = attention_mask.to(pi.dtype)
    masked_pi = pi * mask.unsqueeze(-1)
    token_entropy = -(masked_pi.clamp_min(eps) * masked_pi.clamp_min(eps).log()).sum(dim=-1)
    total_tokens = mask.sum(dim=1).clamp_min(1.0)
    avg_entropy = (token_entropy * mask).sum(dim=1) / total_tokens

    usage = masked_pi.sum(dim=1) / total_tokens.unsqueeze(-1)
    usage_var = usage.var(dim=-1, unbiased=False)
    gini = torch.stack([torch.tensor(_gini(row)) for row in usage])

    hard = pi.argmax(dim=-1)
    counts = []
    for b in range(pi.size(0)):
        valid = mask[b] > 0
        hist = torch.bincount(hard[b][valid], minlength=pi.size(-1)).to(torch.float)
        counts.append(hist)
    counts = torch.stack(counts)

    continuity = []
    for b in range(pi.size(0)):
        valid = mask[b] > 0
        if valid.sum() <= 1:
            continuity.append(torch.tensor(1.0, device=pi.device))
            continue
        ids = hard[b][valid]
        same = (ids[1:] == ids[:-1]).to(torch.float)
        continuity.append(same.mean())
    continuity = torch.stack(continuity)

    return {
        "token_entropy_mean": _to_float(avg_entropy.mean()),
        "token_entropy_per_sample": avg_entropy.cpu().tolist(),
        "usage": usage.cpu().tolist(),
        "usage_variance_mean": _to_float(usage_var.mean()),
        "gini_mean": _to_float(gini.mean()),
        "hard_counts": counts.cpu().tolist(),
        "continuity_mean": _to_float(continuity.mean()),
    }


def analyze_experts(z: torch.Tensor) -> Dict:
    """
    z: [K, D] expert embeddings for one sentence (or [B, K, D] -> will be averaged over batch)
    """
    if z.dim() == 3:
        z = z.mean(dim=0)
    z_norm = F.normalize(z, dim=-1)
    cosine_matrix = z_norm @ z_norm.t()
    frob = torch.linalg.norm(cosine_matrix)
    norms = z.norm(dim=-1)

    off_diag = cosine_matrix - torch.eye(cosine_matrix.size(0), device=cosine_matrix.device)
    off_mask = torch.ones_like(cosine_matrix, dtype=torch.bool)
    off_mask.fill_(True)
    off_mask.fill_diagonal_(False)
    diversity = (1.0 - cosine_matrix[off_mask]).mean() if off_mask.any() else torch.tensor(0.0)

    rank = torch.linalg.matrix_rank(z)

    return {
        "cosine_matrix": cosine_matrix.cpu().tolist(),
        "frobenius_norm": _to_float(frob),
        "norms": norms.cpu().tolist(),
        "norm_mean": _to_float(norms.mean()),
        "diversity": _to_float(diversity),
        "rank": int(rank.item()),
    }


@dataclass
class StabilityState:
    mean: torch.Tensor
    covariance: torch.Tensor
    stability: List[float]


class ExpertTracker:
    """Tracks moving averages for expert embeddings."""

    def __init__(self, maxlen: int = 100) -> None:
        self.history: Deque[torch.Tensor] = deque(maxlen=maxlen)

    def update(self, z: torch.Tensor) -> StabilityState:
        """
        z: [K, D]
        """
        self.history.append(z.detach().cpu())
        stacked = torch.stack(list(self.history))  # [N, K, D]
        mean = stacked.mean(dim=0)
        flat = stacked.view(stacked.size(0), -1).t()
        covariance = torch.cov(flat)

        stability = []
        for k in range(z.size(0)):
            current = z[k]
            ref = mean[k].to(current.device)
            num = F.cosine_similarity(current.unsqueeze(0), ref.unsqueeze(0)).item()
            stability.append(num)

        return StabilityState(mean=mean, covariance=covariance, stability=stability)


def analyze_sbert_alignment(h: torch.Tensor, h_sbert: torch.Tensor, z: torch.Tensor) -> Dict:
    h_norm = F.normalize(h, dim=-1)
    h_sbert_norm = F.normalize(h_sbert, dim=-1)
    cos_h = F.cosine_similarity(h_norm.unsqueeze(0), h_sbert_norm.unsqueeze(0)).item()

    z_norm = F.normalize(z, dim=-1)
    contributions = torch.matmul(z_norm, h_sbert_norm)
    captured = contributions.abs()
    coop = (contributions > 0).float().mean()

    return {
        "cosine_pred_target": _to_float(cos_h),
        "contributions": contributions.cpu().tolist(),
        "captured_norm_fraction": captured.cpu().tolist(),
        "cooperation_fraction": _to_float(coop),
    }


def perturbation_impact(
    z: torch.Tensor,
    combine_fn: Callable[[torch.Tensor], torch.Tensor],
    h_orig: torch.Tensor,
) -> Dict:
    impacts = []
    with torch.no_grad():
        base = F.normalize(h_orig, dim=-1)
        for k in range(z.size(0)):
            z_mod = z.clone()
            z_mod[k] = 0.0
            h_mod = combine_fn(z_mod)
            h_mod = F.normalize(h_mod, dim=-1)
            dist = 1.0 - F.cosine_similarity(base.unsqueeze(0), h_mod.unsqueeze(0)).item()
            impacts.append(dist)
    impacts_tensor = torch.tensor(impacts)
    sorted_idx = torch.argsort(impacts_tensor, descending=True).tolist()
    return {
        "impacts": impacts_tensor.cpu().tolist(),
        "sorted_indices": sorted_idx,
        "impact_mean": _to_float(impacts_tensor.mean()),
        "impact_max": _to_float(impacts_tensor.max()),
    }


def extract_spans(pi: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
    """
    Return spans per expert where it is the dominant expert.
    """
    if attention_mask is None:
        attention_mask = torch.ones(pi.shape[:2], device=pi.device, dtype=torch.long)
    mask = attention_mask > 0
    hard = pi.argmax(dim=-1)
    K = pi.size(-1)
    spans = {k: [] for k in range(K)}
    span_stats = {k: {"count": 0, "avg_len": 0.0} for k in range(K)}

    for b in range(pi.size(0)):
        seq = hard[b]
        valid = mask[b]
        start = None
        current_k = None
        for t in range(seq.size(0)):
            if not valid[t]:
                continue
            if current_k is None:
                current_k = seq[t].item()
                start = t
                continue
            if seq[t].item() == current_k:
                continue
            spans[current_k].append((b, start, t))
            current_k = seq[t].item()
            start = t
        if current_k is not None and start is not None:
            spans[current_k].append((b, start, seq.size(0)))

    for k, lst in spans.items():
        lengths = [end - start for (_, start, end) in lst]
        if lengths:
            span_stats[k]["count"] = len(lengths)
            span_stats[k]["avg_len"] = sum(lengths) / len(lengths)

    return {
        "spans": spans,
        "span_stats": span_stats,
    }


def detect_collapse(metrics: Dict, thresholds: Optional[Dict] = None) -> List[str]:
    thresholds = thresholds or {}
    warnings = []

    ent = metrics.get("routing", {}).get("token_entropy_mean")
    if ent is not None and ent < thresholds.get("entropy_min", 0.1):
        warnings.append("Routing entropy is low; experts may have collapsed.")

    gini = metrics.get("routing", {}).get("gini_mean")
    if gini is not None and gini > thresholds.get("gini_max", 0.7):
        warnings.append("Expert usage Gini is high; few experts dominate.")

    rank = metrics.get("experts", {}).get("rank")
    if rank is not None and rank <= thresholds.get("rank_min", 1):
        warnings.append("Expert matrix is near rank-1; experts look identical.")

    norm_mean = metrics.get("experts", {}).get("norm_mean")
    if norm_mean is not None:
        if norm_mean < thresholds.get("norm_min", 0.1):
            warnings.append("Expert norms are vanishing.")
        if norm_mean > thresholds.get("norm_max", 10.0):
            warnings.append("Expert norms are exploding.")

    impacts = metrics.get("perturbation", {}).get("impact_mean")
    if impacts is not None and impacts < thresholds.get("impact_min", 0.01):
        warnings.append("Perturbation impacts are tiny; experts may be unused.")

    stability = metrics.get("stability", {}).get("stability_mean")
    if stability is not None:
        if stability > thresholds.get("stability_max", 0.99):
            warnings.append("Experts are too stable; may be stuck.")
        if stability < thresholds.get("stability_min", 0.1):
            warnings.append("Experts are unstable across sentences.")

    return warnings


def debug_report(
    pi: torch.Tensor,
    z: torch.Tensor,
    h: torch.Tensor,
    h_sbert: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    tracker: Optional[ExpertTracker] = None,
    combine_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    writer: Optional["SummaryWriter"] = None,
    step: Optional[int] = None,
) -> Dict:
    if combine_fn is None:
        combine_fn = lambda factors: factors.sum(dim=0)

    routing_metrics = analyze_routing(pi, attention_mask)
    expert_metrics = analyze_experts(z)
    stability_metrics = {}
    if tracker is not None:
        state = tracker.update(z)
        stability_metrics = {
            "stability": state.stability,
            "stability_mean": _to_float(torch.tensor(state.stability).mean()),
        }
    alignment_metrics = analyze_sbert_alignment(h, h_sbert, z)
    perturb_metrics = perturbation_impact(z, combine_fn, h)
    spans = extract_spans(pi, attention_mask)

    report = {
        "routing": routing_metrics,
        "experts": expert_metrics,
        "stability": stability_metrics,
        "alignment": alignment_metrics,
        "perturbation": perturb_metrics,
        "spans": spans["span_stats"],
    }

    warnings = detect_collapse(report)
    if warnings:
        report["warnings"] = warnings

    if writer is not None and step is not None:
        for name, block in report.items():
            if not isinstance(block, dict):
                continue
            for key, val in block.items():
                if isinstance(val, (int, float)):
                    writer.add_scalar(f"{name}/{key}", val, global_step=step)
        if warnings:
            for w in warnings:
                writer.add_text("warnings", w, global_step=step)

    return json.loads(json.dumps(report, default=_to_float))
