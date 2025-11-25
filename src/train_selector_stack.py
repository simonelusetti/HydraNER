import os
import copy

import torch
from torch import nn
from tqdm import tqdm
from prettytable import PrettyTable

from dora import get_xp, hydra_main

from .selector.models import RationaleSelectorModel
from .expert.models import ExpertModel
from .utils import (
    load_sbert_pooler,
    should_disable_tqdm,
    get_logger,
    configure_runtime,
    counts,
    metrics_from_counts,
    nt_xent,
    complement_loss,
    sparsity_loss as rat_sparsity_loss,
    total_variation_1d,
)
from .data import initialize_dataloaders

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _build_selection_mask(gates: torch.Tensor, attention_mask: torch.Tensor, threshold: float) -> torch.Tensor:
    """Binary selection mask with fallback to keep at least one token."""
    mask_float = attention_mask.to(dtype=gates.dtype)
    selection = (gates >= threshold).to(dtype=gates.dtype) * mask_float
    selected_counts = selection.sum(dim=1)
    need_fallback = selected_counts == 0
    if need_fallback.any():
        masked_gates = gates.masked_fill(mask_float == 0, -1e9)
        top_indices = masked_gates.argmax(dim=1)
        rows = torch.arange(gates.size(0), device=gates.device)
        selection[rows[need_fallback], top_indices[need_fallback]] = 1.0
        selection = selection * mask_float
    return selection


class StackLayer(nn.Module):
    """Single selector+expert block with optional mixing of previous layer factors."""

    def __init__(
        self,
        selector_cfg,
        expert_cfg,
        selector_backbone,
        expert_backbone,
        device,
        mix_init=0.5,
        mix_trainable=True,
    ) -> None:
        super().__init__()
        selector_pooler, selector_hidden_dim = selector_backbone
        expert_pooler, expert_hidden_dim = expert_backbone

        self.selector_cfg = selector_cfg
        self.expert_cfg = expert_cfg
        self.device = device

        self.selector = RationaleSelectorModel(
            self.selector_cfg,
            pooler=copy.deepcopy(selector_pooler),
            embedding_dim=selector_hidden_dim,
        ).to(device)

        self.expert = ExpertModel(
            self.expert_cfg,
            pooler=copy.deepcopy(expert_pooler),
            embedding_dim=expert_hidden_dim,
        ).to(device)

        mix_init = float(mix_init)
        if mix_trainable:
            initial = torch.tensor(mix_init).clamp(1e-4, 1.0 - 1e-4)
            self.mix_logit = nn.Parameter(torch.log(initial / (1.0 - initial)))
        else:
            self.register_parameter("mix_logit", None)

    def _selector_forward(self, embeddings, mask):
        outputs = self.selector(embeddings, mask)

        h_anchor = outputs["h_anchor"]
        h_rat = outputs["h_rat"]
        h_comp = outputs["h_comp"]
        gates = outputs["gates"]

        rat_loss = nt_xent(h_rat, h_anchor, temperature=self.selector_cfg.loss.tau)
        comp_loss = complement_loss(
            h_comp,
            h_anchor,
            temperature=float(self.selector_cfg.loss.tau),
        )
        sparsity = rat_sparsity_loss(gates, mask)
        tv = total_variation_1d(gates, mask)

        loss = rat_loss
        loss = loss + float(self.selector_cfg.loss.l_comp) * comp_loss
        loss = loss + float(self.selector_cfg.loss.l_s) * sparsity
        loss = loss + float(self.selector_cfg.loss.l_tv) * tv

        losses = {
            "rat": float(rat_loss.detach()),
            "comp": float(comp_loss.detach()),
            "sparsity": float(sparsity.detach()),
            "tv": float(tv.detach()),
            "total": float(loss.detach()),
        }

        return loss, losses, outputs

    def _reconstruct_from_factors(self, factors, routing_weights):
        recon_parts = []
        for idx, head in enumerate(self.expert.reconstruction_heads):
            recon_parts.append(head(factors[:, idx, :]))
        reconstruction = torch.stack(recon_parts, dim=1).sum(dim=1)

        token_reconstruction = None
        if self.expert.token_decoder is not None:
            mixture = torch.einsum("btk,bkf->btf", routing_weights, factors)
            token_reconstruction = self.expert.token_decoder(mixture)
        return reconstruction, token_reconstruction

    def _expert_loss(self, routing_weights, factors, anchor, reconstruction, token_reconstruction, mask, selected_embeddings):
        mask_float = mask.to(dtype=routing_weights.dtype)

        anchor = torch.nn.functional.normalize(anchor, dim=-1)
        reconstruction = torch.nn.functional.normalize(reconstruction, dim=-1)

        logits = anchor @ reconstruction.t() / max(float(self.expert_cfg.contrastive_tau), 1e-6)
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_ab = torch.nn.functional.cross_entropy(logits, targets)
        loss_ba = torch.nn.functional.cross_entropy(logits.t(), targets)
        sent_loss = 0.5 * (loss_ab + loss_ba)

        token_loss = routing_weights.new_tensor(0.0)
        if token_reconstruction is not None and selected_embeddings is not None:
            mask_float_tok = mask.unsqueeze(-1).to(dtype=token_reconstruction.dtype)
            diff = token_reconstruction - selected_embeddings.to(dtype=token_reconstruction.dtype)
            token_loss = (diff.pow(2) * mask_float_tok).sum() / mask_float_tok.sum().clamp_min(1.0)

        entropy = -(routing_weights.clamp_min(self.expert.small_value).log() * routing_weights)
        entropy = (entropy.sum(dim=-1) * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        entropy_loss = entropy.mean()

        pi_sq = (routing_weights ** 2).sum(dim=-1)
        overlap = 0.5 * (1.0 - pi_sq)
        overlap = (overlap * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        overlap_loss = overlap.mean()

        if self.expert.use_balance:
            expert_mass = routing_weights.sum(dim=1)
            total_tokens = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            balanced_mass = expert_mass / total_tokens
            target = routing_weights.new_full((1, self.expert.num_experts), 1.0 / self.expert.num_experts)
            balance_loss = ((balanced_mass.mean(dim=0, keepdim=True) - target) ** 2).sum()
        else:
            balance_loss = routing_weights.new_zeros(())

        if self.expert.use_diversity:
            diversity_loss = self.expert._compute_diversity_penalty(factors)
        else:
            diversity_loss = routing_weights.new_zeros(())

        continuity_loss = None
        if self.expert.use_continuity:
            if routing_weights.size(1) > 1:
                pair_mask = mask_float[:, 1:] * mask_float[:, :-1]
                diff = routing_weights[:, 1:, :] - routing_weights[:, :-1, :]
                diff_sq = diff.pow(2).sum(dim=-1)
                numerator = (diff_sq * pair_mask).sum(dim=1)
                denominator = pair_mask.sum(dim=1).clamp_min(self.expert.small_value)
                continuity_loss = numerator / denominator
            else:
                continuity_loss = routing_weights.new_zeros(routing_weights.size(0))

        loss_components = {
            "sent": sent_loss,
            "token": token_loss,
            "entropy": entropy_loss,
            "overlap": overlap_loss,
            "diversity": diversity_loss,
            "balance": balance_loss,
        }

        if getattr(self.expert_cfg.loss_weights, "continuity", None) is not None:
            if continuity_loss is None:
                raise KeyError("Continuity weight configured but model continuity output missing.")
            loss_components["continuity"] = continuity_loss.mean()

        weights_cfg = self.expert_cfg.loss_weights
        loss = 0.0
        for key, val in loss_components.items():
            weight = float(getattr(weights_cfg, key))
            loss = loss + weight * val

        losses = {key: float(value.detach()) for key, value in loss_components.items()}
        losses["total"] = float(loss.detach())
        return loss, losses

    def forward(self, embeddings, attention_mask, prev_factors=None):
        selector_loss, selector_losses, selector_output = self._selector_forward(
            embeddings, attention_mask
        )

        selection_mask = _build_selection_mask(selector_output["gates"], attention_mask, float(self.selector_cfg.threshold))
        selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
        selected_mask = (attention_mask * selection_mask.long()).clamp(max=1)

        expert_output = self.expert(selected_embeddings, selected_mask)
        factors = expert_output["factors"]
        routing_weights = expert_output["pi"]

        mix_alpha = None
        if prev_factors is not None:
            mix_alpha = torch.sigmoid(self.mix_logit) if self.mix_logit is not None else factors.new_tensor(1.0)
            factors_mixed = mix_alpha * factors + (1.0 - mix_alpha) * prev_factors
        else:
            factors_mixed = factors

        reconstruction, token_reconstruction = self._reconstruct_from_factors(factors_mixed, routing_weights)
        expert_loss, expert_losses = self._expert_loss(
            routing_weights,
            factors_mixed,
            expert_output["anchor"],
            reconstruction,
            token_reconstruction,
            selected_mask,
            selected_embeddings,
        )

        return {
            "selector_loss": selector_loss,
            "selector_losses": selector_losses,
            "expert_loss": expert_loss,
            "expert_losses": expert_losses,
            "selection_mask": selection_mask,
            "routing": routing_weights,
            "factors": factors,
            "mixed_factors": factors_mixed,
            "mix_alpha": float(mix_alpha.detach()) if mix_alpha is not None else None,
            "selected_mask": selected_mask,
        }


class SelectorStack(nn.Module):
    """Sequential stack of selector+expert layers that mix factors across depth."""

    def __init__(
        self,
        selector_cfg,
        expert_cfg,
        device,
        num_layers,
        mix_init=0.5,
        mix_trainable=True,
    ) -> None:
        super().__init__()
        self.selector_cfg = selector_cfg
        self.expert_cfg = expert_cfg
        self.device = device
        self.num_layers = int(num_layers)

        selector_backbone = load_sbert_pooler(selector_cfg.encoder_name, device=self.device)
        expert_backbone = load_sbert_pooler(expert_cfg.encoder_name, device=self.device)

        layers = []
        for _ in range(self.num_layers):
            layer = StackLayer(
                selector_cfg,
                expert_cfg,
                selector_backbone,
                expert_backbone,
                device,
                mix_init=mix_init,
                mix_trainable=mix_trainable,
            )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, embeddings, attention_mask):
        outputs = []
        prev_factors = None
        for layer in self.layers:
            out = layer(embeddings, attention_mask, prev_factors=prev_factors)
            prev_factors = out["mixed_factors"]
            outputs.append(out)
        return outputs


class SelectorStackTrainer:
    def __init__(self, cfg, train_dl, eval_dl, dev_dl, logger, xp, device) -> None:
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.xp = xp
        self.disable_progress = should_disable_tqdm()

        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.dev_dl = dev_dl

        self.selector_weight = float(cfg.train.loss_weights.selector)
        self.expert_weight = float(cfg.train.loss_weights.expert)
        self.grad_clip = float(cfg.train.grad_clip)
        self.num_epochs = int(cfg.train.epochs)

        self.model = SelectorStack(
            cfg.selector_model,
            cfg.expert_model,
            device,
            num_layers=cfg.stack.num_layers,
            mix_init=cfg.stack.mix_init,
            mix_trainable=cfg.stack.mix_trainable,
        )

        selector_params = []
        expert_params = []
        for layer in self.model.layers:
            selector_params += list(layer.selector.parameters())
            expert_params += list(layer.expert.parameters())
            if layer.mix_logit is not None:
                expert_params.append(layer.mix_logit)

        self.selector_params = selector_params
        self.expert_params = expert_params

        sel_cfg = cfg.selector_model.optim
        exp_cfg = cfg.expert_model.optim
        self.selector_optimizer = torch.optim.AdamW(
            self.selector_params,
            lr=float(sel_cfg.lr),
            weight_decay=float(sel_cfg.weight_decay),
            betas=tuple(sel_cfg.betas),
        )
        self.expert_optimizer = torch.optim.AdamW(
            self.expert_params,
            lr=float(exp_cfg.lr),
            weight_decay=float(exp_cfg.weight_decay),
            betas=tuple(exp_cfg.betas),
        )

    def _metrics_table(self, metrics):
        if not metrics:
            return ""
        table = PrettyTable()
        table.field_names = metrics[0].keys()
        for row in metrics:
            formatted_row = [
                f"{v:.5f}" if isinstance(v, float) else v
                for v in row.values()
            ]
            table.add_row(formatted_row)
        return table

    def _single_epoch(self, epoch_idx):
        total_loss = 0.0
        self.model.train()

        for batch in tqdm(self.train_dl, desc=f"Selector Stack Train {epoch_idx + 1}", disable=self.disable_progress):
            embeddings = batch["embeddings"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            self.selector_optimizer.zero_grad(set_to_none=True)
            self.expert_optimizer.zero_grad(set_to_none=True)

            outputs = self.model(embeddings, attention_mask)
            selector_loss = sum(out["selector_loss"] for out in outputs)
            expert_loss = sum(out["expert_loss"] for out in outputs)
            loss = self.selector_weight * selector_loss + self.expert_weight * expert_loss

            loss.backward()

            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.selector_params, self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.expert_params, self.grad_clip)

            self.selector_optimizer.step()
            self.expert_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dl)

    def _evaluate_loader(self, loader, desc):
        self.model.eval()
        per_layer_counts = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(self.model.num_layers)]
        with torch.no_grad():
            iterator = tqdm(loader, desc=desc, disable=self.disable_progress)
            for batch in iterator:
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                ner_tags = batch["ner_tags"].to(self.device, non_blocking=True)
                gold_mask = (ner_tags != -100) & (ner_tags > 0) & attention_mask.bool()

                outputs = self.model(embeddings, attention_mask)
                for idx, out in enumerate(outputs):
                    pred_mask = out["selection_mask"].to(torch.bool)
                    tp, fp, fn = counts(pred_mask, gold_mask)
                    per_layer_counts[idx]["tp"] += tp
                    per_layer_counts[idx]["fp"] += fp
                    per_layer_counts[idx]["fn"] += fn
        return per_layer_counts

    def evaluate(self):
        eval_counts = self._evaluate_loader(self.eval_dl, desc="Selector Stack Eval")
        dev_counts = []
        if self.dev_dl is not None:
            dev_counts = self._evaluate_loader(self.dev_dl, desc="Selector Stack Dev")

        metrics = []
        for idx, counts_eval in enumerate(eval_counts):
            tp = counts_eval["tp"]
            fp = counts_eval["fp"]
            fn = counts_eval["fn"]
            eval_f1, eval_p, eval_r = metrics_from_counts(tp, fp, fn)

            dev_f1 = dev_p = dev_r = None
            if self.dev_dl is not None:
                c_dev = dev_counts[idx]
                dev_f1, dev_p, dev_r = metrics_from_counts(
                    c_dev["tp"], c_dev["fp"], c_dev["fn"]
                )

            metrics.append(
                {
                    "layer": idx,
                    "eval_f1": eval_f1,
                    "eval_precision": eval_p,
                    "eval_recall": eval_r,
                    "dev_f1": dev_f1,
                    "dev_precision": dev_p,
                    "dev_recall": dev_r,
                }
            )
        return metrics

    def _save_checkpoint(self):
        state = {
            "model": self.model.state_dict(),
            "selector_opt": self.selector_optimizer.state_dict(),
            "expert_opt": self.expert_optimizer.state_dict(),
            "cfg": self.cfg,
        }
        torch.save(state, "selector_stack.pth", _use_new_zipfile_serialization=False)
        self.logger.info("Saved selector stack checkpoint to %s", os.getcwd())

    def _load_checkpoint(self):
        path = "selector_stack.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)
        if "selector_opt" in state and "expert_opt" in state:
            try:
                self.selector_optimizer.load_state_dict(state["selector_opt"])
                self.expert_optimizer.load_state_dict(state["expert_opt"])
            except Exception:
                pass
        self.logger.info("Loaded selector stack checkpoint from %s", path)

    def train(self):
        for epoch in range(self.num_epochs):
            avg_loss = self._single_epoch(epoch)
            self.logger.info("Epoch %d/%d completed. Avg loss: %.4f", epoch + 1, self.num_epochs, avg_loss)
            if self.cfg.train.validate_epoch:
                metrics = self.evaluate()
                self.logger.info("Epoch %d evaluation metrics:\n%s", epoch + 1, self._metrics_table(metrics))
            self._save_checkpoint()


@hydra_main(config_path="conf", config_name="selector_stack", version_base="1.1")
def main(cfg):
    logger = get_logger("train_selector_stack.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    configure_runtime(cfg)

    if cfg.runtime.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    cfg.runtime.device = device.type

    train_dl, eval_dl, dev_dl = initialize_dataloaders(cfg, logger)
    trainer = SelectorStackTrainer(
        cfg,
        train_dl,
        eval_dl,
        dev_dl,
        logger,
        xp,
        device,
    )

    if cfg.train.eval_only:
        trainer._load_checkpoint()
        metrics = trainer.evaluate()
        table = trainer._metrics_table(metrics)
        logger.info("Evaluation metrics:\n%s", table)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
