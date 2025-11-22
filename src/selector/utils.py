import copy
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
def should_disable_tqdm():
    """Return True when tqdm progress bars should be disabled."""
    override = os.environ.get("RATCON_DISABLE_TQDM")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}

    try:
        return not sys.stderr.isatty()
    except Exception:
        return True


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:  # pragma: no cover - logging fallback
            self.handleError(record)


def get_logger(logfile="train.log"):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch_format = "%(asctime)s - %(levelname)s - %(message)s"
    ch.setFormatter(logging.Formatter(ch_format))

    fh = logging.FileHandler(Path(logfile))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# -------------------------------------------------------------------
# Losses
# -------------------------------------------------------------------
def sparsity_loss(gates, mask):
    # average gate value over valid tokens only (mask==1)
    valid = (mask > 0).float()
    return (gates * valid).sum() / (valid.sum() + 1e-8)


def total_variation_1d(gates, mask):
    # penalize changes across adjacent valid tokens
    valid = (mask > 0).float()
    diff = torch.abs(gates[:, 1:] - gates[:, :-1])
    # only count where both tokens are valid
    both = valid[:, 1:] * valid[:, :-1]
    return (diff * both).sum() / (both.sum() + 1e-8)


def complement_margin_loss(h_anchor, h_comp, margin=0.3):
    # we want cosine(anchor, comp) to be LOW -> (1 - cos) to be HIGH
    cos = (h_anchor * h_comp).sum(dim=-1)            # [B]
    neg = 1.0 - cos
    return torch.relu(margin - neg).mean()


def nt_xent(anchor, positive, temperature=0.07):
    """
    InfoNCE with in-batch negatives: anchors vs. positives (one-to-one).
    """
    temperature = max(float(temperature), 1e-3)
    B = anchor.size(0)
    logits = anchor @ positive.t() / temperature              # [B,B]
    labels = torch.arange(B, device=anchor.device)
    return F.cross_entropy(logits, labels)


def complement_loss(h_comp, h_anchor, temperature=0.07):
    """Repel complements from anchors (no null embedding target)."""
    return -nt_xent(h_comp, h_anchor, temperature=temperature)


def compute_training_objectives(
    output,
    attention_mask,
    model_cfg,
    *,
    temperature,
):
    """Compute total loss for single-model training."""
    anchors = output["h_anchor"]
    gates = output["gates"]

    l_rat = nt_xent(output["h_rat"], anchors, temperature=temperature)
    l_comp = complement_loss(output["h_comp"], anchors, temperature=temperature)
    l_s = sparsity_loss(gates, attention_mask)
    l_tv = total_variation_1d(gates, attention_mask)

    loss_cfg = model_cfg.loss
    loss = l_rat
    loss = loss + float(loss_cfg.l_comp) * l_comp
    loss = loss + float(loss_cfg.l_s) * l_s
    loss = loss + float(loss_cfg.l_tv) * l_tv

    return loss

# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------
def format_gold_spans(ids, tokens, gold_labels, tokenizer):
    buf = ""
    buf_labels = []
    words, word_labels = [], []
    for tok_id, tok_str, lab in zip(ids, tokens, gold_labels):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_labels.append(lab)
        else:
            if buf:
                words.append(buf)
                word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
            buf = tok_str
            buf_labels = [lab]
    if buf:
        words.append(buf)
        word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
    out, span = [], []
    for w, l in zip(words, word_labels):
        if l:
            span.append(w)
        else:
            if span:
                out.append(f"[[{' '.join(span)}]]")
                span = []
            out.append(w)
    if span:
        out.append(f"[[{' '.join(span)}]]")
    return " ".join(out)


def merge_subwords(ids, tokens, tokenizer):
    buf = ""
    words = []

    def flush(acc):
        if acc:
            words.append(acc)
        return ""

    for tok_id, tok_str in zip(ids, tokens):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
        else:
            buf = flush(buf)
            buf = tok_str

    buf = flush(buf)
    return words


def merge_spans(ids, tokens, gates, tokenizer, thresh=0.5):
    buf, buf_gs = "", []
    words, word_gates = [], []

    def flush(acc, gs):
        if acc:
            words.append(acc)
            word_gates.append(sum(gs) / len(gs))
        return "", []

    for tok_id, tok_str, g in zip(ids, tokens, gates):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_gs.append(g)
        else:
            buf, buf_gs = flush(buf, buf_gs)
            buf, buf_gs = tok_str, [g]

    buf, buf_gs = flush(buf, buf_gs)

    out_tokens, span_buf = [], []

    def flush_span(span_buf):
        if span_buf:
            out_tokens.append(f"[[{' '.join(span_buf)}]]")
        return []

    for word, g in zip(words, word_gates):
        if g >= thresh:
            span_buf.append(word)
        else:
            span_buf = flush_span(span_buf)
            out_tokens.append(word)

    span_buf = flush_span(span_buf)
    return " ".join(out_tokens)


# -------------------------------------------------------------------
# Misc
# -------------------------------------------------------------------
def load_sbert_pooler(model_name: str, device: torch.device | None = None):
    """Load a SentenceTransformer pooler and hidden size for reuse."""
    base = SentenceTransformer(model_name)
    encoder = base[0]
    if device is not None:
        encoder = encoder.to(device)
    pooler = copy.deepcopy(base[1]).to(device) if device is not None else copy.deepcopy(base[1])
    hidden_dim = encoder.auto_model.config.hidden_size
    return pooler, hidden_dim


def prepare_batch(batch, device, *, require_labels: bool = False, labels_key: str = "ner_tags"):
    """Move batch tensors to device and optionally require labels."""
    embeddings = batch["embeddings"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch.get(labels_key)
    if labels is None and require_labels:
        return embeddings, attention_mask, None
    if labels is not None:
        labels = labels.to(device, non_blocking=True)
    return embeddings, attention_mask, labels

def configure_runtime(cfg):
    runtime = cfg.runtime
    num_threads = runtime.num_threads
    if not num_threads:
        return
    try:
        num_threads = int(num_threads)
    except (TypeError, ValueError):
        return
    if num_threads <= 0:
        return
    value = str(num_threads)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = value
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except Exception:
        pass