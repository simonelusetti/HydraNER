import os
from typing import Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from prettytable import PrettyTable

from dora import get_xp, hydra_main

from .sae import SparseAutoencoder
from .data import initialize_dataloaders
from .utils import (
    get_logger,
    configure_runtime,
    should_disable_tqdm,
    metrics_from_counts,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _flatten_tokens_with_labels(batch,device):
    embeddings = batch["embeddings"].to(device, non_blocking=True)
    mask = batch["attention_mask"].to(device, non_blocking=True).bool()
    if not mask.any():
        return None
    labels_raw = batch["ner_tags"].to(device, non_blocking=True)
    labels_flat = labels_raw[mask]
    entities = (labels_flat > 0).to(torch.bool)  # treat any non-zero tag as entity

    tokens = embeddings[mask]
    return tokens, entities

class SparseAETrainer:
    def __init__(self, cfg, train_dl, eval_dl, dev_dl, logger, xp, device) -> None:
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.xp = xp

        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.dev_dl = dev_dl

        self.grad_clip = float(cfg.sae.grad_clip)
        self.log_interval = int(cfg.train.log_interval)
        self.checkpoint_path = cfg.train.checkpoint
        self.disable_progress = should_disable_tqdm()
        self.run_entity_eval = bool(cfg.entity_eval.run_after_epoch)

        self.model: Optional[SparseAutoencoder] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.global_step = 0

    def _init_model(self, tokens: torch.Tensor):
        if self.model is not None:
            return
        sae_cfg = self.cfg.sae
        d_model = tokens.size(-1)
        self.model = SparseAutoencoder(
            d_model=d_model,
            expansion=sae_cfg.expansion,
            alpha=sae_cfg.alpha,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(sae_cfg.lr),
            weight_decay=float(sae_cfg.weight_decay),
        )
        self.logger.info(
            "Initialized SAE: d_model=%d d_hidden=%d expansion=%.2f alpha=%.6f",
            self.model.d_model,
            self.model.d_hidden,
            self.model.expansion,
            self.model.alpha,
        )

    def _metrics_table(self, metrics: Dict[str, float]) -> PrettyTable:
        table = PrettyTable()
        table.field_names = list(metrics.keys())
        row = [f"{v:.6f}" if isinstance(v, float) else v for v in metrics.values()]
        table.add_row(row)
        return table

    def _train_epoch(self, epoch_idx: int) -> Dict[str, float]:
        if self.model is not None:
            self.model.train()
        totals = {"loss": 0.0, "mse": 0.0, "l1": 0.0}
        token_count = 0

        iterator = tqdm(self.train_dl, desc=f"SAE Train {epoch_idx + 1}", disable=self.disable_progress)
        for batch in iterator:
            flattened = _flatten_tokens_with_labels(batch, self.device)
            if flattened is None:
                continue
            tokens, _ = flattened
            self._init_model(tokens)
            self.model.train()

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(tokens)
            loss = output["loss"]
            loss.backward()

            if self.grad_clip > 0.0:
                clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            self.model.normalize_dictionary()

            num_tokens = tokens.size(0)
            token_count += num_tokens
            totals["loss"] += loss.item() * num_tokens
            totals["mse"] += float(output["mse"].detach()) * num_tokens
            totals["l1"] += float(output["l1"].detach())

            self.global_step += 1

        if token_count == 0:
            raise RuntimeError("No tokens were processed in this epoch; check that the dataset has ner_tags and non-empty batches.")

        return totals["loss"] / token_count, totals["mse"] / token_count, totals["l1"] / token_count
    
    def _save_checkpoint(self):
        if self.model is None:
            raise RuntimeError("Cannot save checkpoint without a model.")
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "meta": {
                "d_model": self.model.d_model,
                "d_hidden": self.model.d_hidden,
                "expansion": self.model.expansion,
                "alpha": self.model.alpha,
            },
        }
        torch.save(state, self.checkpoint_path, _use_new_zipfile_serialization=False)
        self.logger.info("Saved SAE checkpoint to %s", os.path.abspath(self.checkpoint_path))

    def _load_checkpoint(self):
        path = self.checkpoint_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state = torch.load(path, map_location=self.device)
        meta = state.get("meta", {})
        d_model = int(meta.get("d_model"))
        expansion = float(meta.get("expansion", self.cfg.sae.expansion))
        alpha = float(meta.get("alpha", self.cfg.sae.alpha))
        self.model = SparseAutoencoder(d_model=d_model, expansion=expansion, alpha=alpha).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.sae.lr),
            weight_decay=float(self.cfg.sae.weight_decay),
        )
        self.model.load_state_dict(state["model"], strict=True)
        if "optimizer" in state and state["optimizer"] is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        self.logger.info(
            "Loaded SAE checkpoint from %s (d_model=%d d_hidden=%d expansion=%.2f alpha=%.6f)",
            path,
            self.model.d_model,
            self.model.d_hidden,
            self.model.expansion,
            self.model.alpha,
        )

    def _select_entity_dimension(self, loader) -> Optional[int]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        self.model.eval()
        sum_ent = torch.zeros(self.model.d_hidden, device=self.device)
        sum_non = torch.zeros(self.model.d_hidden, device=self.device)
        count_ent = 0
        count_non = 0
        with torch.no_grad():
            iterator = tqdm(loader, desc="Finding entity dimension", disable=self.disable_progress)
            for batch in iterator:
                flattened = _flatten_tokens_with_labels(batch, self.device)
                if flattened is None:
                    continue
                tokens, labels = flattened
                out = self.model(tokens)
                codes = out["codes"]
                ent_mask = labels
                non_mask = ~labels
                if ent_mask.any():
                    sum_ent += codes[ent_mask].sum(dim=0)
                    count_ent += int(ent_mask.sum().item())
                if non_mask.any():
                    sum_non += codes[non_mask].sum(dim=0)
                    count_non += int(non_mask.sum().item())
        if count_ent == 0 or count_non == 0:
            self.logger.warning("Unable to select entity dimension (missing entities or non-entities).")
            return None
        mean_ent = sum_ent / max(1, count_ent)
        mean_non = sum_non / max(1, count_non)
        diff = mean_ent - mean_non
        best_dim = int(torch.argmax(diff).item())
        self.logger.info(
            "Selected entity dimension %d (mean_ent=%.6f mean_non=%.6f diff=%.6f)",
            best_dim,
            mean_ent[best_dim].item(),
            mean_non[best_dim].item(),
            diff[best_dim].item(),
        )
        return best_dim

    def evaluate(self) -> Dict[str, float]:
        best_dim = self._select_entity_dimension(self.eval_dl)
        if best_dim is None:
            return {"dimension": -1, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        thresh = float(self.cfg.entity_eval.threshold)
        self.model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            iterator = tqdm(self.dev_dl, desc="Entity dim eval", disable=self.disable_progress)
            for batch in iterator:
                flattened = _flatten_tokens_with_labels(batch, self.device)
                if flattened is None:
                    continue
                tokens, labels = flattened
                codes = self.model(tokens)["codes"]
                preds = (codes[:, best_dim] > thresh)
                gold = labels
                tp += int((preds & gold).sum().item())
                fp += int((preds & ~gold).sum().item())
                fn += int((~preds & gold).sum().item())
        f1, precision, recall = metrics_from_counts(tp, fp, fn)
        return best_dim, f1 , precision, recall

    def train(self):
        for epoch in range(int(self.cfg.train.epochs)):
            loss, mse, l1 = self._train_epoch(epoch)
            self.logger.info(
                "Epoch %d/%d train metrics:\n%s",
                epoch + 1,
                self.cfg.train.epochs,
                self._metrics_table({"loss": loss, "mse": mse, "l1": l1}),
            )
            if self.run_entity_eval and (self.dev_dl is not None or self.eval_dl is not None):
                best_dim, f1, precision, recall = self.evaluate()
                self.logger.info("Epoch %d entity dimension=%s", epoch + 1, best_dim)
                self.logger.info("Epoch %d entity metrics:\n%s", epoch + 1, self._metrics_table({"f1": f1, "precision": precision, "recall": recall}))
            self._save_checkpoint()


@hydra_main(config_path="conf", config_name="sparse_autoencoder", version_base="1.1")
def main(cfg):
    logger = get_logger("train_sae.log")
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
    trainer = SparseAETrainer(cfg, train_dl, eval_dl, dev_dl, logger, xp, device)

    if cfg.train.eval_only:
        trainer._load_checkpoint()
        metrics = trainer.evaluate(eval_dl, desc="SAE Eval")
        logger.info("Evaluation metrics:\n%s", trainer._metrics_table(metrics))
        best_dim, f1, precision, recall = trainer.evaluate()
        logger.info(
            "Entity dimension=%s",
            best_dim
        )
        logger.info("Entity metrics:\n%s", trainer._metrics_table({"f1": f1, "precision": precision, "recall": recall}))
    else:
        trainer.train()


if __name__ == "__main__":
    main()
