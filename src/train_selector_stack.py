import os
import copy

import torch
from dora import get_xp, hydra_main
from torch.optim import AdamW
from tqdm import tqdm
from prettytable import PrettyTable

from .data import initialize_dataloaders
from .selector.models import RationaleSelectorModel
from .utils import (
    compute_training_objectives,
    configure_runtime,
    get_logger,
    load_sbert_pooler,
    metrics_from_counts,
    should_disable_tqdm,
    counts,
)

class SelectorStackTrainer:
    def __init__(self, cfg, train_dl, eval_dl, logger, device, xp) -> None:
        self.cfg = cfg
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.logger = logger
        self.device = device
        self.xp = xp
        self.disable_progress = should_disable_tqdm()

        self.selector_cfg = cfg.selector_model
        self.selector_cfg.threshold = float(self.selector_cfg.threshold)

        self.grad_clip = float(cfg.train.grad_clip)
        self.epochs_per_selector = int(cfg.train.epochs_per_selector)
        self.num_selectors = int(cfg.train.num_selectors)

        selector_backbone = load_sbert_pooler(self.selector_cfg.sbert_name, device=self.device)
        self.selector_pooler, self.selector_hidden_dim = selector_backbone

        self.selectors: list[RationaleSelectorModel] = []
        
    def metrics_table(self, metrics):
        table = PrettyTable()
        table.field_names = metrics[0].keys()
        for row in metrics:
            formatted_row = [
                f"{v:.5f}" if isinstance(v, float) else v
                for v in row.values()
            ]
            table.add_row(formatted_row)
        return table

    def _build_selection_mask(self, gates: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_float = attention_mask.to(dtype=gates.dtype)
        selection = (gates >= self.selector_cfg.threshold).to(dtype=gates.dtype) * mask_float
        selected_counts = selection.sum(dim=1)
        need_fallback = selected_counts == 0
        if need_fallback.any():
            masked_gates = gates.masked_fill(mask_float == 0, -1e9)
            top_indices = masked_gates.argmax(dim=1)
            rows = torch.arange(gates.size(0), device=gates.device)
            selection[rows[need_fallback], top_indices[need_fallback]] = 1.0
            selection = selection * mask_float
        return selection

    def _scheduled_selector_cfg(self, selector_idx: int):
        """Return a selector cfg with loss weights scheduled for the given depth."""
        cfg = copy.deepcopy(self.selector_cfg)
        cfg.loss.l_comp = float(cfg.loss.l_comp)
        cfg.loss.l_s = float(cfg.loss.l_s) * (2 ** selector_idx)
        cfg.loss.l_tv = float(cfg.loss.l_tv)
        cfg.loss.tau = float(cfg.loss.tau)
        return cfg
    
    def _save_checkpoint(self, selector_idx: int):
        state = {
            "selectors": [sel.state_dict() for sel in self.selectors],
            "selector_cfg": self.selector_cfg,
            "num_trained": selector_idx + 1,
        }
        path_latest = "selector_stack.pth"
        path_epoch = f"selector_stack_epoch{selector_idx+1}.pth"
        torch.save(state, path_latest, _use_new_zipfile_serialization=False)
        torch.save(state, path_epoch, _use_new_zipfile_serialization=False)
        self.logger.info(
            "Saved selector stack checkpoint at depth %d to %s (and %s)",
            selector_idx + 1,
            os.path.join(os.getcwd(), path_latest),
            os.path.join(os.getcwd(), path_epoch),
        )

    def _load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state = torch.load(path, map_location=self.device)
        selector_states = state.get("selectors", [])
        self.selectors = []
        for sel_state in selector_states:
            selector = RationaleSelectorModel(
                self.selector_cfg,
                pooler=self.selector_pooler,
                embedding_dim=self.selector_hidden_dim,
            ).to(self.device)
            selector.load_state_dict(sel_state, strict=False)
            selector.eval()
            for p in selector.parameters():
                p.requires_grad_(False)
            self.selectors.append(selector)
        self.logger.info("Loaded %d selectors from %s", len(self.selectors), path)

    def _tail(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        selectors: list[RationaleSelectorModel] | None = None,
    ):
        selectors = selectors if selectors is not None else self.selectors
        mask = attention_mask
        selected_embeddings = embeddings
        for selector in selectors:
            with torch.no_grad():
                outputs = selector(selected_embeddings, mask)
                gates = outputs["gates"]
                selection_mask = self._build_selection_mask(gates, mask)
                selected_embeddings = selected_embeddings * selection_mask.unsqueeze(-1)
                mask = (mask * selection_mask.long()).clamp(max=1)
        return selected_embeddings, mask

    def _train_head(self):
        scheduled_cfg = self._scheduled_selector_cfg(len(self.selectors))

        selector = RationaleSelectorModel(
            self.selector_cfg,
            pooler=self.selector_pooler,
            embedding_dim=self.selector_hidden_dim,
        ).to(self.device)
        
        optimizer = AdamW(
            selector.parameters(),
            lr=float(self.selector_cfg.optim.lr),
            weight_decay=float(self.selector_cfg.optim.weight_decay),
            betas=tuple(self.selector_cfg.optim.betas),
        )

        selector.train()
        for epoch in range(self.epochs_per_selector):
            total_loss = 0.0
            iterator = tqdm(self.train_dl, desc=f"Selector {len(self.selectors)+1} Epoch {epoch+1}", disable=self.disable_progress)
            for batch in iterator:
                base_embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                tail_embeddings, tail_mask = self._tail(base_embeddings, attention_mask)

                output = selector(tail_embeddings, tail_mask)
                loss = compute_training_objectives(
                    output,
                    tail_mask,
                    scheduled_cfg,
                    temperature=float(scheduled_cfg.loss.tau),
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(selector.parameters(), self.grad_clip)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dl)
            self.logger.info(
                "Selector %d epoch %d/%d avg loss: %.4f",
                len(self.selectors  ) + 1,
                epoch + 1,
                self.epochs_per_selector,
                avg_loss,
            )

            # Evaluate the current stack (including this selector) at the end of each epoch
            selector.eval()
            metrics = self.evaluate(selectors=self.selectors + [selector])
            self.logger.info(
                "Selector %d epoch %d/%d eval metrics:\n%s",
                len(self.selectors) + 1,
                epoch + 1,
                self.epochs_per_selector,
                self.metrics_table(metrics),
            )
            selector.train()

        for param in selector.parameters():
            param.requires_grad_(False)
        selector.eval()
        self.selectors.append(selector)

    def train(self):
        for idx in range(self.num_selectors):
            self.logger.info("Training selector %d/%d", idx + 1, self.num_selectors)
            self._train_head()
            self._save_checkpoint(idx)

        self.logger.info("Training complete. Running final evaluation.")
        metrics = self.evaluate()
        self.logger.info("Final selector stack metrics:\n%s", self.metrics_table(metrics))

    def evaluate(self, selectors: list[RationaleSelectorModel] | None = None):
        selectors = selectors if selectors is not None else self.selectors
        for selector in selectors:
            selector.eval()
        total_tp = total_fp = total_fn = 0
        total_valid_tokens = 0
        total_kept_tokens = 0
        with torch.no_grad():
            iterator = tqdm(self.eval_dl, desc="Selector Stack Eval", disable=self.disable_progress)
            for batch in iterator:
                base_embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                ner_tags = batch["ner_tags"].to(self.device, non_blocking=True)

                _, final_mask = self._tail(base_embeddings, attention_mask, selectors=selectors)
                pred_mask = final_mask.bool()
                valid_mask = attention_mask.bool()
                gold_mask = (ner_tags != -100) & (ner_tags > 0) & valid_mask

                total_valid_tokens += valid_mask.sum().item()
                total_kept_tokens += (pred_mask & valid_mask).sum().item()

                tp, fp, fn = counts(pred_mask, gold_mask)
                total_tp += tp
                total_fp += fp
                total_fn += fn

        f1, precision, recall = metrics_from_counts(total_tp, total_fp, total_fn)
        survival_rate = total_kept_tokens / total_valid_tokens if total_valid_tokens > 0 else 0.0
        return [{
            "depth": len(selectors),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "survival_rate": survival_rate,
        }]


@hydra_main(config_path="conf", config_name="selector_stack", version_base="1.1")
def main(cfg):
    logger = get_logger("train_selector_stack.log")
    xp = get_xp()
    logger.info(f"Exp signature: {xp.sig}")
    logger.info(repr(cfg))
    logger.info(f"Work dir: {os.getcwd()}")

    configure_runtime(cfg)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, using CPU.")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = device.type

    train_dl, eval_dl, _ = initialize_dataloaders(cfg, logger)
    trainer = SelectorStackTrainer(cfg, train_dl, eval_dl, logger, device, xp)

    if cfg.eval.eval_only:
        trainer._load_checkpoint("selector_stack.pth")
        metrics = trainer.evaluate()
        logger.info("Evaluation metrics:\n%s", trainer.metrics_table(metrics))
    else:
        trainer.train()


if __name__ == "__main__":
    main()
