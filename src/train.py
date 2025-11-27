import os
from collections import defaultdict
from math import ceil
from math import ceil

import torch
from prettytable import PrettyTable
from tqdm import tqdm

from dora import get_xp, hydra_main

from .utils import load_sbert_pooler, get_logger, counts,\
    configure_runtime, metrics_from_counts, should_disable_tqdm
from .tree import BranchTree
from .data import initialize_dataloaders
from . import diagnostics
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class BranchingCompositeTrainer:
    def __init__(
            self,
            cfg, 
            train_dl,
            eval_dl,
            dev_dl,
            logger,
            xp,
            device
        ):
        self.disable_progress = should_disable_tqdm()
        
        self.cfg = cfg
        self.device = device
        
        self.selector_cfg = cfg.selector_model
        self.expert_cfg = cfg.expert_model
        selector_backbone = load_sbert_pooler(self.selector_cfg.encoder_name)
        expert_backbone = load_sbert_pooler(self.expert_cfg.encoder_name)
        
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.dev_dl = dev_dl
        self.logger = logger
        self.xp = xp

        self.num_factors = int(self.expert_cfg.expert.num_experts)

        self.selector_weight = float(cfg.train.loss_weights.selector)
        self.expert_weight = float(cfg.train.loss_weights.expert)
         
        self.stage_epochs = [int(x) for x in cfg.train.stages_epochs]
        self.num_stages = len(self.stage_epochs)
        self.validate_epoch = int(cfg.train.validate_epoch)

        self.sel_optim_cfg = self.selector_cfg.optim
        self.exp_optim_cfg = self.expert_cfg.optim
        self.grad_clip = float(cfg.train.grad_clip)
        pct = float(cfg.train.group_random_pct)
        if pct > 1.0:
            pct = pct / 100.0
        self.group_random_pct = max(0.0, pct)
        self.group_random_trials = int(cfg.train.group_random_trials)
        self.group_union_enabled = bool(cfg.train.group_union_enabled)

        diag_cfg = cfg.diagnostics
        self.diag_interval = None if diag_cfg.interval_steps is None else int(diag_cfg.interval_steps)
        self.diag_enabled = self.diag_interval is not None
        self.diag_max_leaves = int(diag_cfg.max_leaves)
        self.diag_skip_eval = bool(diag_cfg.skip_eval)
        tb_dir = diag_cfg.tensorboard_dir or ""
        self.tb_writer = None
        if self.diag_enabled and SummaryWriter is not None:
            log_dir = tb_dir if tb_dir else os.path.join(os.getcwd(), "tb")
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.diag_trackers = {}
        self.global_step = 0

        self.tree = BranchTree(
            self.selector_cfg,
            self.expert_cfg,
            self.device,
            self.num_factors,
            selector_backbone,
            expert_backbone,
        )
        
    def metrics_table(self, metrics, sort_by="eval_f1", reverse=True):
        if not metrics:
            return PrettyTable()
        metrics = sorted(metrics, key=lambda x: x[sort_by], reverse=reverse)

        table = PrettyTable()
        table.field_names = metrics[0].keys()

        for row in metrics:
            formatted_row = [
                f"{v:.5f}" if isinstance(v, float) else v
                for v in row.values()
            ]
            table.add_row(formatted_row)

        return table

            
    def train(self):
        self.tree.set_mode(train=True)
        for stage, num_epochs in enumerate(self.stage_epochs):
            self.logger.info("Training stage %d/%d", stage + 1, self.num_stages)
            avg_loss = self._train_stage(num_epochs, stage)
            self.logger.info(
                "Stage %d training completed. Avg loss: %.4f",
                stage + 1,
                avg_loss,
            )
            self._save_checkpoint()
            if stage < self.num_stages - 1:
                self.tree.extend()
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
        
    def _train_stage(self, num_epochs, stage):
        total_loss = 0.0
        for epoch in range(num_epochs):
            loss = self._single_epoch(epoch)
            self.logger.info(
                "Stage %d Epoch %d/%d completed. Avg loss: %.4f",
                stage + 1,
                epoch + 1,
                num_epochs,
                loss,
            )
            if self.validate_epoch or epoch == num_epochs - 1:
                metrics = self.evaluate()
                self.logger.info("Stage %d evaluation metrics:\n%s", stage + 1, self.metrics_table(metrics))
            total_loss += loss
        return total_loss / num_epochs

    def _single_epoch(self, epoch):
        total_loss = 0.0
        loss_sums = defaultdict(float)
        loss_counts = defaultdict(int)
        self.tree.set_mode(train=True)

        for batch in tqdm(self.train_dl, f"Branch Composite Train {epoch + 1}", disable=self.disable_progress):
            self.global_step += 1
            want_diag = self.diag_enabled and self.diag_interval and self.diag_interval > 0 and (self.global_step % self.diag_interval == 0)

            leaves = self.tree.forward_train(
                batch["embeddings"].to(self.device, non_blocking=True),
                batch["attention_mask"].to(self.device, non_blocking=True),
                return_debug=want_diag,
            )
            for idx, leaf in enumerate(leaves):
                node = leaf["node"]
                loss = self.selector_weight * leaf["selector_loss"] + self.expert_weight * leaf["expert_loss"]
                total_loss += loss.item()
                loss_sums["total"] += loss.item()
                loss_counts["total"] += 1
                loss_sums["selector_loss"] += float(leaf["selector_loss"].detach())
                loss_counts["selector_loss"] += 1
                loss_sums["expert_loss"] += float(leaf["expert_loss"].detach())
                loss_counts["expert_loss"] += 1

                sel_comp = leaf["selector_losses"]
                for key, val in sel_comp.items():
                    loss_sums[f"selector/{key}"] += float(val)
                    loss_counts[f"selector/{key}"] += 1
                exp_comp = leaf["expert_losses"]
                for key, val in exp_comp.items():
                    loss_sums[f"expert/{key}"] += float(val)
                    loss_counts[f"expert/{key}"] += 1

                node.selector_optimizer.zero_grad(set_to_none=True)
                node.expert_optimizer.zero_grad(set_to_none=True)

                loss.backward(retain_graph= idx < len(leaves) - 1)

                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(node.selector.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(node.expert.parameters(), self.grad_clip)

                node.selector_optimizer.step()
                node.expert_optimizer.step()

            if want_diag:
                self._run_diagnostics(batch, leaves)

        if loss_counts:
            avg_parts = {k: loss_sums[k] / max(1, loss_counts[k]) for k in loss_sums}
            table = PrettyTable()
            table.field_names = ["loss", "avg"]
            for key in sorted(avg_parts.keys()):
                table.add_row([key, f"{avg_parts[key]:.6f}"])
            self.logger.info("Epoch %d loss breakdown:\n%s", epoch + 1, table)

        return total_loss / len(self.train_dl)

    def _run_diagnostics(self, batch, leaves):
        """Run diagnostics on a subset of leaves."""
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        for leaf in leaves[: self.diag_max_leaves]:
            debug = leaf["debug"] if "debug" in leaf else None
            if debug is None:
                continue
            name = BranchTree.path_to_name(leaf["node"].path)
            tracker = self.diag_trackers[name] if name in self.diag_trackers else None
            if tracker is None:
                tracker = diagnostics.ExpertTracker()
                self.diag_trackers[name] = tracker

            pi = debug["pi"]
            factors = debug["factors"]
            anchor = debug["anchor"]
            recon = debug["reconstruction"]
            selected_mask = debug["selected_mask"] if "selected_mask" in debug else attention_mask

            # use the first sample for expert/alignment metrics; routing keeps batch
            sample_z = factors.mean(dim=0)
            sample_h = recon[0]
            sample_h_sbert = anchor[0]
            report = diagnostics.debug_report(
                pi=pi,
                z=sample_z,
                h=sample_h,
                h_sbert=sample_h_sbert,
                attention_mask=selected_mask,
                tracker=tracker,
                writer=self.tb_writer,
                step=self.global_step,
            )
            self.global_step = 0
            if "warnings" in report and report["warnings"]:
                for w in report["warnings"]:
                    self.logger.warning("[diag %s] %s", name, w)
    
    def evaluate_leaves(self, loader, desc="Branch Composite Eval"):
        self.tree.set_mode(train=False)
        branches_counts = {}
        with torch.no_grad():
            iterator = tqdm(loader, desc=desc, disable=self.disable_progress)
            for batch in iterator:
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                embeddings = batch["embeddings"].to(self.device, non_blocking=True)

                ner_tags = batch["ner_tags"].to(self.device, non_blocking=True)
                gold_mask = (ner_tags != -100) & (ner_tags > 0) & attention_mask.bool()

                branches_pred = self.tree.forward_eval(
                    embeddings,
                    attention_mask,
                )
                for (name, pred) in branches_pred:
                    if name not in branches_counts:
                        branches_counts[name] = {"tp": 0, "fp": 0, "fn": 0}
                    if gold_mask is None:
                        continue
                    tp, fp, fn = counts(pred.to(torch.bool), gold_mask)
                    branches_counts[name]["tp"] += tp
                    branches_counts[name]["fp"] += fp
                    branches_counts[name]["fn"] += fn
        return branches_counts

    def evaluate_union(self, loader, selected_names, desc="Branch Composite Union", disable_progress=False):
        self.tree.set_mode(train=False)
        total_tp, total_fp, total_fn = 0, 0, 0
        disable_progress = disable_progress or self.disable_progress
        with torch.no_grad():
            iterator = tqdm(loader, desc=desc, disable=disable_progress)
            for batch in iterator:
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                embeddings = batch["embeddings"].to(self.device, non_blocking=True)
                ner_tags = batch["ner_tags"].to(self.device, non_blocking=True)
                gold_mask = (ner_tags != -100) & (ner_tags > 0) & attention_mask.bool()

                branches_pred = self.tree.forward_eval(
                    embeddings,
                    attention_mask,
                )
                union_pred = None
                for (name, pred) in branches_pred:
                    if name not in selected_names:
                        continue
                    mask = pred.to(torch.bool)
                    union_pred = mask if union_pred is None else (union_pred | mask)
                if union_pred is None:
                    union_pred = torch.zeros_like(gold_mask, dtype=torch.bool)
                tp, fp, fn = counts(union_pred, gold_mask)
                total_tp += tp
                total_fp += fp
                total_fn += fn
        return total_tp, total_fp, total_fn
       
    def evaluate(self):
        if self.diag_enabled and self.diag_skip_eval:
            self.logger.info("Diagnostics enabled; skipping evaluation metrics.")
            return []
        branches_eval_counts = self.evaluate_leaves(self.eval_dl, desc="Branch Composite Eval")
        branches_dev_counts = {}
        if self.dev_dl is not None:
            branches_dev_counts = self.evaluate_leaves(self.dev_dl, desc="Branch Composite Dev")

        branch_names = set(branches_eval_counts.keys()) | set(branches_dev_counts.keys())

        per_branch = []
        for name in branch_names:
            eval_counts = branches_eval_counts[name] if name in branches_eval_counts else {"tp": 0, "fp": 0, "fn": 0}
            dev_counts = branches_dev_counts[name] if name in branches_dev_counts else {"tp": 0, "fp": 0, "fn": 0}

            tp = eval_counts["tp"]
            fp = eval_counts["fp"]
            fn = eval_counts["fn"]
            eval_f1, eval_p, eval_r = metrics_from_counts(tp, fp, fn)

            if self.dev_dl is not None:
                dev_f1, dev_p, dev_r = metrics_from_counts(
                    dev_counts["tp"],
                    dev_counts["fp"],
                    dev_counts["fn"],
                )
            else:
                dev_f1 = dev_p = dev_r = None

            per_branch.append({
                "name": name,
                "eval_f1": eval_f1,
                "eval_precision": eval_p,
                "eval_recall": eval_r,
                "dev_f1": dev_f1,
                "dev_precision": dev_p,
                "dev_recall": dev_r,
            })

        total_metrics = []
        if (
            self.group_union_enabled
            and self.group_random_pct > 0.0
            and per_branch
            and self.dev_dl is not None
        ):
            k = max(1, min(len(per_branch), ceil(self.group_random_pct * len(per_branch))))
            # higher eval_f1 rank => higher weight (linear with reversed rank)
            sorted_by_f1 = sorted(per_branch, key=lambda x: x["eval_f1"], reverse=True)
            weights = [len(sorted_by_f1) - idx for idx, _ in enumerate(sorted_by_f1)]
            weight_tensor = torch.tensor(weights, dtype=torch.float)
            branch_order = [row["name"] for row in sorted_by_f1]

            trials = max(1, self.group_random_trials)
            for trial in tqdm(range(trials), disable=self.disable_progress, desc="Branch Composite Random Group Eval"):
                # sample without replacement, weighted by rank
                sampled_indices = torch.multinomial(weight_tensor, num_samples=k, replacement=False)
                selected_names = [branch_order[idx] for idx in sampled_indices.tolist()]

                tp_eval_union, fp_eval_union, fn_eval_union = self.evaluate_union(
                    self.eval_dl, selected_names, desc=f"Branch Composite Union Eval rand trial {trial + 1}", disable_progress=True
                )
                tp_dev_union, fp_dev_union, fn_dev_union = self.evaluate_union(
                    self.dev_dl, selected_names, desc=f"Branch Composite Union Dev rand trial {trial + 1}", disable_progress=True
                )

                eval_union_f1, eval_union_p, eval_union_r = metrics_from_counts(
                    tp_eval_union,
                    fp_eval_union,
                    fn_eval_union,
                ) if tp_eval_union is not None else (0.0, 0.0, 0.0)
                dev_union_f1, dev_union_p, dev_union_r = metrics_from_counts(
                    tp_dev_union,
                    fp_dev_union,
                    fn_dev_union,
                ) if tp_dev_union is not None else (0.0, 0.0, 0.0)

                pct_display = int(round(self.group_random_pct * 100))
                total_metrics.append({
                    "name": f"group_rand_{pct_display}pct_k{k}_trial{trial + 1}",
                    "eval_f1": eval_union_f1,
                    "eval_precision": eval_union_p,
                    "eval_recall": eval_union_r,
                    "dev_f1": dev_union_f1,
                    "dev_precision": dev_union_p,
                    "dev_recall": dev_union_r,
                })

        if not total_metrics:
            total_metrics = per_branch

        return total_metrics

    def _save_checkpoint(self):
        state = {"selectors": {}, "experts": {}, "prototypes": {}, "depth": self.tree.depth}
        for node in self.tree.iter_nodes():
            key = BranchTree.path_to_name(node.path)
            state["selectors"][key] = node.selector.state_dict()
            state["experts"][key] = node.expert.state_dict()
            if node.prototype is not None:
                state["prototypes"][key] = node.prototype.detach().cpu()
        torch.save(state, "branching_composite.pth", _use_new_zipfile_serialization=False)
        self.logger.info("Saved branching composite checkpoints to %s", os.getcwd())

    def _load_checkpoint(self):
        path = "branching_composite.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state = torch.load(path, map_location=self.device)
        selectors = state["selectors"]
        experts = state["experts"]
        prototypes = state["prototypes"] if "prototypes" in state else {}
        depth = int(state["depth"])
        if not depth and selectors:
            depth = max(len(BranchTree.name_to_path(name)) for name in selectors.keys())
        self.tree.build_tree_to_depth(depth)
        for node in self.tree.iter_nodes():
            key = BranchTree.path_to_name(node.path)
            if key in selectors:
                node.selector.load_state_dict(selectors[key], strict=False)
            if key in experts:
                node.expert.load_state_dict(experts[key], strict=False)
            if node.prototype is not None and key in prototypes:
                proto = prototypes[key].to(self.device)
                if proto.shape == node.prototype.shape:
                    node.prototype.copy_(proto)
        self.logger.info("Loaded branching composite checkpoints from %s", path)


@hydra_main(config_path="conf", config_name="composite", version_base="1.1")
def main(cfg):
    logger = get_logger("train_composite_branching.log")
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
    trainer = BranchingCompositeTrainer(
        cfg, 
        train_dl,
        eval_dl,
        dev_dl,
        logger,
        xp,
        device
    )

    if cfg.train.eval_only:
        trainer._load_checkpoint()
        metrics = trainer.evaluate()
        table = trainer.metrics_table(metrics)
        logger.info("Evaluation metrics:\n%s", table)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
