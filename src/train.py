import os

import torch
from prettytable import PrettyTable
from tqdm import tqdm

from dora import get_xp, hydra_main

from .utils import load_sbert_pooler, get_logger, counts,\
    configure_runtime, metrics_from_counts, should_disable_tqdm
from .tree import BranchTree
from .data import initialize_dataloaders

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

        self.sel_optim_cfg = self.selector_cfg.optim
        self.exp_optim_cfg = self.expert_cfg.optim
        self.grad_clip = float(cfg.train.grad_clip)

        self.tree = BranchTree(
            self.selector_cfg,
            self.expert_cfg,
            self.device,
            self.num_factors,
            selector_backbone,
            expert_backbone,
        )
        
    def metrics_table(self, metrics, sort_by="eval_f1", reverse=True):
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
            metrics = self.evaluate()
            self.logger.info("Stage %d evaluation metrics:\n%s", stage + 1, self.metrics_table(metrics))
            self._save_checkpoint()
            if stage < self.num_stages - 1:
                self.tree.extend()
        
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
            metrics = self.evaluate()
            self.logger.info("Stage %d evaluation metrics:\n%s", stage + 1, self.metrics_table(metrics))
            total_loss += loss
        return total_loss / num_epochs

    def _single_epoch(self, epoch):
        total_loss = 0.0
        self.tree.set_mode(train=True)

        for batch in tqdm(self.train_dl, f"Branch Composite Train {epoch + 1}", disable=self.disable_progress):
            leaves = self.tree.forward_train(
                batch["embeddings"].to(self.device, non_blocking=True),
                batch["attention_mask"].to(self.device, non_blocking=True)
            )
            for idx, leaf in enumerate(leaves):
                node = leaf["node"]
                loss = self.selector_weight * leaf["selector_loss"] + self.expert_weight * leaf["expert_loss"]
                total_loss += loss.item()

                node.selector_optimizer.zero_grad(set_to_none=True)
                node.expert_optimizer.zero_grad(set_to_none=True)

                loss.backward(retain_graph= idx < len(leaves) - 1)

                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(node.selector.parameters(), self.grad_clip)
                    torch.nn.utils.clip_grad_norm_(node.expert.parameters(), self.grad_clip)

                node.selector_optimizer.step()
                node.expert_optimizer.step()

        return total_loss / len(self.train_dl)
    
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
       
    def evaluate(self):
        branches_eval_counts = self.evaluate_leaves(self.eval_dl, desc="Branch Composite Eval")
        branches_dev_counts = {}
        if self.dev_dl is not None:
            branches_dev_counts = self.evaluate_leaves(self.dev_dl, desc="Branch Composite Dev")

        branch_names = set(branches_eval_counts.keys()) | set(branches_dev_counts.keys())
        
        total_metrics = []
        for name in branch_names:
            eval_counts = branches_eval_counts.get(name, {"tp": 0, "fp": 0, "fn": 0})
            dev_counts = branches_dev_counts.get(name, {"tp": 0, "fp": 0, "fn": 0})

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

            total_metrics.append({
                "name": name,
                "eval_f1": eval_f1,
                "eval_precision": eval_p,
                "eval_recall": eval_r,
                "dev_f1": dev_f1,
                "dev_precision": dev_p,
                "dev_recall": dev_r,
            })

        return total_metrics

    def _save_checkpoint(self):
        state = {"selectors": {}, "experts": {}, "depth": self.tree.depth}
        for node in self.tree.iter_nodes():
            key = BranchTree.path_to_name(node.path)
            state["selectors"][key] = node.selector.state_dict()
            state["experts"][key] = node.expert.state_dict()
        torch.save(state, "branching_composite.pth", _use_new_zipfile_serialization=False)
        self.logger.info("Saved branching composite checkpoints to %s", os.getcwd())

    def _load_checkpoint(self):
        path = "branching_composite.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        state = torch.load(path, map_location=self.device)
        selectors = state["selectors"]
        experts = state["experts"]
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
