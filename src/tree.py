import copy

from attr import dataclass
from prettytable import PrettyTable
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .selector.models import RationaleSelectorModel
from .expert.models import ExpertModel

from .utils import (
    metrics_from_counts,
    nt_xent,
    complement_loss,
    load_sbert_pooler,
    sparsity_loss as rat_sparsity_loss,
    total_variation_1d,
)


class BranchNode:
    def __init__(
        self,
        path: tuple[int, ...],
        selector_hidden_dim: int,
        selector_pooler,
        selector_cfg,
        expert_hidden_dim: int,
        expert_pooler,
        expert_cfg,
        device: torch.device,
    ) -> None:
        self.path = path
        self.selector_cfg = selector_cfg
        self.expert_cfg = expert_cfg
        self.children: list["BranchNode"] = []
        weights_cfg = expert_cfg.loss_weights
        self.expert_weights = {
            "sent": float(weights_cfg.sent),
            "token": float(weights_cfg.token),
            "entropy": float(weights_cfg.entropy),
            "overlap": float(weights_cfg.overlap),
            "diversity": float(weights_cfg.diversity),
            "balance": float(weights_cfg.balance),
        }
        if getattr(expert_cfg.expert, "use_continuity", False):
            self.expert_weights["continuity"] = float(getattr(weights_cfg, "continuity", 0.0))
        self.contrastive_tau = float(expert_cfg.contrastive_tau)

        self.selector = RationaleSelectorModel(
            self.selector_cfg,
            pooler=selector_pooler,
            embedding_dim=selector_hidden_dim,
        ).to(device)

        self.expert = ExpertModel(
            self.expert_cfg,
            pooler=expert_pooler,
            embedding_dim=expert_hidden_dim,
        ).to(device)

    def no_grad(self):
        for param in self.selector.parameters():
            param.requires_grad_(False)
        for param in self.expert.parameters():
            param.requires_grad_(False)

    def path_to_name(self) -> str:
        return BranchTree.path_to_name(self.path)

    def _selector_forward(self, selector, embeddings, mask):
        outputs = selector(embeddings, mask)

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

    def _expert_forward(self, expert, embeddings, mask):
        outputs = expert(embeddings, mask)
        routing_weights = outputs["pi"]
        anchor = outputs["anchor"]
        reconstruction = outputs["reconstruction"]

        anchor = F.normalize(anchor, dim=-1)
        reconstruction = F.normalize(reconstruction, dim=-1)

        logits = anchor @ reconstruction.t() / max(self.contrastive_tau, 1e-6)
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, targets)
        loss_ba = F.cross_entropy(logits.t(), targets)
        sent_loss = 0.5 * (loss_ab + loss_ba)

        token_reconstruction = outputs.get("token_reconstruction")
        if token_reconstruction is not None:
            mask_float_tok = mask.unsqueeze(-1).to(dtype=token_reconstruction.dtype)
            diff = token_reconstruction - embeddings
            token_loss = (diff.pow(2) * mask_float_tok).sum() / mask_float_tok.sum().clamp_min(1.0)
        else:
            token_loss = embeddings.new_tensor(0.0)

        mask_float = mask.to(dtype=routing_weights.dtype)

        entropy = -(routing_weights.clamp_min(expert.small_value).log() * routing_weights)
        entropy = (entropy.sum(dim=-1) * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        entropy_loss = entropy.mean()

        pi_sq = (routing_weights ** 2).sum(dim=-1)
        overlap = 0.5 * (1.0 - pi_sq)
        overlap = (overlap * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
        overlap_loss = overlap.mean()

        if expert.use_balance:
            expert_mass = routing_weights.sum(dim=1)
            total_tokens = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            balanced_mass = expert_mass / total_tokens
            target = routing_weights.new_full((1, expert.num_experts), 1.0 / expert.num_experts)
            balance_loss = ((balanced_mass.mean(dim=0, keepdim=True) - target) ** 2).sum()
        else:
            balance_loss = routing_weights.new_zeros(())

        if expert.use_diversity:
            diversity_loss = expert._compute_diversity_penalty(outputs["factors"])
        else:
            diversity_loss = routing_weights.new_zeros(())

        continuity_loss = None
        if expert.use_continuity:
            if routing_weights.size(1) > 1:
                pair_mask = mask_float[:, 1:] * mask_float[:, :-1]
                diff = routing_weights[:, 1:, :] - routing_weights[:, :-1, :]
                diff_sq = diff.pow(2).sum(dim=-1)
                numerator = (diff_sq * pair_mask).sum(dim=1)
                denominator = pair_mask.sum(dim=1).clamp_min(expert.small_value)
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

        if "continuity" in self.expert_weights:
            if continuity_loss is None:
                raise KeyError("Continuity weight configured but model continuity output missing.")
            loss_components["continuity"] = continuity_loss.mean()

        loss = sum(self.expert_weights[key] * loss_components[key] for key in loss_components)
        losses = {key: float(value.detach()) for key, value in loss_components.items()}
        losses["total"] = float(loss.detach())
        return loss, losses, outputs

    def _forward_train(
        self,
        embeddings,
        attention_mask
    ):
        if attention_mask.sum() == 0:
            return [{
                "node": self,
                "selector_loss": 0,
                "expert_loss": 0,
            }]

        if self.children:
            children_stats = []
            _, _, selector_output = self._selector_forward(self.selector, embeddings, attention_mask)
            selection_mask = self._build_selection_mask(selector_output["gates"], attention_mask)
            selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
            selected_mask = (attention_mask * selection_mask.long()).clamp(max=1)

            _, _, expert_output = self._expert_forward(self.expert, selected_embeddings, selected_mask)
            predictions = expert_output["pi"].argmax(dim=-1)
            valid = selected_mask > 0
            for idx, child in enumerate(self.children):
                child_bool = valid & (predictions == idx)
                if not child_bool.any():
                    continue
                child_mask = child_bool.long()
                child_embeddings = selected_embeddings * child_bool.unsqueeze(-1).to(selected_embeddings.dtype)
                child_stats = child._forward_train(
                    child_embeddings,
                    child_mask
                )
                children_stats = children_stats + child_stats
            return children_stats
       
        selector_loss, _, selector_output = self._selector_forward(
            self.selector, embeddings, attention_mask
        )
        
        selection_mask = self._build_selection_mask(selector_output["gates"], attention_mask)
        selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
        selected_mask = (attention_mask * selection_mask.long()).clamp(max=1)

        expert_loss, _, expert_output = self._expert_forward(
            self.expert, selected_embeddings, selected_mask
        )

        return [{
                "node": self,
                "selector_loss": selector_loss,
                "expert_loss": expert_loss
            }]
        
    def _forward_eval(
        self,
        embeddings,
        attention_mask
    ):
        if attention_mask.sum() == 0:
            return []

        if self.children:
            children_pred = []
            _, _, selector_output = self._selector_forward(self.selector, embeddings, attention_mask)
            selection_mask = self._build_selection_mask(selector_output["gates"], attention_mask)
            selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
            selected_mask = (attention_mask * selection_mask.long()).clamp(max=1)

            _, _, expert_output = self._expert_forward(self.expert, selected_embeddings, selected_mask)
            predictions = expert_output["pi"].argmax(dim=-1)
            valid = selected_mask > 0
            for idx, child in enumerate(self.children):
                child_bool = valid & (predictions == idx)
                if not child_bool.any():
                    continue
                child_mask = child_bool.long()
                child_embeddings = selected_embeddings * child_bool.unsqueeze(-1).to(selected_embeddings.dtype)
                child_pred = child._forward_eval(
                    child_embeddings,
                    child_mask,
                )
                children_pred = children_pred + child_pred
            return children_pred
       
        _, _, selector_output = self._selector_forward(
            self.selector, embeddings, attention_mask
        )
        
        with torch.no_grad():
            selection_mask = self._build_selection_mask(selector_output["gates"], attention_mask)
            selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
            selected_mask = (attention_mask * selection_mask.long()).clamp(max=1)

            _, _, expert_output = self._expert_forward(
                self.expert, selected_embeddings, selected_mask
            )
        
            predictions = expert_output["pi"].argmax(dim=-1)
            valid = selected_mask > 0
            predictions_list = []
            for idx in range(self.expert_cfg.expert.num_experts):
                name_branch = self.path + (idx,)
                name = BranchTree.path_to_name(name_branch)
                predictions_list.append((name, (predictions == idx) & valid))

            return predictions_list

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

class BranchTree:
    def __init__(
        self,
        selector_cfg,
        expert_cfg,
        device,
        num_factors,
        selector_backbone=None,
        expert_backbone=None,
    ) -> None:
        self.selector_cfg = selector_cfg
        self.expert_cfg = expert_cfg
        self.device = device
        self.num_factors = int(num_factors)

        selector_backbone = selector_backbone or load_sbert_pooler(selector_cfg.sbert_name, device=self.device)
        expert_backbone = expert_backbone or load_sbert_pooler(expert_cfg.sbert_name, device=self.device)
        (
            self.selector_pooler_template,
            self.selector_hidden_dim,
        ) = selector_backbone
        self.expert_pooler_template, self.expert_hidden_dim = expert_backbone

        self.root = BranchNode(
            path=(),
            selector_hidden_dim=self.selector_hidden_dim,
            selector_pooler=copy.deepcopy(self.selector_pooler_template),
            selector_cfg=self.selector_cfg,
            expert_hidden_dim=self.expert_hidden_dim,
            expert_pooler=copy.deepcopy(self.expert_pooler_template),
            expert_cfg=self.expert_cfg,
            device=self.device,
        )
        self.root.tree = self
        self.depth = 0
        self.path_to_node = {self.root.path: self.root}
        self._initialize_leaf_optimizers([self.root])

    def _set_mode(self, train: bool, node: BranchNode):
        if train and not node.children:
            node.selector.train()
            node.expert.train()
        else:
            node.selector.eval()
            node.expert.eval()

        if node.children:
            for child in node.children:
                self._set_mode(train, child)

    def set_mode(self, train: bool):
        self._set_mode(train, self.root)

    def extend(self) -> BranchNode:
        self.depth += 1
        new_leaves = self._extend(self.root)
        if new_leaves:
            self._initialize_leaf_optimizers(new_leaves)
        return self.root

    def _extend(self, node) -> list:
        new_leaves: list[BranchNode] = []
        if node.children:
            node.no_grad()
            for child in node.children:
                new_leaves.extend(self._extend(child))
        else:
            for factor_idx in range(self.num_factors):
                child_path = node.path + (factor_idx,)
                child_node = BranchNode(
                    path=child_path,
                    selector_hidden_dim=self.selector_hidden_dim,
                    selector_pooler=copy.deepcopy(self.selector_pooler_template),
                    selector_cfg=self.selector_cfg,
                    expert_hidden_dim=self.expert_hidden_dim,
                    expert_pooler=copy.deepcopy(self.expert_pooler_template),
                    expert_cfg=self.expert_cfg,
                    device=self.device,
                )
                child_node.tree = self
                node.children.append(child_node)
                self.path_to_node[child_path] = child_node
                new_leaves.append(child_node)
        return new_leaves

    def _initialize_leaf_optimizers(self, leaves):
        for leaf in leaves:
            if getattr(leaf, "selector_optimizer", None) is not None:
                continue
            selector_params = list(leaf.selector.parameters())
            expert_params = list(leaf.expert.parameters())
            leaf.selector_optimizer = torch.optim.AdamW(
                selector_params,
                lr=float(self.selector_cfg.optim.lr),
                weight_decay=float(self.selector_cfg.optim.weight_decay),
                betas=tuple(self.selector_cfg.optim.betas),
            )
            leaf.expert_optimizer = torch.optim.AdamW(
                expert_params,
                lr=float(self.expert_cfg.optim.lr),
                weight_decay=float(self.expert_cfg.optim.weight_decay),
                betas=tuple(self.expert_cfg.optim.betas),
            )

    def forward_train(
        self,
        embeddings,
        attention_mask
    ):
        return self.root._forward_train(
            embeddings,
            attention_mask
        )
        
    def forward_eval(
        self,
        embeddings,
        attention_mask
    ):
        return self.root._forward_eval(
            embeddings,
            attention_mask
        )

    def build_tree_to_depth(self, depth: int):
        needed_extensions = max(0, depth - self.depth)
        for _ in range(needed_extensions):
            self.extend()
            
    @staticmethod
    def sorted_leaves(metrics_dict: dict[str, dict[str, float]]):
        return sorted(metrics_dict.items(), key=lambda item: item[1].get("f1", 0.0), reverse=True)

    @staticmethod
    def best_leaf(metrics_dict: dict[str, dict[str, float]]):
        if not metrics_dict:
            return None
        return max(metrics_dict.items(), key=lambda item: item[1].get("f1", 0.0))

    @staticmethod
    def path_to_name(path: tuple[int, ...]) -> str:
        if not path:
            return "leaf_root"
        return "leaf_" + "_".join(str(idx) for idx in path)

    @staticmethod
    def name_to_path(name: str) -> tuple[int, ...]:
        prefix = "leaf_"
        if not name.startswith(prefix):
            return ()
        suffix = name[len(prefix):]
        if suffix in {"", "root"}:
            return ()
        try:
            return tuple(int(part) for part in suffix.split("_") if part)
        except ValueError:
            return ()

    def iter_nodes(self):
        """Yield all nodes in the tree (root-first)."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            for child in reversed(node.children):
                stack.append(child)
