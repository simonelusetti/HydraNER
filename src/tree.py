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
        self.prototype_cfg = expert_cfg.prototype
        self.children: list["BranchNode"] = []
        weights_cfg = expert_cfg.loss_weights
        self.expert_weights = {}
        for key in ["sent", "token", "entropy", "overlap", "diversity", "balance", "continuity"]:
            value = getattr(weights_cfg, key)
            if value is None:
                continue
            self.expert_weights[key] = float(value)
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

        self.prototype = None
        proto_cons = self.prototype_cfg.lambda_cons
        proto_sep = self.prototype_cfg.lambda_sep
        if self.prototype_cfg is not None and (
            (proto_cons is not None and proto_cons != 0.0)
            or (proto_sep is not None and proto_sep != 0.0)
        ):
            factor_dim = int(self.expert_cfg.expert.factor_dim)
            self.prototype = torch.zeros(self.expert_cfg.expert.num_experts, factor_dim, device=device)

    def no_grad(self):
        for param in self.selector.parameters():
            param.requires_grad_(False)
        for param in self.expert.parameters():
            param.requires_grad_(False)

    def path_to_name(self) -> str:
        return BranchTree.path_to_name(self.path)

    def _selector_forward(self, selector, embeddings, mask):
        outputs = selector(embeddings, mask)
        cfg_loss = self.selector_cfg.loss

        h_anchor = outputs["h_anchor"]
        h_rat = outputs["h_rat"]
        h_comp = outputs["h_comp"]
        gates = outputs["gates"]

        rat_loss = nt_xent(h_rat, h_anchor, temperature=cfg_loss.tau)
        comp_loss = complement_loss(h_comp, h_anchor, temperature=float(cfg_loss.tau))
        sparsity = rat_sparsity_loss(gates, mask)
        tv = total_variation_1d(gates, mask)

        loss = rat_loss
        loss = loss + float(cfg_loss.l_comp) * comp_loss
        loss = loss + float(cfg_loss.l_s) * sparsity
        loss = loss + float(cfg_loss.l_tv) * tv

        outputs["losses"] = {
            "rat": float(rat_loss.detach()),
            "comp": float(comp_loss.detach()),
            "sparsity": float(sparsity.detach()),
            "tv": float(tv.detach()),
            "total": float(loss.detach()),
        }
        return loss, outputs["losses"], outputs

    def _expert_forward(self, expert, embeddings, mask):
        outputs = expert(embeddings, mask)
        routing_weights = outputs["pi"]
        anchor = F.normalize(outputs["anchor"], dim=-1)
        reconstruction = F.normalize(outputs["reconstruction"], dim=-1)

        sent_loss = nt_xent(anchor, reconstruction, temperature=self.contrastive_tau)

        token_reconstruction = outputs["token_reconstruction"] if "token_reconstruction" in outputs else None
        if token_reconstruction is not None:
            mask_float_tok = mask.unsqueeze(-1).to(dtype=token_reconstruction.dtype)
            diff = token_reconstruction - embeddings
            token_loss = (diff.pow(2) * mask_float_tok).sum() / mask_float_tok.sum().clamp_min(1.0)
        else:
            token_loss = embeddings.new_tensor(0.0)

        mask_float = mask.to(dtype=routing_weights.dtype)

        loss_components = {"sent": sent_loss, "token": token_loss}

        if "overlap" in self.expert_weights:
            pi_sq = (routing_weights ** 2).sum(dim=-1)
            overlap = 1.0 - pi_sq
            overlap = (overlap * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
            loss_components["overlap"] = overlap.mean()

        if "balance" in self.expert_weights and expert.use_balance:
            expert_mass = routing_weights.sum(dim=1)
            total_tokens = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            balanced_mass = expert_mass / total_tokens
            target = routing_weights.new_full((1, expert.num_experts), 1.0 / expert.num_experts)
            loss_components["balance"] = ((balanced_mass.mean(dim=0, keepdim=True) - target) ** 2).sum()

        if "continuity" in self.expert_weights and expert.use_continuity:
            if routing_weights.size(1) > 1:
                pair_mask = mask_float[:, 1:] * mask_float[:, :-1]
                diff = routing_weights[:, 1:, :] - routing_weights[:, :-1, :]
                diff_sq = diff.pow(2).sum(dim=-1)
                numerator = (diff_sq * pair_mask).sum(dim=1)
                denominator = pair_mask.sum(dim=1).clamp_min(expert.small_value)
                continuity_loss = numerator / denominator
            else:
                continuity_loss = routing_weights.new_zeros(routing_weights.size(0))
            loss_components["continuity"] = continuity_loss.mean()

        loss = 0.0
        for key, weight in self.expert_weights.items():
            if key not in loss_components:
                continue
            loss = loss + weight * loss_components[key]

        losses = {key: float(value.detach()) for key, value in loss_components.items()}

        # Prototype consistency / separation (optional)
        if self.prototype is not None:
            eps = float(self.prototype_cfg.eps)
            decay = float(self.prototype_cfg.ema_decay)
            lambda_cons = float(self.prototype_cfg.lambda_cons) if self.prototype_cfg.lambda_cons is not None else 0.0
            lambda_sep = float(self.prototype_cfg.lambda_sep) if self.prototype_cfg.lambda_sep is not None else 0.0
            margin = float(self.prototype_cfg.margin)

            mask_float = mask.to(routing_weights.dtype)
            token_counts = mask_float.sum(dim=1, keepdim=True).clamp_min(eps)
            usage = (routing_weights * mask_float.unsqueeze(-1)).sum(dim=1) / token_counts

            proto = self.prototype.detach()
            diff = outputs["factors"] - proto.unsqueeze(0)
            cons_per = (usage * diff.pow(2).sum(dim=-1)).sum(dim=1)
            cons_loss = cons_per.mean()

            if lambda_cons > 0.0:
                loss = loss + lambda_cons * cons_loss
                losses["proto_cons"] = float(cons_loss.detach())

            # EMA update (no grad)
            with torch.no_grad():
                num = (usage.unsqueeze(-1) * outputs["factors"]).sum(dim=0)
                denom = usage.sum(dim=0).unsqueeze(-1).clamp_min(eps)
                update = num / denom
                self.prototype.mul_(decay).add_(update * (1.0 - decay))

            if lambda_sep > 0.0:
                proto_norm = proto / (proto.norm(dim=-1, keepdim=True).clamp_min(eps))
                cos = proto_norm @ proto_norm.t()
                off_diag = cos - torch.eye(cos.size(0), device=cos.device)
                sep = torch.clamp(off_diag - (1.0 - margin), min=0.0)
                sep_loss = sep.sum() / max(1, cos.numel() - cos.size(0))
                loss = loss + lambda_sep * sep_loss
                losses["proto_sep"] = float(sep_loss.detach())

        losses["total"] = float(loss.detach())
        outputs["losses"] = losses
        return loss, losses, outputs

    def _forward_train(
        self,
        embeddings,
        attention_mask,
        return_debug: bool = False,
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
                    child_mask,
                    return_debug=return_debug,
                )
                children_stats = children_stats + child_stats
            return children_stats
       
        selector_loss, selector_losses, selector_output = self._selector_forward(
            self.selector, embeddings, attention_mask
        )
        
        selection_mask = self._build_selection_mask(selector_output["gates"], attention_mask)
        selected_embeddings = embeddings * selection_mask.unsqueeze(-1)
        selected_mask = (attention_mask * selection_mask.long()).clamp(max=1)

        expert_loss, expert_losses, expert_output = self._expert_forward(
            self.expert, selected_embeddings, selected_mask
        )

        debug = None
        if return_debug:
            debug = {
                "pi": expert_output["pi"].detach(),
                "factors": expert_output["factors"].detach(),
                "anchor": expert_output["anchor"].detach(),
                "reconstruction": expert_output["reconstruction"].detach(),
                "selection_mask": selection_mask.detach(),
                "selected_mask": selected_mask.detach(),
                "gates": selector_output["gates"].detach(),
            }

        return [{
                "node": self,
                "selector_loss": selector_loss,
                "selector_losses": selector_losses,
                "expert_loss": expert_loss,
                "expert_losses": expert_losses,
                "debug": debug,
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
            if hasattr(leaf, "selector_optimizer") and leaf.selector_optimizer is not None:
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
        attention_mask,
        return_debug: bool = False,
    ):
        return self.root._forward_train(
            embeddings,
            attention_mask,
            return_debug=return_debug,
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
        return sorted(metrics_dict.items(), key=lambda item: item[1]["f1"], reverse=True)

    @staticmethod
    def best_leaf(metrics_dict: dict[str, dict[str, float]]):
        if not metrics_dict:
            return None
        return max(metrics_dict.items(), key=lambda item: item[1]["f1"])

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
