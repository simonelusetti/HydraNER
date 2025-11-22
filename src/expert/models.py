import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer


class ExpertModel(nn.Module):
    """
    Token-to-factor mixture-of-experts rationale model.
    Routes each token to latent experts, aggregates expert embeddings,
    and reconstructs the anchor sentence embedding (and optionally token states).
    """

    def __init__(self, cfg, *, pooler=None, embedding_dim=None):
        super().__init__()
        self.cfg = cfg
        expert_cfg = cfg.expert

        self.num_experts = int(expert_cfg.num_experts)
        if self.num_experts < 1:
            raise ValueError("cfg.model.expert.num_experts must be >= 1")

        self.routing = expert_cfg.routing.lower()
        self.routing_tau = float(expert_cfg.routing_tau)
        self.normalize_factors = bool(expert_cfg.normalize)
        self.dropout = float(expert_cfg.dropout)
        self.use_balance = expert_cfg.use_balance
        self.use_diversity = expert_cfg.use_diversity
        self.use_continuity = bool(expert_cfg.use_continuity)

        if pooler is not None and embedding_dim is not None:
            self.sbert = None
            self.pooler = pooler
            enc_hidden = int(embedding_dim)
        else:
            self.sbert = SentenceTransformer(cfg.sbert_name)
            self.pooler = self.sbert[1]
            enc_hidden = self.sbert[0].auto_model.config.hidden_size

        factor_dim = int(expert_cfg.factor_dim)
        factor_hidden = int(expert_cfg.factor_hidden)
        self.factor_dim = factor_dim
        self.expert_attention = None
        self.attention_layer_norm = None

        gate_hidden = int(expert_cfg.gate_hidden)
        gate_dropout = float(expert_cfg.gate_dropout)
        gate_layers = []
        in_dim = enc_hidden
        if gate_hidden > 0:
            gate_layers.append(nn.Linear(in_dim, gate_hidden))
            gate_layers.append(nn.GELU())
            if gate_dropout > 0.0:
                gate_layers.append(nn.Dropout(gate_dropout))
            in_dim = gate_hidden
        gate_layers.append(nn.Linear(in_dim, self.num_experts))
        self.gate = nn.Sequential(*gate_layers)

        use_shared_transform = bool(expert_cfg.shared_transform)
        transform_dropout = float(expert_cfg.transform_dropout)
        if use_shared_transform:
            self.expert_transforms = None
            self.shared_transform = self._build_transform(
                enc_hidden, factor_hidden, factor_dim, transform_dropout
            )
        else:
            self.shared_transform = None
            modules = []
            for _ in range(self.num_experts):
                modules.append(
                    self._build_transform(enc_hidden, factor_hidden, factor_dim, transform_dropout)
                )
            self.expert_transforms = nn.ModuleList(modules)

        sentence_dim = self.pooler.get_sentence_embedding_dimension()
        recon_modules = []
        recon_hidden = int(expert_cfg.reconstruction_hidden)
        for _ in range(self.num_experts):
            layers = []
            in_features = factor_dim
            if recon_hidden > 0:
                layers.append(nn.Linear(in_features, recon_hidden))
                layers.append(nn.GELU())
                if transform_dropout > 0:
                    layers.append(nn.Dropout(transform_dropout))
                in_features = recon_hidden
            layers.append(nn.Linear(in_features, sentence_dim))
            recon_modules.append(nn.Sequential(*layers))
        self.reconstruction_heads = nn.ModuleList(recon_modules)

        self.use_token_decoder = bool(expert_cfg.use_token_decoder)
        if self.use_token_decoder:
            decoder_hidden = int(expert_cfg.decoder_hidden)
            decoder_layers = []
            decoder_in = factor_dim
            if decoder_hidden > 0:
                decoder_layers.append(nn.Linear(decoder_in, decoder_hidden))
                decoder_layers.append(nn.GELU())
                if transform_dropout > 0:
                    decoder_layers.append(nn.Dropout(transform_dropout))
                decoder_in = decoder_hidden
            decoder_layers.append(nn.Linear(decoder_in, enc_hidden))
            self.token_decoder = nn.Sequential(*decoder_layers)
        else:
            self.token_decoder = None

        self.small_value = 1e-6

    @staticmethod
    def _build_transform(in_dim, hidden_dim, out_dim, dropout):
        layers = []
        if hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def _apply_routing(self, logits, mask):
        mask = mask.unsqueeze(-1).to(logits.dtype)
        logits = logits.masked_fill(mask == 0, -1e9)

        if self.routing == "gumbel":
            routed = F.gumbel_softmax(logits, tau=self.routing_tau, hard=False, dim=-1)
        elif self.routing == "softmax":
            routed = torch.softmax(logits, dim=-1)
        else:
            raise ValueError(f"Unsupported routing '{self.routing}'. Use 'softmax' or 'gumbel'.")

        routed = routed * mask
        routed_sum = routed.sum(dim=-1, keepdim=True).clamp_min(self.small_value)
        routed = routed / routed_sum
        return routed

    def _transform_factors(self, factors):
        if self.shared_transform is not None:
            flat = factors.reshape(-1, factors.size(-1))
            transformed = self.shared_transform(flat)
            return transformed.view(factors.size(0), self.num_experts, -1)

        outputs = []
        for idx, module in enumerate(self.expert_transforms):
            outputs.append(module(factors[:, idx, :]))
        return torch.stack(outputs, dim=1)

    def _compute_diversity_penalty(self, factors):
        if factors.size(0) < 2:
            return factors.new_tensor(0.0)
        centered = factors - factors.mean(dim=0, keepdim=True)
        flat = centered.view(factors.size(0), self.num_experts, -1)
        gram = torch.einsum("bkd,bjd->kj", flat, flat) / flat.size(0)
        diag = torch.diag(gram)
        off_diag = gram - torch.diag_embed(diag)
        return off_diag.pow(2).sum()

    def forward(self, embeddings, attention_mask):
        mask_float = attention_mask.to(dtype=embeddings.dtype)
        gate_logits = self.gate(embeddings)
        routing_weights = self._apply_routing(gate_logits, mask_float)

        factors_raw = torch.einsum("btk,btd->bkd", routing_weights, embeddings)
        mass = routing_weights.sum(dim=1).clamp_min(self.small_value).unsqueeze(-1)
        if self.normalize_factors:
            factors_raw = factors_raw / mass
        transformed_factors = self._transform_factors(factors_raw)

        pooled = self.pooler(
            {
                "token_embeddings": embeddings,
                "attention_mask": attention_mask,
            }
        )["sentence_embedding"]

        recon_parts = []
        for idx, head in enumerate(self.reconstruction_heads):
            recon_parts.append(head(transformed_factors[:, idx, :]))
        reconstruction = torch.stack(recon_parts, dim=1).sum(dim=1)

        token_reconstruction = None
        if self.token_decoder is not None:
            mixture = torch.einsum("btk,bkf->btf", routing_weights, transformed_factors)
            token_reconstruction = self.token_decoder(mixture)

        outputs = {
            "pi": routing_weights,
            "factors_raw": factors_raw,
            "factors": transformed_factors,
            "anchor": pooled,
            "reconstruction": reconstruction,
            "token_reconstruction": token_reconstruction,
        }
        return outputs
