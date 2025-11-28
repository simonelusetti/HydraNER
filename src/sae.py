import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Tied-weight sparse autoencoder for LM activations."""

    def __init__(self, d_model: int, expansion: float = 4.0, alpha: float = 8.6e-4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.expansion = float(expansion)
        self.alpha = float(alpha)

        self.d_hidden = int(round(self.expansion * self.d_model))
        if self.d_hidden < 1:
            raise ValueError("expansion must yield at least 1 hidden unit.")

        # Dictionary weights (encoder rows / decoder columns) and bias
        self.M = nn.Parameter(torch.randn(self.d_hidden, self.d_model) / (self.d_model ** 0.5))
        self.b = nn.Parameter(torch.zeros(self.d_hidden))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, d_model] activations.
        Returns:
            dict with x_hat, codes, loss, mse, l1, active_frac, active_per_sample.
        """
        codes = torch.relu(torch.matmul(x, self.M.t()) + self.b)          # [N, d_hidden]
        x_hat = torch.matmul(codes, self.M)                               # [N, d_model]

        mse = torch.mean((x_hat - x) ** 2)
        l1 = torch.sum(torch.abs(codes))
        loss = mse + self.alpha * l1

        with torch.no_grad():
            active = (codes > 0).float()
            active_frac = active.mean()
            active_per_sample = active.sum(dim=1).mean()

        return {
            "x_hat": x_hat,
            "codes": codes,
            "loss": loss,
            "mse": mse,
            "l1": l1,
            "active_frac": active_frac,
            "active_per_sample": active_per_sample,
        }

    @torch.no_grad()
    def normalize_dictionary(self):
        """Row-normalize the dictionary matrix to prevent scale drift."""
        norms = self.M.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.M.div_(norms)
