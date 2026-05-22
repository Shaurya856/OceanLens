import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    """3-layer feed-forward network: in_dim → hidden_dim → out_dim."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
