"""
Hierarchical classification head.

Produces independent logits for each taxonomy level from the decoder
query embeddings. Each level is a two-layer MLP. Training applies
auxiliary losses at all levels with the species head as the primary output.
"""
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalClassifier(nn.Module):
    """
    Args:
        d_model:        Input embedding dimension.
        taxonomy_sizes: Ordered dict {level_name: num_classes} from coarse→fine.
                        Must end with "species".
    """

    def __init__(self, d_model: int, taxonomy_sizes: dict[str, int]) -> None:
        super().__init__()
        self.levels = list(taxonomy_sizes.keys())
        self.heads = nn.ModuleDict(
            {
                level: _MLP(d_model, d_model, n_cls)
                for level, n_cls in taxonomy_sizes.items()
            }
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [B, N, d_model]
        Returns:
            Dict mapping each taxonomy level to its logits [B, N, num_classes].
        """
        return {level: self.heads[level](embeddings) for level in self.levels}
