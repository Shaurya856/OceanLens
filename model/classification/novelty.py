"""
Novel / unseen species detector.

Two complementary signals are combined:

1. Confidence gate  — if max softmax probability across all species < CONF_THRESHOLD
                      the detection is flagged as potentially novel.

2. Prototype distance — during / after training, call `update_prototypes()` to
                        accumulate per-class mean embeddings. At inference, the
                        cosine similarity to the nearest prototype is compared
                        against DIST_THRESHOLD. If no embedding is close enough
                        the detection is flagged as novel.

A detection is marked `is_novel=True` when *either* signal fires.
The `novelty_score` is 1 − max_softmax_confidence, ranging from 0 (certain
known species) to 1 (completely uncertain / novel).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoveltyDetector(nn.Module):
    """
    Args:
        d_model:        Embedding dimension (matches decoder output).
        num_species:    Total number of known species.
        conf_threshold: Softmax confidence below which → flag as novel.
        dist_threshold: Cosine similarity to nearest prototype below which → flag as novel.
    """

    def __init__(
        self,
        d_model: int,
        num_species: int,
        conf_threshold: float = 0.5,
        dist_threshold: float = 0.7,
        confidence_gate_only: bool = False,
    ) -> None:
        super().__init__()
        self.conf_threshold = conf_threshold
        self.dist_threshold = dist_threshold
        # When True, skip the prototype distance check entirely.
        # Used by SeabedLite where prototype population is not performed.
        self.confidence_gate_only = confidence_gate_only

        # Prototype embeddings — not a learnable parameter, updated via
        # update_prototypes(). Saved/loaded with model state dict.
        self.register_buffer("prototypes", torch.zeros(num_species, d_model))
        self.register_buffer("prototype_counts", torch.zeros(num_species))

    @property
    def _prototypes_initialised(self) -> bool:
        return bool(self.prototype_counts.sum().item() > 0)

    def forward(
        self,
        embeddings: torch.Tensor,   # [B, N, d_model]
        species_logits: torch.Tensor,  # [B, N, num_species]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            novelty_score: [B, N]  — 1 − max_softmax_conf (higher = more novel).
            is_novel:      [B, N]  — bool mask.
        """
        probs = species_logits.softmax(dim=-1)           # [B, N, S]
        max_conf, _ = probs.max(dim=-1)                  # [B, N]
        novelty_score = 1.0 - max_conf

        conf_novel = max_conf < self.conf_threshold      # [B, N]

        if not self.confidence_gate_only and self._prototypes_initialised:
            emb_norm   = F.normalize(embeddings, dim=-1)         # [B, N, d]
            proto_norm = F.normalize(self.prototypes, dim=-1)    # [S, d]
            # Cosine similarity: [B, N, S]
            sim = torch.einsum("bnd,sd->bns", emb_norm, proto_norm)
            max_sim, _ = sim.max(dim=-1)                         # [B, N]
            dist_novel = max_sim < self.dist_threshold
        else:
            dist_novel = torch.zeros_like(conf_novel)

        is_novel = conf_novel | dist_novel

        return novelty_score, is_novel

    @torch.no_grad()
    def update_prototypes(
        self,
        embeddings: torch.Tensor,  # [N, d_model]
        labels: torch.Tensor,      # [N]  integer species indices
    ) -> None:
        """
        Online mean update — call after each training batch or epoch.
        Thread-safe as long as the model is not being trained simultaneously.
        """
        for cls_idx in labels.unique():
            mask = labels == cls_idx
            cls = int(cls_idx.item())
            n_new = mask.sum().item()
            n_old = self.prototype_counts[cls].item()
            self.prototypes[cls] = (
                self.prototypes[cls] * n_old + embeddings[mask].sum(0)
            ) / (n_old + n_new)
            self.prototype_counts[cls] += n_new
