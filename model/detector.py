"""
SeabedDetector — full assembled model.

Architecture:
    DualPathBackbone  (ConvNeXt-S + Swin-T + CBAM)  → [P3, P4, P5]
    BiFPN × 3                                         → [P3, P4, P5, P6]
    DETRDecoder       (cross-attn over P3+P4+P5)      → query embeddings
    ┌─ bbox_head      MLP → 4  (cx, cy, w, h normalised)
    ├─ conf_head      Linear → 1
    ├─ HierarchicalClassifier → logits per taxonomy level
    └─ NoveltyDetector        → novelty_score, is_novel

Forward output (dict):
    boxes           [B, N, 4]   sigmoid'd (0–1 normalised cx/cy/w/h)
    confidence      [B, N]      raw logits (apply sigmoid at inference)
    class_logits    dict[level → [B, N, num_classes]]
    novelty_scores  [B, N]
    is_novel        [B, N]  bool
    embeddings      [B, N, d_model]
"""
import torch
import torch.nn as nn

from model._common import _MLP
from model.detection.backbone import DualPathBackbone
from model.detection.bifpn import BiFPN
from model.detection.decoder import DETRDecoder
from model.classification.classifier import HierarchicalClassifier
from model.classification.novelty import NoveltyDetector


class SeabedDetector(nn.Module):
    def __init__(
        self,
        taxonomy_sizes: dict[str, int],
        d_model: int = 256,
        num_queries: int = 300,
        bifpn_iters: int = 3,
        conf_threshold: float = 0.5,
        dist_threshold: float = 0.7,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        num_species = taxonomy_sizes["species"]

        self.backbone = DualPathBackbone(pretrained=pretrained_backbone)
        self.bifpn    = BiFPN(d_model=d_model, num_iters=bifpn_iters)
        self.decoder  = DETRDecoder(
            d_model=d_model,
            nhead=8,
            num_layers=6,
            num_queries=num_queries,
        )

        # Detection heads
        self.bbox_head = _MLP(d_model, d_model, 4)
        self.conf_head = nn.Linear(d_model, 1)

        # Classification
        self.classifier = HierarchicalClassifier(d_model, taxonomy_sizes)

        # Novel species
        self.novelty = NoveltyDetector(
            d_model=d_model,
            num_species=num_species,
            conf_threshold=conf_threshold,
            dist_threshold=dist_threshold,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        """
        Args:
            x: [B, 3, H, W]  — normalised RGB tensor.
        Returns:
            Dict with keys: boxes, confidence, class_logits,
                            novelty_scores, is_novel, embeddings.
        """
        # Feature extraction
        feats   = self.backbone(x)           # [P3, P4, P5]
        pyramid = self.bifpn(feats)          # [P3, P4, P5, P6]

        # Decoder only uses P3–P5 for cross-attention (P6 stays in BiFPN)
        queries = self.decoder(pyramid[:3])  # [B, N, d_model]

        # Detection outputs
        boxes = self.bbox_head(queries).sigmoid()    # [B, N, 4]
        conf  = self.conf_head(queries).squeeze(-1)  # [B, N]

        # Taxonomy logits
        class_logits = self.classifier(queries)      # {level: [B, N, C]}

        # Novelty
        novelty_scores, is_novel = self.novelty(queries, class_logits["species"])

        return {
            "boxes":          boxes,
            "confidence":     conf,
            "class_logits":   class_logits,
            "novelty_scores": novelty_scores,
            "is_novel":       is_novel,
            "embeddings":     queries,
        }
