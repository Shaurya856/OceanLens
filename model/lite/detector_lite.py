"""
SeabedLite — lightweight assembled model for laptop demo / development.

Architecture (vs full SeabedDetector):
    LiteBackbone   MobileNetV3-Small (~2.5M params)  ← ConvNeXt-S + Swin-T
    LiteFPN        1 top-down pass, 128 ch, 3 levels  ← BiFPN × 3, 256 ch
    DETRDecoder    2 layers, 4 heads, 50 queries       ← 6 layers, 8 heads, 300
    HierarchicalClassifier  same, smaller taxonomy    ← same
    NoveltyDetector         confidence gate only       ← conf + prototype dist

Forward output dict is identical to SeabedDetector so inference/runner.py,
the loss function, and the matcher work without modification.

Estimated params:  ~3.5–4M
Inference on M3 MPS (320×320): ~10–20 ms
"""
import torch
import torch.nn as nn

from model._common import _MLP
from model.lite.backbone_lite import LiteBackbone
from model.lite.neck_lite import LiteFPN
from model.detection.decoder import DETRDecoder      # fully reused, just smaller params
from model.classification.classifier import HierarchicalClassifier
from model.classification.novelty import NoveltyDetector


class SeabedLite(nn.Module):
    """
    Args:
        taxonomy_sizes:    {level: num_classes} — pass LITE_TAXONOMY_SIZES or
                           the actual counts read from annotations.json.
        d_model:           Feature dimension (default 128).
        num_queries:       Object queries (default 50).
        decoder_layers:    Transformer decoder depth (default 2).
        decoder_heads:     Attention heads (default 4).
        conf_threshold:    Novelty confidence gate (default 0.5).
        pretrained_backbone: Load MobileNetV3-Small ImageNet weights (default True).
    """

    def __init__(
        self,
        taxonomy_sizes: dict[str, int],
        d_model: int = 128,
        num_queries: int = 50,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        conf_threshold: float = 0.5,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        num_species = taxonomy_sizes["species"]

        self.backbone = LiteBackbone(pretrained=pretrained_backbone)
        self.neck = LiteFPN(
            in_channels=self.backbone.out_channels,
            d_model=d_model,
        )
        self.decoder = DETRDecoder(
            d_model=d_model,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            num_queries=num_queries,
        )

        # Detection heads — smaller than full model (d_model=128 vs 256)
        self.bbox_head = _MLP(d_model, d_model // 2, 4)
        self.conf_head = nn.Linear(d_model, 1)

        # Classification — same module, smaller taxonomy & d_model
        self.classifier = HierarchicalClassifier(d_model, taxonomy_sizes)

        # Novelty — confidence gate only (no prototype distance for demo)
        self.novelty = NoveltyDetector(
            d_model=d_model,
            num_species=num_species,
            conf_threshold=conf_threshold,
            confidence_gate_only=True,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        """
        Args:
            x: [B, 3, H, W]  — normalised RGB, H=W=LITE_INPUT_SIZE (320).
        Returns:
            Same dict as SeabedDetector:
                boxes, confidence, class_logits, novelty_scores, is_novel, embeddings.
        """
        feats   = self.backbone(x)       # [P3, P4, P5]
        pyramid = self.neck(feats)       # [P3, P4, P5] all 128-ch

        queries = self.decoder(pyramid)  # [B, num_queries, d_model]

        boxes = self.bbox_head(queries).sigmoid()    # [B, N, 4]
        conf  = self.conf_head(queries).squeeze(-1)  # [B, N]

        class_logits = self.classifier(queries)      # {level: [B, N, C]}

        novelty_scores, is_novel = self.novelty(queries, class_logits["species"])

        return {
            "boxes":          boxes,
            "confidence":     conf,
            "class_logits":   class_logits,
            "novelty_scores": novelty_scores,
            "is_novel":       is_novel,
            "embeddings":     queries,
        }
