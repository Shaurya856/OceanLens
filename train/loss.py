"""
Detection loss for SeabedDetector.

Per-image flow
──────────────
1. Hungarian matching  → (pred_idx, gt_idx) pairs
2. Matched queries:
     • Species      Focal CE loss          (weight 2.0)
     • Coarser taxa Auxiliary CE losses    (phylum 0.05 … family 0.3)
     • Boxes        L1 + GIoU             (weights 5.0 + 2.0)
     • Confidence   BCE (positive targets) (weight 1.0)
3. Unmatched queries:
     • Confidence   BCE (negative targets, no-object weight 0.1)

All per-image losses are averaged over the number of ground-truth objects
in the batch (standard DETR normalisation) then summed across images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, sigmoid_focal_loss

from core.config import TAXONOMY_LEVELS
from core.utils import cxcywh_to_xyxy
from train.matcher import match_batch


# Auxiliary loss weights per taxonomy level (coarse → fine)
_LEVEL_WEIGHTS: dict[str, float] = {
    "phylum": 0.05,
    "class_": 0.10,
    "order":  0.20,
    "family": 0.30,
    "species": 2.00,
}
_NO_OBJ_WEIGHT = 0.1   # down-weight the many no-object queries


class DetectionLoss(nn.Module):
    def __init__(
        self,
        lambda_l1:    float = 5.0,
        lambda_giou:  float = 2.0,
        lambda_conf:  float = 1.0,
        focal_alpha:  float = 0.25,
        focal_gamma:  float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_conf = lambda_conf
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: dict,
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: SeabedDetector forward output dict.
            targets: List of per-image target dicts (from DataLoader).
        Returns:
            Dict of named scalar losses + "total" key.
        """
        pred_boxes   = outputs["boxes"]       # [B, N, 4]
        pred_conf    = outputs["confidence"]  # [B, N]  raw logits
        class_logits = outputs["class_logits"]  # {level: [B, N, C]}

        B, N, _ = pred_boxes.shape
        device   = pred_boxes.device

        # Total number of GT objects in this batch (for normalisation)
        num_gt = max(1, sum(len(t["boxes"]) for t in targets))

        # Hungarian matching (uses species logits only for assignment cost)
        matches = match_batch(pred_boxes, class_logits["species"], targets)

        loss_cls   = torch.tensor(0.0, device=device)
        loss_l1    = torch.tensor(0.0, device=device)
        loss_giou  = torch.tensor(0.0, device=device)
        loss_conf  = torch.tensor(0.0, device=device)

        for b, (pred_idx, gt_idx) in enumerate(matches):
            gt_boxes = targets[b]["boxes"].to(device)   # [M, 4]
            gt_labels = {
                lvl: targets[b]["labels"][lvl].to(device)
                for lvl in TAXONOMY_LEVELS
            }
            M = gt_boxes.shape[0]

            # ── Confidence loss (all queries) ─────────────────────────────────
            conf_targets = torch.zeros(N, device=device)
            if pred_idx.numel():
                conf_targets[pred_idx] = 1.0
            # Weighted BCE: positives get weight 1, negatives get _NO_OBJ_WEIGHT
            conf_weight = torch.full((N,), _NO_OBJ_WEIGHT, device=device)
            conf_weight[pred_idx] = 1.0
            loss_conf = loss_conf + (
                F.binary_cross_entropy_with_logits(
                    pred_conf[b], conf_targets, weight=conf_weight, reduction="sum"
                )
                / num_gt
            ) * self.lambda_conf

            if pred_idx.numel() == 0:
                continue

            # ── Matched queries only ──────────────────────────────────────────
            matched_pred_boxes = pred_boxes[b][pred_idx]    # [K, 4]
            matched_gt_boxes   = gt_boxes[gt_idx]           # [K, 4]

            # Classification losses for each taxonomy level
            for level in TAXONOMY_LEVELS:
                logits    = class_logits[level][b][pred_idx]        # [K, C]
                gt_cls    = gt_labels[level][gt_idx]                # [K]
                w         = _LEVEL_WEIGHTS[level]

                if level == "species":
                    # Focal loss for the fine-grained species head
                    one_hot = F.one_hot(gt_cls, logits.shape[-1]).float()
                    fl = sigmoid_focal_loss(
                        logits, one_hot,
                        alpha=self.focal_alpha,
                        gamma=self.focal_gamma,
                        reduction="sum",
                    )
                    loss_cls = loss_cls + w * fl / num_gt
                else:
                    # Standard cross-entropy for coarser levels
                    loss_cls = loss_cls + w * (
                        F.cross_entropy(logits, gt_cls, reduction="sum") / num_gt
                    )

            # L1 box loss
            loss_l1 = loss_l1 + self.lambda_l1 * (
                F.l1_loss(matched_pred_boxes, matched_gt_boxes, reduction="sum") / num_gt
            )

            # GIoU box loss
            pred_xyxy = cxcywh_to_xyxy(matched_pred_boxes)
            gt_xyxy   = cxcywh_to_xyxy(matched_gt_boxes)
            giou = generalized_box_iou(pred_xyxy, gt_xyxy)
            loss_giou = loss_giou + self.lambda_giou * (
                (1 - giou.diag()).sum() / num_gt
            )

        total = loss_cls + loss_l1 + loss_giou + loss_conf

        return {
            "loss_cls":  loss_cls,
            "loss_l1":   loss_l1,
            "loss_giou": loss_giou,
            "loss_conf": loss_conf,
            "total":     total,
        }
