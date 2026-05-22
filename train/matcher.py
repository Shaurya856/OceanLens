"""
Hungarian bipartite matcher for DETR-style training.

For each image, we compute a cost matrix between the N predicted queries
and M ground-truth objects, then find the minimum-cost one-to-one assignment.

Cost components (matching only uses species for classification cost):
    cost = λ_cls  * (-p_correct_species)
         + λ_l1   * L1(pred_box, gt_box)
         + λ_giou * (1 - GIoU(pred_box, gt_box))

Returns a list of (pred_indices, gt_indices) index tensors, one per image.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

from core.utils import cxcywh_to_xyxy


@torch.no_grad()
def hungarian_match(
    pred_boxes: torch.Tensor,          # [N, 4]  (cx,cy,w,h) normalised
    pred_species_logits: torch.Tensor, # [N, S]
    gt_boxes: torch.Tensor,            # [M, 4]  (cx,cy,w,h) normalised
    gt_species: torch.Tensor,          # [M]     integer
    lambda_cls: float  = 2.0,
    lambda_l1:  float  = 5.0,
    lambda_giou: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        pred_idx: LongTensor [K]  — matched query indices
        gt_idx:   LongTensor [K]  — corresponding GT indices
    where K = min(N, M).
    """
    N, M = pred_boxes.shape[0], gt_boxes.shape[0]
    if M == 0:
        empty = torch.zeros(0, dtype=torch.long, device=pred_boxes.device)
        return empty, empty

    # ── Classification cost ───────────────────────────────────────────────────
    # Negative probability of the correct ground-truth class for each (query, gt) pair
    probs      = pred_species_logits.softmax(dim=-1)          # [N, S]
    cost_cls   = -probs[:, gt_species]                        # [N, M]

    # ── L1 box cost ───────────────────────────────────────────────────────────
    cost_l1    = torch.cdist(pred_boxes, gt_boxes, p=1)       # [N, M]

    # ── GIoU box cost ─────────────────────────────────────────────────────────
    pred_xyxy  = cxcywh_to_xyxy(pred_boxes)                  # [N, 4]
    gt_xyxy    = cxcywh_to_xyxy(gt_boxes)                    # [M, 4]
    # generalized_box_iou returns [N, M]
    giou_mat   = generalized_box_iou(pred_xyxy, gt_xyxy)
    cost_giou  = 1.0 - giou_mat                               # [N, M]

    # ── Combined cost ─────────────────────────────────────────────────────────
    cost = (
        lambda_cls  * cost_cls  +
        lambda_l1   * cost_l1   +
        lambda_giou * cost_giou
    )  # [N, M]

    row_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())

    device = pred_boxes.device
    return (
        torch.as_tensor(row_idx, dtype=torch.long, device=device),
        torch.as_tensor(col_idx, dtype=torch.long, device=device),
    )


def match_batch(
    pred_boxes_batch: torch.Tensor,          # [B, N, 4]
    pred_species_batch: torch.Tensor,        # [B, N, S]
    targets: list[dict],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run Hungarian matching independently for each image in the batch."""
    matches = []
    for b, target in enumerate(targets):
        gt_boxes   = target["boxes"].to(pred_boxes_batch.device)
        gt_species = target["labels"]["species"].to(pred_boxes_batch.device)
        matches.append(
            hungarian_match(
                pred_boxes_batch[b],
                pred_species_batch[b],
                gt_boxes,
                gt_species,
            )
        )
    return matches
