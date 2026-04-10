"""
Two-phase training loop for SeabedDetector.

Phase 1 — Backbone frozen  (epochs 0 … warmup_epochs-1)
    Only BiFPN, decoder, and all heads are trained.
    Larger effective LR is safe since backbone features are stable.

Phase 2 — Full fine-tune    (epochs warmup_epochs … total_epochs-1)
    Backbone unfrozen with 10× smaller LR (layer-wise decay).
    Gradient clipping at max_norm=0.1 (standard for DETR-family models).

Checkpoints
───────────
Saved to  {checkpoint_dir}/epoch_{n:03d}.pt  after every epoch.
Best model (lowest val loss) saved to  {checkpoint_dir}/best.pt.
After training, detector weights are also copied to MODEL_WEIGHTS_PATH
so the inference pipeline picks them up immediately.

Prototype update
────────────────
After Phase 2 completes, one pass over the training set is made to
populate novelty_detector.prototypes from real embedding distributions.
"""
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split

from core.config import (
    MODEL_WEIGHTS_PATH,
    LITE_WEIGHTS_PATH,
    TAXONOMY_LEVELS,
)
from model.detector import SeabedDetector
from train.dataset import SeabedDataset, collate_fn
from train.augmentations import UnderwaterAugmentation
from train.loss import DetectionLoss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Optimizer builders ────────────────────────────────────────────────────────

def _build_optimizer(
    model: SeabedDetector,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    freeze_backbone: bool,
) -> AdamW:
    """
    Parameter groups:
      • Backbone (ConvNeXt + Swin)     → backbone_lr  (or 0 when frozen)
      • Everything else                 → head_lr

    Bias and LayerNorm parameters are excluded from weight decay.
    """
    decay, no_decay = [], []

    def _is_no_decay(name: str) -> bool:
        return any(nd in name for nd in ("bias", "norm", "LayerNorm"))

    backbone_names = {"backbone", "convnext", "swin"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = any(b in name for b in backbone_names)
        lr = (0.0 if freeze_backbone else backbone_lr) if is_backbone else head_lr
        group = {"params": param, "lr": lr}
        if _is_no_decay(name):
            group["weight_decay"] = 0.0
        else:
            group["weight_decay"] = weight_decay
        (no_decay if _is_no_decay(name) else decay).append(group)

    return AdamW(decay + no_decay, lr=head_lr, weight_decay=weight_decay)


# ── Mixed-precision helpers ───────────────────────────────────────────────────

def _autocast(device: torch.device):
    """Return the appropriate autocast context for the device."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if device.type == "mps":
        return torch.autocast("mps", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


# ── Training / validation steps ───────────────────────────────────────────────

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: DetectionLoss,
    device: torch.device,
    scaler: "torch.cuda.GradScaler | None" = None,
    max_norm: float = 0.1,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    ac = _autocast(device)

    for images, targets in loader:
        images = images.to(device)

        with ac:
            outputs = model(images)
            losses  = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    ac = _autocast(device)

    for images, targets in loader:
        images = images.to(device)
        with ac:
            outputs = model(images)
            losses  = criterion(outputs, targets)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ── Prototype population ──────────────────────────────────────────────────────

@torch.no_grad()
def _update_prototypes(
    model: SeabedDetector,
    loader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.3,
) -> None:
    """
    One forward pass to populate per-class prototype embeddings.
    Only high-confidence, non-novel predictions are used.
    """
    model.eval()
    for images, targets in loader:
        images  = images.to(device)
        outputs = model(images)

        embeddings   = outputs["embeddings"]   # [B, N, d]
        conf         = outputs["confidence"].sigmoid()
        species_pred = outputs["class_logits"]["species"].argmax(-1)  # [B, N]

        for b, target in enumerate(targets):
            gt_species = target["labels"]["species"].to(device)
            mask = conf[b] > conf_thresh
            if not mask.any():
                continue
            emb = embeddings[b][mask]
            # Use ground-truth species labels (training phase, GT is known)
            # Only use embeddings whose argmax matches a GT label
            pred_sp = species_pred[b][mask]
            gt_flat = gt_species.unique()
            valid = torch.isin(pred_sp, gt_flat)
            if not valid.any():
                continue
            model.novelty.update_prototypes(emb[valid], pred_sp[valid])

    logger.info("Prototypes updated for %d species", (model.novelty.prototype_counts > 0).sum().item())


# ── Main trainer ──────────────────────────────────────────────────────────────

def build_model(taxonomy_sizes: dict[str, int], lite: bool = False) -> nn.Module:
    """
    Factory — returns SeabedLite (for laptop demo) or SeabedDetector (full).

    Args:
        taxonomy_sizes: {level: num_classes} derived from the dataset.
        lite:           If True, returns SeabedLite (~4M params, MPS-friendly).
    """
    if lite:
        from model.lite.detector_lite import SeabedLite
        from core.config import (
            LITE_D_MODEL, LITE_NUM_QUERIES,
            LITE_DECODER_LAYERS, LITE_DECODER_HEADS,
        )
        return SeabedLite(
            taxonomy_sizes=taxonomy_sizes,
            d_model=LITE_D_MODEL,
            num_queries=LITE_NUM_QUERIES,
            decoder_layers=LITE_DECODER_LAYERS,
            decoder_heads=LITE_DECODER_HEADS,
        )
    return SeabedDetector(taxonomy_sizes=taxonomy_sizes)


def train(
    annotation_path: str,
    image_dir: str,
    checkpoint_dir: str = "checkpoints",
    total_epochs: int = 80,
    warmup_epochs: int = 10,
    batch_size: int = 8,
    head_lr: float = 1e-4,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    num_workers: int = 0,
    max_samples: int | None = None,
    lite: bool = False,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH) or ".", exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Training on %s", device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = SeabedDataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        transforms=UnderwaterAugmentation(),
    )

    # Capture taxonomy vocab before any subsetting
    taxonomy_sizes = {
        level: len(full_dataset.taxonomy_labels[level])
        for level in TAXONOMY_LEVELS
    }

    # Optionally cap dataset size (e.g. smoke-test)
    if max_samples is not None and max_samples < len(full_dataset):
        indices = torch.randperm(len(full_dataset))[:max_samples].tolist()
        full_dataset = Subset(full_dataset, indices)

    n_val   = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    # Validation set should not have augmentations — re-create without them
    val_dataset   = SeabedDataset(annotation_path=annotation_path, image_dir=image_dir)
    val_set_clean = Subset(val_dataset, val_set.indices)

    pin      = torch.cuda.is_available()   # pin_memory unsupported on MPS / CPU
    persist  = num_workers > 0             # keep worker processes alive between epochs
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin, persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_set_clean, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=pin, persistent_workers=persist,
    )

    # ── Model + loss ──────────────────────────────────────────────────────────
    model = build_model(taxonomy_sizes, lite=lite).to(device)
    criterion = DetectionLoss()
    weights_out = LITE_WEIGHTS_PATH if lite else MODEL_WEIGHTS_PATH
    logger.info("Model: %s  |  Output weights: %s", "SeabedLite" if lite else "SeabedDetector", weights_out)

    # GradScaler only on CUDA (MPS does not support it)
    scaler = torch.cuda.GradScaler() if device.type == "cuda" else None

    best_val_loss = float("inf")

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    logger.info("Phase 1 — backbone frozen (%d epochs)", warmup_epochs)
    for param in model.backbone.parameters():
        param.requires_grad_(False)

    optimizer = _build_optimizer(model, backbone_lr, head_lr, weight_decay, freeze_backbone=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=head_lr * 0.1)

    for epoch in range(warmup_epochs):
        train_losses = _train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_losses   = _val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  (cls=%.4f l1=%.4f giou=%.4f conf=%.4f)",
            epoch + 1, total_epochs,
            train_losses["total"], val_losses["total"],
            val_losses["loss_cls"], val_losses["loss_l1"],
            val_losses["loss_giou"], val_losses["loss_conf"],
        )

        _save_checkpoint(model, checkpoint_dir, epoch)
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            _save_checkpoint(model, checkpoint_dir, epoch, name="best.pt")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    # Flush all pending MPS ops before the computation graph changes.
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    logger.info("Phase 2 — full fine-tune (%d epochs)", total_epochs - warmup_epochs)
    for param in model.backbone.parameters():
        param.requires_grad_(True)

    optimizer = _build_optimizer(model, backbone_lr, head_lr, weight_decay, freeze_backbone=False)
    remaining = total_epochs - warmup_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=backbone_lr * 0.1)

    for epoch in range(warmup_epochs, total_epochs):
        train_losses = _train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_losses   = _val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  (cls=%.4f l1=%.4f giou=%.4f conf=%.4f)",
            epoch + 1, total_epochs,
            train_losses["total"], val_losses["total"],
            val_losses["loss_cls"], val_losses["loss_l1"],
            val_losses["loss_giou"], val_losses["loss_conf"],
        )

        _save_checkpoint(model, checkpoint_dir, epoch)
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            _save_checkpoint(model, checkpoint_dir, epoch, name="best.pt")

    # ── Prototype population (full model only) ─────────────────────────────────
    if not lite:
        logger.info("Populating novelty prototypes from training embeddings …")
        proto_loader = DataLoader(
            SeabedDataset(annotation_path=annotation_path, image_dir=image_dir),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn,
        )
        _update_prototypes(model, proto_loader, device)

    # ── Export final weights ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(weights_out) or ".", exist_ok=True)
    torch.save(model.state_dict(), weights_out)
    logger.info("Final weights saved to %s", weights_out)

    best_src = os.path.join(checkpoint_dir, "best.pt")
    if os.path.isfile(best_src):
        logger.info("Best checkpoint: val_loss=%.4f", best_val_loss)


def _save_checkpoint(
    model: nn.Module,
    directory: str,
    epoch: int,
    name: str | None = None,
) -> None:
    filename = name or f"epoch_{epoch + 1:03d}.pt"
    path = os.path.join(directory, filename)
    torch.save(model.state_dict(), path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SeabedDetector or SeabedLite")
    parser.add_argument("--annotation-path", default="data/annotations.json")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (default 2; safe on macOS with this __main__ guard)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap dataset size for smoke-tests")
    parser.add_argument("--lite", action="store_true",
                        help="Train SeabedLite (~4M params) instead of the full model. "
                             "Recommended for laptop / Apple Silicon MPS.")
    args = parser.parse_args()

    train(
        annotation_path=args.annotation_path,
        image_dir=args.image_dir,
        checkpoint_dir=args.checkpoint_dir,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        lite=args.lite,
    )
