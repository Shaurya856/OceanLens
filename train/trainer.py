"""
Two-phase training loop for SeabedDetector / SeabedLite.

Phase 1 — Backbone frozen  (epochs 0 … warmup_epochs-1)
    Only neck, decoder, and all heads are trained.
    Larger effective LR is safe since backbone features are stable.

Phase 2 — Full fine-tune    (epochs warmup_epochs … total_epochs-1)
    Backbone unfrozen with 10× smaller LR (layer-wise decay).
    Gradient clipping at max_norm=0.1 (standard for DETR-family models).

Checkpoints
───────────
Saved to  {checkpoint_dir}/epoch_{n:03d}.pt  after every epoch.
Best model (lowest val loss) saved to  {checkpoint_dir}/best.pt.
After training, detector weights are also copied to the appropriate
weights path so the inference pipeline picks them up immediately.

Prototype update
────────────────
After Phase 2 completes (SeabedDetector only), one pass over the
training set populates novelty_detector.prototypes from real embedding
distributions.
"""
import logging
import os

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
from core.utils import get_device
from model.detector import SeabedDetector
from train.dataset import SeabedDataset, collate_fn
from train.augmentations import UnderwaterAugmentation
from train.loss import DetectionLoss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Optimizer builders ────────────────────────────────────────────────────────

def _build_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    freeze_backbone: bool,
) -> AdamW:
    """
    Parameter groups:
      • Backbone                           → backbone_lr  (or 0 when frozen)
      • Everything else                    → head_lr

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
        # BF16 is native on Ampere (SM≥80) and has a wider dynamic range than
        # FP16 — no overflow risk, no GradScaler needed.  Fall back to FP16 on
        # older Volta/Turing cards (SM<80) where BF16 is not supported.
        cap = torch.cuda.get_device_capability(device)[0]
        dtype = torch.bfloat16 if cap >= 8 else torch.float16
        return torch.autocast("cuda", dtype=dtype)
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
    scaler: "torch.amp.GradScaler | None" = None,
    phylo_criterion=None,
    max_norm: float = 0.1,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    ac = _autocast(device)

    for images, targets in loader:
        images = images.to(device)

        with ac:
            outputs = model(images)
            losses  = dict(criterion(outputs, targets))
            if phylo_criterion is not None:
                loss_phylo = phylo_criterion(outputs["class_logits"])
                losses["loss_phylo"] = loss_phylo
                losses["total"] = losses["total"] + loss_phylo

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
    phylo_criterion=None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    ac = _autocast(device)

    for images, targets in loader:
        images = images.to(device)
        with ac:
            outputs = model(images)
            losses  = dict(criterion(outputs, targets))
            if phylo_criterion is not None:
                loss_phylo = phylo_criterion(outputs["class_logits"])
                losses["loss_phylo"] = loss_phylo
                losses["total"] = losses["total"] + loss_phylo
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def _log_losses(epoch: int, total_epochs: int, train: dict, val: dict) -> None:
    extras = "  ".join(
        f"{k}={val[k]:.4f}"
        for k in ("loss_cls", "loss_l1", "loss_giou", "loss_conf", "loss_phylo")
        if k in val
    )
    logger.info(
        "Epoch %3d/%d  train=%.4f  val=%.4f  (%s)",
        epoch + 1, total_epochs,
        train["total"], val["total"],
        extras,
    )


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
            pred_sp = species_pred[b][mask]
            gt_flat = gt_species.unique()
            valid = torch.isin(pred_sp, gt_flat)
            if not valid.any():
                continue
            model.novelty.update_prototypes(emb[valid], pred_sp[valid])

    logger.info("Prototypes updated for %d species", (model.novelty.prototype_counts > 0).sum().item())


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    taxonomy_sizes: dict[str, int],
    lite: bool = False,
) -> nn.Module:
    """
    Factory — returns one of two model variants.

    Args:
        taxonomy_sizes: {level: num_classes} derived from the dataset.
        lite:           SeabedLite (~4M params, MPS-friendly).
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
    compile_model: bool = False,
    resume: str | None = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = get_device()
    logger.info("Training on %s", device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = SeabedDataset(
        annotation_path=annotation_path,
        image_dir=image_dir,
        transforms=UnderwaterAugmentation(),
    )
    # Persist vocabulary once so the inference runner can load it later.
    full_dataset.save_vocabulary()

    taxonomy_sizes = {
        level: len(full_dataset.taxonomy_labels[level])
        for level in TAXONOMY_LEVELS
    }

    if max_samples is not None and max_samples < len(full_dataset):
        indices = torch.randperm(len(full_dataset))[:max_samples].tolist()
        full_dataset = Subset(full_dataset, indices)

    n_val   = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    val_dataset   = SeabedDataset(annotation_path=annotation_path, image_dir=image_dir)
    val_set_clean = Subset(val_dataset, val_set.indices)

    pin     = torch.cuda.is_available()
    persist = num_workers > 0
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

    if lite:
        weights_out = LITE_WEIGHTS_PATH
        model_name  = "SeabedLite"
    else:
        weights_out = MODEL_WEIGHTS_PATH
        model_name  = "SeabedDetector"

    logger.info("Model: %s  |  Output weights: %s", model_name, weights_out)

    if compile_model:
        logger.info("Compiling model with torch.compile (first epoch will be slower) …")
        model = torch.compile(model)

    # GradScaler is only needed for FP16 (overflow risk).
    # Ampere+ (SM≥80) uses BF16 which has a wider dynamic range — skip the scaler.
    cuda_cap = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    use_fp16 = device.type == "cuda" and cuda_cap < 8
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None
    best_val_loss = float("inf")
    start_epoch   = 0

    # ── Resume from checkpoint ────────────────────────────────────────────────
    resume_ckpt: dict | None = None
    if resume is not None:
        resume_ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        start_epoch   = resume_ckpt["epoch"] + 1
        best_val_loss = resume_ckpt["best_val_loss"]
        logger.info("Resumed from %s  (epoch %d, best_val_loss=%.4f)", resume, start_epoch, best_val_loss)

    # ── Phase 1: frozen backbone ───────────────────────────────────────────────
    phase1_start = start_epoch
    phase1_end   = warmup_epochs

    if phase1_start < phase1_end:
        logger.info("Phase 1 — backbone frozen (epochs %d–%d)", phase1_start + 1, phase1_end)
        for param in model.backbone.parameters():
            param.requires_grad_(False)

        optimizer = _build_optimizer(model, backbone_lr, head_lr, weight_decay, freeze_backbone=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, warmup_epochs), eta_min=head_lr * 0.1)

        if resume_ckpt is not None and start_epoch < warmup_epochs:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            if scaler is not None and resume_ckpt.get("scaler") is not None:
                scaler.load_state_dict(resume_ckpt["scaler"])

        for epoch in range(phase1_start, phase1_end):
            train_losses = _train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_losses   = _val_epoch(model, val_loader, criterion, device)
            scheduler.step()

            _log_losses(epoch, total_epochs, train_losses, val_losses)
            _save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, checkpoint_dir)
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, checkpoint_dir, name="best.pt")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    phase2_start = max(start_epoch, warmup_epochs)
    phase2_end   = total_epochs

    if phase2_start < phase2_end:
        logger.info("Phase 2 — full fine-tune (epochs %d–%d)", phase2_start + 1, phase2_end)
        for param in model.backbone.parameters():
            param.requires_grad_(True)

        remaining = max(1, total_epochs - warmup_epochs)
        optimizer = _build_optimizer(model, backbone_lr, head_lr, weight_decay, freeze_backbone=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=backbone_lr * 0.1)

        if resume_ckpt is not None and start_epoch >= warmup_epochs:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            if scaler is not None and resume_ckpt.get("scaler") is not None:
                scaler.load_state_dict(resume_ckpt["scaler"])

        for epoch in range(phase2_start, phase2_end):
            train_losses = _train_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_losses   = _val_epoch(model, val_loader, criterion, device)
            scheduler.step()

            _log_losses(epoch, total_epochs, train_losses, val_losses)
            _save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, checkpoint_dir)
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_val_loss, checkpoint_dir, name="best.pt")

    # ── Prototype population (SeabedDetector only) ────────────────────────────
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
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: "torch.amp.GradScaler | None",
    epoch: int,
    best_val_loss: float,
    directory: str,
    name: str | None = None,
) -> None:
    filename = name or f"epoch_{epoch + 1:03d}.pt"
    path = os.path.join(directory, filename)
    torch.save({
        "epoch":         epoch,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "scaler":        scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
    }, path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SeabedDetector or SeabedLite"
    )
    parser.add_argument("--annotation-path", default="data/annotations.json")
    parser.add_argument("--image-dir",        default="data/images")
    parser.add_argument("--checkpoint-dir",   default="checkpoints")
    parser.add_argument("--epochs",           type=int,   default=80)
    parser.add_argument("--warmup-epochs",    type=int,   default=10)
    parser.add_argument("--batch-size",       type=int,   default=8)
    parser.add_argument("--head-lr",          type=float, default=1e-4)
    parser.add_argument("--backbone-lr",      type=float, default=1e-5)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--val-split",        type=float, default=0.1)
    parser.add_argument("--num-workers",      type=int,   default=2,
                        help="DataLoader workers (safe on macOS with this __main__ guard)")
    parser.add_argument("--max-samples",      type=int,   default=None,
                        help="Cap dataset size for smoke-tests")

    # Model variant (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lite", action="store_true",
                       help="Train SeabedLite (~4M params, laptop/MPS)")

    parser.add_argument("--resume", default=None, metavar="CHECKPOINT",
                        help="Path to a checkpoint to resume from (e.g. checkpoints/epoch_045.pt)")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap model with torch.compile for ~20-30%% throughput gain "
                             "(requires PyTorch 2.4+; first epoch ~30-60s slower for compilation)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 epochs on 32 samples to verify the pipeline end-to-end")

    args = parser.parse_args()

    if args.smoke_test:
        print("Smoke-test mode: 2 epochs, batch=2, 32 samples")
        train(
            annotation_path=args.annotation_path,
            image_dir=args.image_dir,
            checkpoint_dir=args.checkpoint_dir,
            total_epochs=2,
            warmup_epochs=1,
            batch_size=2,
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            val_split=0.25,
            num_workers=0,
            max_samples=32,
            lite=args.lite,
            compile_model=args.compile,
            resume=args.resume,
        )
    else:
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
            compile_model=args.compile,
            resume=args.resume,
        )
