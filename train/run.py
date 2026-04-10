"""
Training entry point.

Usage
─────
python train/run.py \\
    --annotations data/annotations.json \\
    --image-dir   data/images/ \\
    --checkpoint-dir checkpoints/ \\
    --epochs 80 \\
    --warmup-epochs 10 \\
    --batch-size 8 \\
    --head-lr 1e-4 \\
    --backbone-lr 1e-5

Annotation format
─────────────────
See train/dataset.py for the full JSON schema.
A minimal example is provided in data/sample_annotations.json.
"""
import argparse
from train.trainer import train


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SeabedDetector")
    p.add_argument("--annotations",    required=True,  help="Path to annotation JSON file")
    p.add_argument("--image-dir",      required=True,  help="Directory containing training images")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Where to save checkpoints")
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--warmup-epochs",  type=int,   default=10,  help="Epochs with backbone frozen")
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--head-lr",        type=float, default=1e-4, help="LR for neck/decoder/heads")
    p.add_argument("--backbone-lr",    type=float, default=1e-5, help="LR for backbone (phase 2)")
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--val-split",      type=float, default=0.1,  help="Fraction held out for val")
    p.add_argument("--num-workers",    type=int,   default=0,
                   help="DataLoader workers (default 0; use 4+ on Linux with GPU)")
    p.add_argument("--smoke-test",     action="store_true",
                   help="Run 2 epochs on 32 samples to verify the pipeline end-to-end")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.smoke_test:
        print("Smoke-test mode: 2 epochs, batch=2, 32 samples")
        train(
            annotation_path=args.annotations,
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
        )
    else:
        train(
            annotation_path=args.annotations,
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
        )
