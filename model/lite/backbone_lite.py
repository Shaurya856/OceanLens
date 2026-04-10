"""
Lightweight backbone: MobileNetV3-Small.

Replaces the heavy DualPathBackbone (ConvNeXt-S + Swin-T, ~80M params)
with a single MobileNetV3-Small (~2.5M params) suitable for training and
inference on Apple Silicon MPS or CPU.

Feature maps are extracted at out_indices (2, 3, 4), corresponding to
strides 8 / 16 / 32.  Channel sizes are read dynamically from timm's
feature_info so this module does not hardcode backbone internals.

Output: [P3, P4, P5] — same interface as DualPathBackbone.
"""
import torch
import torch.nn as nn
import timm


class LiteBackbone(nn.Module):
    """
    Returns three feature maps [P3, P4, P5] at strides [8, 16, 32].

    Attribute `out_channels` exposes the channel count at each scale so
    the neck can project them without hardcoding numbers here.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),   # stride 8, 16, 32
        )
        # (24, 48, 96) for mobilenetv3_small_100 — derived at construction
        # time so the neck can query self.out_channels without a forward pass.
        self.out_channels: tuple[int, ...] = tuple(
            self.backbone.feature_info.channels()
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] — normalised RGB, H=W=320 recommended.
        Returns:
            [P3, P4, P5]  shapes [B, C3, H/8, W/8], [B, C4, H/16, W/16],
                                  [B, C5, H/32, W/32].
        """
        return self.backbone(x)   # list of 3 tensors, already NCHW
