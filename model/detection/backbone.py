"""
Dual-path backbone: ConvNeXt-Small (texture) + Swin-Tiny (context).

Both models are loaded from timm with pretrained ImageNet-21k/1k weights
and frozen during early training. They share the same output channel sizes
(192 / 384 / 768 at strides 8 / 16 / 32), so element-wise weighted fusion
is applied at each scale, followed by CBAM attention.
"""
import os

import torch
import torch.nn as nn
import timm
from core.config import MODEL_INPUT_SIZE

# Optional: set CONVNEXT_LOCAL_WEIGHTS to a folder containing model.safetensors
# to bypass HuggingFace and load ConvNeXt from disk.
_CONVNEXT_LOCAL = os.getenv("CONVNEXT_LOCAL_WEIGHTS")


# ── Attention ─────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ── Dual-path backbone ────────────────────────────────────────────────────────

# Channel sizes at out_indices (1, 2, 3) for both ConvNeXt-S and Swin-Tiny.
_STAGE_CHANNELS = (192, 384, 768)


class DualPathBackbone(nn.Module):
    """
    Returns three feature maps [P3, P4, P5] at strides [8, 16, 32].
    Output channels are preserved at 192 / 384 / 768.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained and _CONVNEXT_LOCAL:
            _convnext_cfg = {"file": os.path.join(_CONVNEXT_LOCAL, "model.safetensors")}
        else:
            _convnext_cfg = {}
        self.convnext = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
            pretrained_cfg_overlay=_convnext_cfg or None,
        )
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=MODEL_INPUT_SIZE,
        )
        self.cbam = nn.ModuleList([CBAM(ch) for ch in _STAGE_CHANNELS])
        # Per-scale learned fusion weights (softmax-normalised)
        self.fusion_w = nn.ParameterList(
            [nn.Parameter(torch.ones(2)) for _ in _STAGE_CHANNELS]
        )

    @staticmethod
    def _to_nchw(feat: torch.Tensor, known_channels: int) -> torch.Tensor:
        """Permute NHWC → NCHW when timm returns Swin outputs in channel-last."""
        if feat.dim() == 4 and feat.shape[1] != known_channels:
            feat = feat.permute(0, 3, 1, 2)
        return feat.contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        cnn = self.convnext(x)  # [(B,192,H/8,W/8), (B,384,…), (B,768,…)]
        swn = self.swin(x)

        out: list[torch.Tensor] = []
        for i, (cf, sf) in enumerate(zip(cnn, swn)):
            ch = _STAGE_CHANNELS[i]
            sf = self._to_nchw(sf, ch)
            w = self.fusion_w[i].softmax(0)
            fused = (w[0] * cf + w[1] * sf).contiguous()
            out.append(self.cbam[i](fused))

        return out  # [P3, P4, P5]
