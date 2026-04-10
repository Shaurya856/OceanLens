"""
BiFPN (Bidirectional Feature Pyramid Network).

Accepts [P3, P4, P5] at strides [8, 16, 32] and emits [P3, P4, P5, P6]
at strides [8, 16, 32, 64] all with uniform `d_model` channels.

The bidirectional paths use fast-normalised fusion (softmax weights)
with depthwise-separable convolutions, matching the EfficientDet design.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Backbone stage channels → BiFPN d_model mapping
_IN_CHANNELS = (192, 384, 768)


class _DepthwiseSepConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class _BiFPNLayer(nn.Module):
    """One BiFPN iteration over four scales: P3, P4, P5, P6."""

    _EPS = 1e-4

    def __init__(self, d: int) -> None:
        super().__init__()
        # Top-down weights (2 inputs each)
        self.w_td5 = nn.Parameter(torch.ones(2))  # P5_td ← P5, upsample(P6)
        self.w_td4 = nn.Parameter(torch.ones(2))  # P4_td ← P4, upsample(P5_td)
        self.w_td3 = nn.Parameter(torch.ones(2))  # P3_out ← P3, upsample(P4_td)
        # Bottom-up weights
        self.w_bu4 = nn.Parameter(torch.ones(3))  # P4_out ← P4, P4_td, down(P3_out)
        self.w_bu5 = nn.Parameter(torch.ones(3))  # P5_out ← P5, P5_td, down(P4_out)
        self.w_bu6 = nn.Parameter(torch.ones(2))  # P6_out ← P6, down(P5_out)

        self.conv_td5 = _DepthwiseSepConv(d)
        self.conv_td4 = _DepthwiseSepConv(d)
        self.conv_p3  = _DepthwiseSepConv(d)
        self.conv_p4  = _DepthwiseSepConv(d)
        self.conv_p5  = _DepthwiseSepConv(d)
        self.conv_p6  = _DepthwiseSepConv(d)

    @staticmethod
    def _fuse(tensors: list[torch.Tensor], weights: torch.Tensor, eps: float) -> torch.Tensor:
        w = weights.relu() + eps
        w = w / w.sum()
        return sum(w[i] * t for i, t in enumerate(tensors))

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    @staticmethod
    def _down(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=2, stride=2)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
        p6: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Top-down ──────────────────────────────────────────────────────────
        p5_td = self.conv_td5(self._fuse([p5, self._up(p6, p5)], self.w_td5, self._EPS))
        p4_td = self.conv_td4(self._fuse([p4, self._up(p5_td, p4)], self.w_td4, self._EPS))
        p3_out = self.conv_p3(self._fuse([p3, self._up(p4_td, p3)], self.w_td3, self._EPS))

        # ── Bottom-up ─────────────────────────────────────────────────────────
        p4_out = self.conv_p4(self._fuse([p4, p4_td, self._down(p3_out)], self.w_bu4, self._EPS))
        p5_out = self.conv_p5(self._fuse([p5, p5_td, self._down(p4_out)], self.w_bu5, self._EPS))
        p6_out = self.conv_p6(self._fuse([p6, self._down(p5_out)], self.w_bu6, self._EPS))

        return p3_out, p4_out, p5_out, p6_out


class BiFPN(nn.Module):
    """
    Projects backbone features to d_model channels, creates P6, then
    runs `num_iters` BiFPN iterations.

    Returns [P3, P4, P5, P6] all with d_model channels.
    """

    def __init__(self, d_model: int = 256, num_iters: int = 3) -> None:
        super().__init__()
        # Input projections (backbone channels → d_model)
        self.proj = nn.ModuleList(
            [nn.Conv2d(ch, d_model, 1, bias=False) for ch in _IN_CHANNELS]
        )
        # P6 is created from P5 via strided conv
        self.p6_conv = nn.Conv2d(d_model, d_model, 3, stride=2, padding=1, bias=False)
        self.layers = nn.ModuleList([_BiFPNLayer(d_model) for _ in range(num_iters)])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = [self.proj[i](f) for i, f in enumerate(features)]
        p6 = self.p6_conv(p5)

        for layer in self.layers:
            p3, p4, p5, p6 = layer(p3, p4, p5, p6)

        return [p3, p4, p5, p6]
