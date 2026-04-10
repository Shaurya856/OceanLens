"""
Lightweight neck: single top-down FPN.

Replaces BiFPN (3 bidirectional iterations, 4 levels, 256 channels) with a
simple top-down FPN (1 pass, 3 levels, LITE_D_MODEL=128 channels).

Data flow:
    P5 → lateral_5 (1×1 proj) → top_down
    P4 → lateral_4 (1×1 proj) + upsample(P5_td) → out_4
    P3 → lateral_3 (1×1 proj) + upsample(P4_td) → out_3

Each level gets a 3×3 output conv to blend the merged features.

Input:  [P3, P4, P5] with arbitrary channel widths (from LiteBackbone).
Output: [P3, P4, P5] all with d_model channels — same interface as BiFPN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteFPN(nn.Module):
    """
    Args:
        in_channels: (C3, C4, C5) from the backbone — queried via
                     LiteBackbone.out_channels.
        d_model:     Output channel width for all FPN levels (default 128).
    """

    def __init__(self, in_channels: tuple[int, ...], d_model: int = 128) -> None:
        super().__init__()
        c3, c4, c5 = in_channels

        # Lateral 1×1 projections: each backbone level → d_model
        self.lat5 = nn.Conv2d(c5, d_model, 1, bias=False)
        self.lat4 = nn.Conv2d(c4, d_model, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, d_model, 1, bias=False)

        # Output 3×3 smoothing convs
        self.out5 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] from LiteBackbone.
        Returns:
            [P3, P4, P5] all [B, d_model, H_i, W_i].
        """
        p3, p4, p5 = features

        # Lateral projections
        l5 = self.lat5(p5)   # [B, d, H/32, W/32]
        l4 = self.lat4(p4)   # [B, d, H/16, W/16]
        l3 = self.lat3(p3)   # [B, d, H/8,  W/8]

        # Top-down merge
        td4 = l4 + F.interpolate(l5, size=l4.shape[-2:], mode="nearest")
        td3 = l3 + F.interpolate(td4, size=l3.shape[-2:], mode="nearest")

        return [self.out3(td3), self.out4(td4), self.out5(l5)]
