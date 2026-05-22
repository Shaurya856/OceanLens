"""
DETR-style transformer decoder.

Memory tokens are the concatenated (flattened) feature maps from P3, P4, P5
enriched with 2D sinusoidal positional encodings. N learned object queries
cross-attend to this memory over `num_layers` decoder layers.

Output: query embeddings [B, num_queries, d_model].
"""
import math

import torch
import torch.nn as nn


def _make_2d_sinusoidal_pe(H: int, W: int, C: int, device) -> torch.Tensor:
    """Return [H*W, C] 2D sinusoidal positional encodings."""
    assert C % 2 == 0
    half = C // 2
    div = torch.exp(
        torch.arange(0, half, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / half)
    )
    y = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)  # [H, 1]
    x = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1)  # [W, 1]

    pe_y = torch.zeros(H, half, device=device)
    pe_y[:, 0::2] = torch.sin(y * div)
    pe_y[:, 1::2] = torch.cos(y * div)

    pe_x = torch.zeros(W, half, device=device)
    pe_x[:, 0::2] = torch.sin(x * div)
    pe_x[:, 1::2] = torch.cos(x * div)

    pe = torch.cat([
        pe_y.unsqueeze(1).expand(-1, W, -1),
        pe_x.unsqueeze(0).expand(H, -1, -1),
    ], dim=-1)
    return pe.reshape(H * W, C)


# ── Decoder ───────────────────────────────────────────────────────────────────

class DETRDecoder(nn.Module):
    """
    Args:
        d_model:     Feature / embedding dimension.
        nhead:       Number of attention heads.
        num_layers:  Number of stacked decoder layers.
        num_queries: Number of object queries.
        dropout:     Dropout rate inside attention / FFN.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        num_queries: int = 300,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        # Learnable object queries (the "tgt" for the TransformerDecoder)
        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm for training stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [P3, P4, P5] each [B, d_model, H_i, W_i].
        Returns:
            query_embeddings: [B, num_queries, d_model]
        """
        B = features[0].shape[0]
        device = features[0].device

        # Build memory: flatten each scale, add 2-D PE, concatenate
        memory_parts: list[torch.Tensor] = []
        for feat in features:
            _, C, H, W = feat.shape
            pe = _make_2d_sinusoidal_pe(H, W, C, device)  # [H*W, C]
            tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]
            memory_parts.append(tokens + pe.unsqueeze(0))

        memory  = torch.cat(memory_parts, dim=1).contiguous()  # [B, total_tokens, d_model]

        # Expand object queries to batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, N, d_model]

        return self.decoder(tgt=queries, memory=memory)  # [B, N, d_model]
