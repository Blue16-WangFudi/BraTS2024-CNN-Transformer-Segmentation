from __future__ import annotations

import math

import torch
import torch.nn as nn


class _LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        return x + self.pe[:, :n, :]


def _sinusoidal_pe(n: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(n, dim, device=device)
    position = torch.arange(0, n, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class SerialBottleneckTransformer(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        max_tokens: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.channels = int(channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.pos_encoding = str(pos_encoding).lower()
        self.max_tokens = int(max_tokens)
        self.learned_pe = _LearnedPositionalEncoding(max_len=self.max_tokens, dim=self.channels) if self.pos_encoding == "learned" else None
        self.fuse = nn.Sequential(
            nn.Conv3d(self.channels * 2, self.channels, kernel_size=1),
            nn.InstanceNorm3d(self.channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        n = tokens.shape[1]
        if self.pos_encoding == "learned":
            if self.learned_pe is None or n > self.max_tokens:
                raise ValueError(f"n_tokens={n} exceeds max_tokens={self.max_tokens} for learned PE.")
            tokens = self.learned_pe(tokens)
        elif self.pos_encoding == "sin":
            tokens = tokens + _sinusoidal_pe(n, c, tokens.device)

        y = self.encoder(tokens)
        y3d = y.transpose(1, 2).reshape(b, c, d, h, w)
        return self.fuse(torch.cat([x, y3d], dim=1))


class EmbeddedTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        max_tokens: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.channels = int(channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channels,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.pos_encoding = str(pos_encoding).lower()
        self.max_tokens = int(max_tokens)
        self.learned_pe = _LearnedPositionalEncoding(max_len=self.max_tokens, dim=self.channels) if self.pos_encoding == "learned" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        n = tokens.shape[1]
        if self.pos_encoding == "learned":
            if self.learned_pe is None or n > self.max_tokens:
                raise ValueError(f"n_tokens={n} exceeds max_tokens={self.max_tokens} for learned PE.")
            tokens = self.learned_pe(tokens)
        elif self.pos_encoding == "sin":
            tokens = tokens + _sinusoidal_pe(n, c, tokens.device)

        y = self.encoder(tokens).transpose(1, 2).reshape(b, c, d, h, w)
        return x + y


class ParallelFusionPlaceholder(nn.Module):
    """
    Placeholder interface for a future parallel CNN+Transformer fusion mode.

    This mode is intentionally not implemented yet; selecting `fusion_mode="parallel"`
    fails fast instead of silently producing incorrect results.
    """

    def forward(self, x: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        raise NotImplementedError("fusion_mode='parallel' is not implemented.")
