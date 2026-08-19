"""Attention-guided Fusion Iteration Unit (FIU)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LBPStyleEncoder(nn.Module):
    """Differentiable local-binary-pattern-style encoder.

    Eight neighbour-to-centre comparisons are generated per input channel and
    processed by a small shared convolutional encoder at every iteration.
    """

    def __init__(self, input_channels: int = 2, hidden_channels: int = 32) -> None:
        super().__init__()
        self.input_channels = input_channels
        encoded_channels = input_channels * 9
        self.encoder = nn.Sequential(
            nn.Conv2d(encoded_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 1, hidden_channels),
            nn.GELU(),
        )
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _normalize(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_float = valid.to(value.dtype)
        count = valid_float.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        mean = (value * valid_float).sum(dim=(-2, -1), keepdim=True) / count
        variance = ((value - mean).square() * valid_float).sum(
            dim=(-2, -1), keepdim=True
        ) / count
        normalized = (value - mean) * torch.rsqrt(variance + 1e-5)
        return torch.where(valid, normalized, torch.zeros_like(normalized))

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.input_channels:
            raise ValueError(
                f"expected [B, {self.input_channels}, H, W], got {tuple(value.shape)}"
            )
        if valid.shape[1] == 1 and value.shape[1] > 1:
            valid = valid.expand(-1, value.shape[1], -1, -1)
        normalized = self._normalize(value, valid)
        batch, channels, height, width = normalized.shape
        patches = F.unfold(normalized, kernel_size=3, padding=1).reshape(
            batch, channels, 9, height, width
        )
        centre = patches[:, :, 4:5]
        neighbour_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        temperature = self.log_temperature.exp().clamp_min(1e-3)
        comparisons = torch.tanh(
            (patches[:, :, neighbour_indices] - centre) / temperature
        )
        encoded = torch.cat([normalized.unsqueeze(2), comparisons], dim=2)
        return self.encoder(encoded.reshape(batch, channels * 9, height, width))


class FusionIterationUnit(nn.Module):
    """Use aligned monocular geometry to gate each stereo residual update."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.encoder = LBPStyleEncoder(2, hidden_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden_channels % 8 == 0 else 1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        aligned_depth: Optional[torch.Tensor],
        disparity: torch.Tensor,
        stereo_residual: torch.Tensor,
        valid_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if aligned_depth is None:
            attention = torch.ones_like(stereo_residual)
            return stereo_residual, attention
        if aligned_depth.shape[-2:] != disparity.shape[-2:]:
            aligned_depth = F.interpolate(
                aligned_depth, size=disparity.shape[-2:], mode="bilinear", align_corners=False
            )
        if valid_prior is None:
            valid_prior = torch.isfinite(aligned_depth) & (aligned_depth > 0)
        elif valid_prior.shape[-2:] != disparity.shape[-2:]:
            valid_prior = F.interpolate(
                valid_prior.float(), size=disparity.shape[-2:], mode="nearest"
            ).bool()
        valid_disparity = torch.isfinite(disparity)
        encoder_valid = torch.cat([valid_prior, valid_disparity], dim=1)
        encoded = self.encoder(torch.cat([aligned_depth, disparity], dim=1), encoder_valid)
        predicted_attention = self.attention(
            torch.cat([encoded, valid_prior.to(encoded.dtype)], dim=1)
        )
        # Outside the calibrated/projected support FIU is an exact identity,
        # preserving the original stereo updater rather than suppressing it.
        attention = torch.where(
            valid_prior, predicted_attention, torch.ones_like(predicted_attention)
        )
        return stereo_residual * attention, attention
