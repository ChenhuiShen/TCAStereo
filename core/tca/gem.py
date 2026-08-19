"""Frequency-domain Geometric Enhancement Module (GEM)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FrequencyGeometryBlock(nn.Module):
    """Apply learnable complex filters to a Gaussian high-pass residual.

    The learned complex kernels are stored on a compact reference grid and are
    interpolated to the current feature resolution. This keeps GEM compatible
    with arbitrary input sizes while implementing Eq. (1) from the paper.
    """

    def __init__(
        self,
        channels: int,
        num_kernels: int = 4,
        reference_frequency_size: Tuple[int, int] = (16, 9),
        gaussian_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_kernels <= 0:
            raise ValueError("channels and num_kernels must be positive")
        self.channels = channels
        self.num_kernels = num_kernels
        self.reference_frequency_size = reference_frequency_size

        real = torch.full(
            (num_kernels, channels, *reference_frequency_size),
            1.0 / float(num_kernels),
        )
        self.frequency_weight_real = nn.Parameter(real)
        self.frequency_weight_imag = nn.Parameter(torch.zeros_like(real))

        coords = torch.arange(3, dtype=torch.float32) - 1.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * gaussian_sigma**2))
        gaussian /= gaussian.sum()
        self.register_buffer("gaussian_kernel", gaussian.view(1, 1, 3, 3), persistent=False)

        groups = _group_count(channels)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.residual_gain = nn.Parameter(torch.tensor(0.1))

    def _frequency_weights(self, height: int, width_rfft: int) -> torch.Tensor:
        target = (height, width_rfft)
        real = self.frequency_weight_real.reshape(
            self.num_kernels * self.channels, 1, *self.reference_frequency_size
        )
        imag = self.frequency_weight_imag.reshape_as(real)
        real = F.interpolate(real, size=target, mode="bilinear", align_corners=False)
        imag = F.interpolate(imag, size=target, mode="bilinear", align_corners=False)
        real = real.reshape(self.num_kernels, self.channels, *target)
        imag = imag.reshape_as(real)
        return torch.complex(real, imag)

    @staticmethod
    def _per_pixel_normalize(value: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        return (value - mean) * torch.rsqrt(variance + eps)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"expected [B, {self.channels}, H, W], got {tuple(feature.shape)}"
            )
        batch, channels, height, width = feature.shape
        original_dtype = feature.dtype

        # FFT kernels are not consistently supported in fp16, so GEM computes
        # the frequency branch in fp32 and casts the result back afterward.
        feature_float = feature.float()
        gaussian = self.gaussian_kernel.to(device=feature.device, dtype=feature_float.dtype)
        blurred = F.conv2d(
            feature_float,
            gaussian.expand(channels, 1, 3, 3),
            padding=1,
            groups=channels,
        )
        high_pass = feature_float - blurred
        spectrum = torch.fft.rfft2(high_pass, norm="ortho")
        weights = self._frequency_weights(height, spectrum.shape[-1]).to(spectrum.device)
        filtered = torch.fft.irfft2(
            spectrum.unsqueeze(1) * weights.unsqueeze(0),
            s=(height, width),
            norm="ortho",
        ).sum(dim=1)
        geometry = self._per_pixel_normalize(filtered).to(original_dtype)
        residual = self.mix(torch.cat([feature, geometry], dim=1))
        return feature + self.residual_gain.to(feature.dtype) * residual


class GeometricEnhancementModule(nn.Module):
    """Multi-scale GEM for the 1/4, 1/8, 1/16 and 1/32 features."""

    def __init__(
        self,
        channels: Sequence[int] = (128, 192, 448, 384),
        num_kernels: int = 4,
        reference_frequency_size: Tuple[int, int] = (16, 9),
    ) -> None:
        super().__init__()
        self.channels = tuple(int(value) for value in channels)
        self.blocks = nn.ModuleList(
            [
                FrequencyGeometryBlock(
                    value,
                    num_kernels=num_kernels,
                    reference_frequency_size=reference_frequency_size,
                )
                for value in self.channels
            ]
        )

    def forward(self, features: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        features = list(features)
        if len(features) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} pyramid levels, got {len(features)}"
            )
        return [block(feature) for block, feature in zip(self.blocks, features)]
