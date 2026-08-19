"""Global Alignment Module (GAM) for calibrated tri-camera fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GAMResult:
    """Outputs and diagnostics from global monocular/stereo alignment."""

    aligned_depth_left: torch.Tensor
    valid_left: torch.Tensor
    aligned_depth_texture: torch.Tensor
    valid_texture: torch.Tensor
    stereo_depth_texture: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


def _as_batch_tensor(
    calibration: Mapping[str, torch.Tensor],
    key: str,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    default: Optional[float] = None,
) -> torch.Tensor:
    value = calibration.get(key)
    if value is None:
        if default is None:
            raise KeyError(f"calibration is missing {key}")
        value = torch.tensor(default)
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.repeat(batch)
    elif value.shape[0] == 1 and batch > 1:
        value = value.expand(batch, *value.shape[1:])
    if value.shape[0] != batch:
        raise ValueError(f"calibration {key} batch {value.shape[0]} != image batch {batch}")
    return value


class GlobalAlignmentModule(nn.Module):
    """Align a texture-camera relative-depth prior to stereo metric depth.

    RANSAC deliberately runs on detached tensors, matching Algorithm 1 in the
    paper. Projection and back-projection remain tensorized and work on CPU or
    CUDA without OpenCV.
    """

    def __init__(
        self,
        ransac_iterations: int = 128,
        ransac_threshold: float = 0.05,
        min_points: int = 32,
        max_points: int = 50000,
        min_depth: float = 1e-4,
        max_depth: float = 1e4,
        min_abs_disparity: float = 1e-3,
        hole_fill_iterations: int = 1,
    ) -> None:
        super().__init__()
        self.ransac_iterations = int(ransac_iterations)
        self.ransac_threshold = float(ransac_threshold)
        self.min_points = int(min_points)
        self.max_points = int(max_points)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.min_abs_disparity = float(min_abs_disparity)
        self.hole_fill_iterations = int(hole_fill_iterations)

    @staticmethod
    def _scale_intrinsics_batch(
        intrinsics: torch.Tensor,
        source_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        source_h, source_w = source_hw
        target_h, target_w = target_hw
        output = intrinsics.clone()
        output[:, 0, :] *= float(target_w) / float(source_w)
        output[:, 1, :] *= float(target_h) / float(source_h)
        output[:, 2, :] = intrinsics[:, 2, :]
        return output

    @staticmethod
    def _z_buffer(
        pixel_indices: torch.Tensor,
        depths: torch.Tensor,
        pixel_count: int,
    ) -> torch.Tensor:
        output = depths.new_full((pixel_count,), float("inf"))
        if pixel_indices.numel() == 0:
            return output
        if hasattr(output, "scatter_reduce_"):
            output.scatter_reduce_(
                0, pixel_indices.long(), depths, reduce="amin", include_self=True
            )
            return output
        # Compatibility fallback for older PyTorch versions.
        for index in torch.unique(pixel_indices):
            selected = depths[pixel_indices == index]
            output[index] = selected.min()
        return output

    def _project_depth(
        self,
        source_depth: torch.Tensor,
        K_source: torch.Tensor,
        T_source_to_target: torch.Tensor,
        K_target: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-project a depth map using a nearest-pixel z-buffer."""

        batch, _, source_h, source_w = source_depth.shape
        target_h, target_w = target_hw
        device, dtype = source_depth.device, source_depth.dtype
        yy, xx = torch.meshgrid(
            torch.arange(source_h, device=device, dtype=dtype),
            torch.arange(source_w, device=device, dtype=dtype),
            indexing="ij",
        )
        pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).reshape(1, 3, -1)
        pixels = pixels.expand(batch, -1, -1)

        rays = torch.linalg.inv(K_source) @ pixels
        depth_flat = source_depth.reshape(batch, 1, -1)
        points_source = rays * depth_flat
        rotation = T_source_to_target[:, :3, :3]
        translation = T_source_to_target[:, :3, 3:4]
        points_target = rotation @ points_source + translation
        z_target = points_target[:, 2]
        projected = K_target @ points_target
        denom = projected[:, 2].clamp_min(self.min_depth)
        u = projected[:, 0] / denom
        v = projected[:, 1] / denom

        outputs = []
        masks = []
        source_valid = (
            torch.isfinite(depth_flat[:, 0])
            & (depth_flat[:, 0] > self.min_depth)
            & (depth_flat[:, 0] < self.max_depth)
        )
        for batch_index in range(batch):
            u_round = torch.round(u[batch_index]).long()
            v_round = torch.round(v[batch_index]).long()
            valid = (
                source_valid[batch_index]
                & torch.isfinite(z_target[batch_index])
                & (z_target[batch_index] > self.min_depth)
                & (z_target[batch_index] < self.max_depth)
                & (u_round >= 0)
                & (u_round < target_w)
                & (v_round >= 0)
                & (v_round < target_h)
            )
            indices = v_round[valid] * target_w + u_round[valid]
            z_buffer = self._z_buffer(indices, z_target[batch_index][valid], target_h * target_w)
            mask = torch.isfinite(z_buffer)
            outputs.append(torch.where(mask, z_buffer, torch.zeros_like(z_buffer)))
            masks.append(mask)
        depth = torch.stack(outputs).reshape(batch, 1, target_h, target_w)
        mask = torch.stack(masks).reshape(batch, 1, target_h, target_w)
        return depth, mask

    @staticmethod
    def _linear_fit(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_mean = x.mean()
        y_mean = y.mean()
        variance = (x - x_mean).square().sum()
        if variance <= torch.finfo(x.dtype).eps:
            return x.new_zeros(()), y_mean
        alpha = ((x - x_mean) * (y - y_mean)).sum() / variance
        beta = y_mean - alpha * x_mean
        return alpha, beta

    def _ransac_fit(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        x = x.detach()
        y = y.detach()
        finite = torch.isfinite(x) & torch.isfinite(y)
        x, y = x[finite], y[finite]
        if x.numel() > self.max_points:
            keep = torch.linspace(
                0, x.numel() - 1, self.max_points, device=x.device
            ).long()
            x, y = x[keep], y[keep]
        if x.numel() < 2:
            return x.new_tensor(1.0), x.new_tensor(0.0), False
        if x.numel() < self.min_points or self.ransac_iterations <= 0:
            alpha, beta = self._linear_fit(x, y)
            return alpha.detach(), beta.detach(), x.numel() >= self.min_points

        first = torch.randint(x.numel(), (self.ransac_iterations,), device=x.device)
        second = torch.randint(x.numel(), (self.ransac_iterations,), device=x.device)
        dx = x[second] - x[first]
        usable = dx.abs() > torch.finfo(x.dtype).eps * 32
        if not usable.any():
            alpha, beta = self._linear_fit(x, y)
            return alpha.detach(), beta.detach(), False
        alpha_candidates = (y[second][usable] - y[first][usable]) / dx[usable]
        beta_candidates = y[first][usable] - alpha_candidates * x[first][usable]

        y_range = (y.max() - y.min()).detach()
        threshold = torch.maximum(
            y.new_tensor(self.ransac_threshold), y_range * 0.01
        )
        best_score = -1
        best_inliers: Optional[torch.Tensor] = None
        for start in range(0, alpha_candidates.numel(), 32):
            alpha_chunk = alpha_candidates[start : start + 32, None]
            beta_chunk = beta_candidates[start : start + 32, None]
            residual = (alpha_chunk * x[None] + beta_chunk - y[None]).abs()
            inliers = residual <= threshold
            scores = inliers.sum(dim=1)
            score, local_index = scores.max(dim=0)
            if int(score) > best_score:
                best_score = int(score)
                best_inliers = inliers[int(local_index)]
        if best_inliers is None or best_inliers.sum() < 2:
            alpha, beta = self._linear_fit(x, y)
            return alpha.detach(), beta.detach(), False
        alpha, beta = self._linear_fit(x[best_inliers], y[best_inliers])
        return alpha.detach(), beta.detach(), int(best_inliers.sum()) >= self.min_points

    @staticmethod
    def _fill_small_holes(
        depth: torch.Tensor, valid: torch.Tensor, iterations: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_float = valid.to(depth.dtype)
        for _ in range(iterations):
            count = F.avg_pool2d(valid_float, 3, stride=1, padding=1) * 9.0
            total = F.avg_pool2d(depth * valid_float, 3, stride=1, padding=1) * 9.0
            can_fill = (~valid) & (count >= 2.0)
            proposal = total / count.clamp_min(1.0)
            depth = torch.where(can_fill, proposal, depth)
            valid = valid | can_fill
            valid_float = valid.to(depth.dtype)
        return depth, valid

    def forward(
        self,
        initial_disparity: torch.Tensor,
        monocular_depth_texture: torch.Tensor,
        calibration: Mapping[str, torch.Tensor],
        stereo_image_hw: Tuple[int, int],
    ) -> GAMResult:
        if initial_disparity.ndim != 4 or initial_disparity.shape[1] != 1:
            raise ValueError("initial_disparity must be [B, 1, H, W]")
        if monocular_depth_texture.ndim == 3:
            monocular_depth_texture = monocular_depth_texture.unsqueeze(1)
        if monocular_depth_texture.ndim != 4 or monocular_depth_texture.shape[1] != 1:
            raise ValueError("monocular_depth_texture must be [B, 1, H, W]")
        if monocular_depth_texture.shape[0] != initial_disparity.shape[0]:
            raise ValueError("stereo and texture batches must match")

        batch, _, low_h, low_w = initial_disparity.shape
        texture_h, texture_w = monocular_depth_texture.shape[-2:]
        device = initial_disparity.device
        # Projection and RANSAC use fp32 for stable inverses and metric fitting.
        disparity = initial_disparity.float()
        mono_depth = monocular_depth_texture.float()
        K_left = _as_batch_tensor(calibration, "K_left", batch, device, torch.float32)
        K_texture = _as_batch_tensor(calibration, "K_texture", batch, device, torch.float32)
        transform = _as_batch_tensor(
            calibration, "T_left_to_texture", batch, device, torch.float32
        )
        baseline = _as_batch_tensor(
            calibration, "baseline", batch, device, torch.float32
        ).reshape(batch, 1, 1, 1)
        offset = _as_batch_tensor(
            calibration, "disparity_offset", batch, device, torch.float32, default=0.0
        ).reshape(batch, 1, 1, 1)
        sign = _as_batch_tensor(
            calibration, "disparity_sign", batch, device, torch.float32, default=1.0
        ).reshape(batch, 1, 1, 1)

        K_left_low = self._scale_intrinsics_batch(
            K_left, stereo_image_hw, (low_h, low_w)
        )
        offset_low = offset * (float(low_w) / float(stereo_image_hw[1]))
        effective_disparity = sign * (disparity - offset_low)
        valid_disparity = torch.isfinite(effective_disparity) & (
            effective_disparity > self.min_abs_disparity
        )
        focal_x = K_left_low[:, 0, 0].reshape(batch, 1, 1, 1)
        stereo_depth_left = focal_x * baseline / effective_disparity.clamp_min(
            self.min_abs_disparity
        )
        valid_disparity &= (
            (stereo_depth_left > self.min_depth) & (stereo_depth_left < self.max_depth)
        )
        stereo_depth_left = torch.where(
            valid_disparity, stereo_depth_left, torch.zeros_like(stereo_depth_left)
        )

        stereo_depth_texture, stereo_valid_texture = self._project_depth(
            stereo_depth_left,
            K_left_low,
            transform,
            K_texture,
            (texture_h, texture_w),
        )

        alpha_values = []
        beta_values = []
        fit_valid_values = []
        for batch_index in range(batch):
            valid = stereo_valid_texture[batch_index, 0] & torch.isfinite(
                mono_depth[batch_index, 0]
            )
            alpha, beta, fit_valid = self._ransac_fit(
                mono_depth[batch_index, 0][valid],
                stereo_depth_texture[batch_index, 0][valid],
            )
            alpha_values.append(alpha)
            beta_values.append(beta)
            fit_valid_values.append(fit_valid)
        alpha = torch.stack(alpha_values).reshape(batch, 1, 1, 1).detach()
        beta = torch.stack(beta_values).reshape(batch, 1, 1, 1).detach()
        fit_valid = torch.tensor(fit_valid_values, device=device, dtype=torch.bool).reshape(
            batch, 1, 1, 1
        )

        aligned_texture = alpha * mono_depth + beta
        valid_texture = (
            fit_valid
            & torch.isfinite(aligned_texture)
            & (aligned_texture > self.min_depth)
            & (aligned_texture < self.max_depth)
        )
        aligned_texture = torch.where(
            valid_texture, aligned_texture, torch.zeros_like(aligned_texture)
        )

        aligned_left, valid_left = self._project_depth(
            aligned_texture,
            K_texture,
            torch.linalg.inv(transform),
            K_left_low,
            (low_h, low_w),
        )
        valid_left &= fit_valid
        aligned_left = torch.where(valid_left, aligned_left, torch.zeros_like(aligned_left))
        if self.hole_fill_iterations > 0:
            aligned_left, valid_left = self._fill_small_holes(
                aligned_left, valid_left, self.hole_fill_iterations
            )

        return GAMResult(
            aligned_depth_left=aligned_left.to(initial_disparity.dtype),
            valid_left=valid_left,
            aligned_depth_texture=aligned_texture.to(initial_disparity.dtype),
            valid_texture=valid_texture,
            stereo_depth_texture=stereo_depth_texture.to(initial_disparity.dtype),
            alpha=alpha.reshape(batch).to(initial_disparity.dtype),
            beta=beta.reshape(batch).to(initial_disparity.dtype),
        )
