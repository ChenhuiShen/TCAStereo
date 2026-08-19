"""Calibration and external rigid-transform loading for the third camera.

The internal convention is explicit and fixed:

* pixel coordinates are ``(u, v)``;
* camera points use ``X_texture = T_left_to_texture @ X_left``;
* image sizes are stored as ``(width, height)``;
* lengths are converted to metres while loading.

Keeping the convention in one module avoids the most common tri-camera failure:
silently using a texture-to-left transform as a left-to-texture transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


_UNIT_TO_METRES = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "millimetre": 1e-3,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "centimetre": 1e-2,
}


def _as_size(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"image size must contain [width, height], got {value}")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {value}")
    return width, height


def _nested_get(mapping: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        value: Any = mapping
        found = True
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                found = False
                break
            value = value[key]
        if found:
            return value
    return None


def _load_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        return value

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML calibration files") from exc
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must contain a YAML mapping")
        return value

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    raise ValueError(f"unsupported mapping format: {path.suffix}")


def _matrix_from_mapping(mapping: Mapping[str, Any]) -> Optional[np.ndarray]:
    matrix = _nested_get(
        mapping,
        "T_left_to_texture",
        "T_stereo_to_texture",
        "T",
        "extrinsics.T",
        "rt.T",
    )
    if matrix is not None:
        array = np.asarray(matrix, dtype=np.float64)
        if array.shape == (3, 4):
            array = np.vstack([array, np.array([[0.0, 0.0, 0.0, 1.0]])])
        if array.shape != (4, 4):
            raise ValueError(f"rigid transform must be 3x4 or 4x4, got {array.shape}")
        return array

    rotation = _nested_get(mapping, "R", "rotation", "extrinsics.R", "rt.R")
    translation = _nested_get(mapping, "t", "translation", "extrinsics.t", "rt.t")
    if rotation is None or translation is None:
        return None
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(
            f"R/t must have shapes (3, 3)/(3,), got {rotation.shape}/{translation.shape}"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _load_transform(path: Path) -> Tuple[np.ndarray, Mapping[str, Any]]:
    suffix = path.suffix.lower()
    metadata: Mapping[str, Any] = {}
    if suffix in {".json", ".yaml", ".yml", ".npz"}:
        metadata = _load_mapping(path)
        transform = _matrix_from_mapping(metadata)
        if transform is None:
            raise ValueError(f"no T or R/t was found in {path}")
        return transform, metadata
    if suffix == ".npy":
        transform = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    else:
        transform = np.asarray(np.loadtxt(path), dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, np.array([[0.0, 0.0, 0.0, 1.0]])])
    if transform.shape != (4, 4):
        raise ValueError(f"RT file must contain a 3x4 or 4x4 matrix, got {transform.shape}")
    return transform, metadata


def _length_scale(mapping: Mapping[str, Any], fallback: str = "m") -> float:
    unit = str(
        _nested_get(mapping, "length_unit", "translation_unit", "units.length") or fallback
    ).lower()
    if unit not in _UNIT_TO_METRES:
        raise ValueError(f"unsupported calibration length unit: {unit}")
    return _UNIT_TO_METRES[unit]


def _validate_intrinsics(name: str, matrix: np.ndarray) -> None:
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {matrix.shape}")
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{name} contains invalid focal lengths or non-finite values")


def _validate_transform(transform: np.ndarray) -> None:
    if not np.isfinite(transform).all():
        raise ValueError("RT contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("the last RT row must be [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5e-3):
        raise ValueError("the RT rotation block is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=5e-3):
        raise ValueError("the RT rotation determinant must be +1")


def scale_intrinsics(
    intrinsics: torch.Tensor,
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Scale a 3x3 intrinsic matrix between image resolutions.

    ``source_size`` and ``target_size`` use ``(width, height)`` order.
    """

    source_width, source_height = source_size
    target_width, target_height = target_size
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    scaled = intrinsics.clone()
    scaled[..., 0, :] *= scale_x
    scaled[..., 1, :] *= scale_y
    scaled[..., 2, :] = intrinsics[..., 2, :]
    return scaled


@dataclass(frozen=True)
class CameraCalibration:
    """Calibrated left-stereo and texture-camera geometry."""

    K_left: torch.Tensor
    K_texture: torch.Tensor
    T_left_to_texture: torch.Tensor
    baseline: float
    left_image_size: Optional[Tuple[int, int]] = None
    texture_image_size: Optional[Tuple[int, int]] = None
    disparity_offset: float = 0.0
    disparity_sign: float = 1.0

    def runtime_dict(
        self,
        left_image_size: Tuple[int, int],
        texture_image_size: Tuple[int, int],
        left_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        texture_padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """Create batched tensors whose intrinsics match the runtime images.

        Padding is ``(left, right, top, bottom)``. The supplied image sizes refer
        to the unpadded images.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        left_source = self.left_image_size or left_image_size
        texture_source = self.texture_image_size or texture_image_size
        K_left = scale_intrinsics(self.K_left, left_source, left_image_size)
        K_texture = scale_intrinsics(self.K_texture, texture_source, texture_image_size)
        K_left = K_left.clone()
        K_texture = K_texture.clone()
        K_left[0, 2] += float(left_padding[0])
        K_left[1, 2] += float(left_padding[2])
        K_texture[0, 2] += float(texture_padding[0])
        K_texture[1, 2] += float(texture_padding[2])

        def batched(value: torch.Tensor) -> torch.Tensor:
            value = value.to(device=device, dtype=dtype)
            return value.unsqueeze(0).expand(batch_size, *value.shape).contiguous()

        return {
            "K_left": batched(K_left),
            "K_texture": batched(K_texture),
            "T_left_to_texture": batched(self.T_left_to_texture),
            "baseline": torch.full(
                (batch_size,), self.baseline, device=device, dtype=dtype
            ),
            "disparity_offset": torch.full(
                (batch_size,), self.disparity_offset, device=device, dtype=dtype
            ),
            "disparity_sign": torch.full(
                (batch_size,), self.disparity_sign, device=device, dtype=dtype
            ),
        }


def load_calibration(
    calibration_path: str,
    rt_path: Optional[str] = None,
    rt_direction: Optional[str] = None,
) -> CameraCalibration:
    """Load intrinsics and an optional external RT file.

    Supported calibration formats are JSON/YAML/NPZ. The external RT may also
    be a NPY or whitespace-delimited 3x4/4x4 TXT file. When RT direction is
    ``texture_to_left`` the matrix is inverted on load.
    """

    calibration_file = Path(calibration_path).expanduser().resolve()
    if not calibration_file.is_file():
        raise FileNotFoundError(calibration_file)
    mapping = _load_mapping(calibration_file)

    K_left_value = _nested_get(
        mapping, "K_left", "K_stereo", "left.K", "cameras.left.K"
    )
    K_texture_value = _nested_get(
        mapping,
        "K_texture",
        "K_rgb",
        "texture.K",
        "rgb.K",
        "cameras.texture.K",
    )
    if K_left_value is None or K_texture_value is None:
        raise ValueError("calibration must define K_left and K_texture")
    K_left = np.asarray(K_left_value, dtype=np.float64)
    K_texture = np.asarray(K_texture_value, dtype=np.float64)
    _validate_intrinsics("K_left", K_left)
    _validate_intrinsics("K_texture", K_texture)

    global_scale = _length_scale(mapping)
    baseline_value = _nested_get(mapping, "baseline", "stereo.baseline")
    if baseline_value is None:
        raise ValueError("calibration must define the stereo baseline")
    baseline = float(np.asarray(baseline_value).reshape(-1)[0]) * global_scale
    if baseline <= 0:
        raise ValueError("stereo baseline must be positive")

    rt_metadata: Mapping[str, Any] = mapping
    if rt_path is not None:
        rt_file = Path(rt_path).expanduser().resolve()
        if not rt_file.is_file():
            raise FileNotFoundError(rt_file)
        transform, rt_metadata = _load_transform(rt_file)
    else:
        transform = _matrix_from_mapping(mapping)
        if transform is None:
            raise ValueError("no RT found; provide rt_path or T/R/t in calibration")

    rt_scale = _length_scale(rt_metadata, fallback=str(_nested_get(mapping, "length_unit") or "m"))
    transform = transform.copy()
    transform[:3, 3] *= rt_scale
    direction = str(
        rt_direction
        or _nested_get(rt_metadata, "direction", "rt_direction")
        or _nested_get(mapping, "rt_direction")
        or "left_to_texture"
    ).lower()
    if direction in {"texture_to_left", "rgb_to_left", "t_to_s"}:
        transform = np.linalg.inv(transform)
    elif direction not in {"left_to_texture", "left_to_rgb", "s_to_t"}:
        raise ValueError(f"unsupported RT direction: {direction}")
    _validate_transform(transform)

    left_size = _as_size(
        _nested_get(mapping, "left_image_size", "left.image_size", "cameras.left.image_size")
    )
    texture_size = _as_size(
        _nested_get(
            mapping,
            "texture_image_size",
            "rgb_image_size",
            "texture.image_size",
            "cameras.texture.image_size",
        )
    )
    disparity_offset = float(
        _nested_get(mapping, "disparity_offset", "stereo.disparity_offset") or 0.0
    )
    disparity_sign = float(
        _nested_get(mapping, "disparity_sign", "stereo.disparity_sign") or 1.0
    )
    if disparity_sign not in {-1.0, 1.0}:
        raise ValueError("disparity_sign must be +1 or -1")

    return CameraCalibration(
        K_left=torch.as_tensor(K_left, dtype=torch.float32),
        K_texture=torch.as_tensor(K_texture, dtype=torch.float32),
        T_left_to_texture=torch.as_tensor(transform, dtype=torch.float32),
        baseline=baseline,
        left_image_size=left_size,
        texture_image_size=texture_size,
        disparity_offset=disparity_offset,
        disparity_sign=disparity_sign,
    )
