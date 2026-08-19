"""Manifest-based tri-camera dataset for TCAStereo training."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.tca.calibration import load_calibration
from core.utils.frame_utils import readPFM


def _load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    return image[..., :3].astype(np.uint8)


def _load_disparity(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        disparity = np.load(path, allow_pickle=False)
    elif suffix == ".pfm":
        disparity = readPFM(str(path))
    else:
        disparity = np.asarray(Image.open(path), dtype=np.float32)
    if isinstance(disparity, tuple):
        disparity = disparity[0]
    disparity = np.asarray(disparity, dtype=np.float32)
    if disparity.ndim == 3:
        disparity = disparity[..., 0]
    return disparity


def _resolve(base: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@lru_cache(maxsize=64)
def _cached_calibration(path: str, rt_path: Optional[str], direction: Optional[str]):
    return load_calibration(path, rt_path=rt_path, rt_direction=direction)


class TriCameraDataset(Dataset):
    """Load synchronized left/right IR, texture RGB and disparity samples.

    The JSONL manifest accepts per-sample ``calibration`` and ``rt`` entries or
    global paths supplied to the constructor. A cached ``mono_depth`` NPY/PFM is
    optional; when omitted, Depth Anything V2 runs online in the model.
    """

    def __init__(
        self,
        manifest_path: str,
        calibration_path: Optional[str] = None,
        rt_path: Optional[str] = None,
        rt_direction: Optional[str] = None,
        crop_size: Optional[Sequence[int]] = None,
        texture_size: Optional[Sequence[int]] = None,
        random_crop: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.base = self.manifest_path.parent
        self.global_calibration = _resolve(self.base, calibration_path)
        self.global_rt = _resolve(self.base, rt_path)
        self.rt_direction = rt_direction
        self.crop_size = None if crop_size is None else (int(crop_size[0]), int(crop_size[1]))
        self.texture_size = (
            None if texture_size is None else (int(texture_size[0]), int(texture_size[1]))
        )
        self.random_crop = random_crop
        if self.crop_size is not None and any(value <= 0 for value in self.crop_size):
            raise ValueError("crop_size [height, width] must be positive")
        if self.texture_size is not None and any(value <= 0 for value in self.texture_size):
            raise ValueError("texture_size [height, width] must be positive")

        self.samples: List[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                missing = {"left", "right", "texture", "disparity"} - set(sample)
                if missing:
                    raise ValueError(
                        f"manifest line {line_number} is missing {sorted(missing)}"
                    )
                self.samples.append(sample)
        if not self.samples:
            raise ValueError("tri-camera manifest is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_paths(self, sample: Dict[str, Any]) -> Dict[str, Optional[Path]]:
        calibration = _resolve(self.base, sample.get("calibration")) or self.global_calibration
        rt = _resolve(self.base, sample.get("rt")) or self.global_rt
        if calibration is None:
            raise ValueError("every sample requires a calibration path")
        return {
            "left": _resolve(self.base, sample["left"]),
            "right": _resolve(self.base, sample["right"]),
            "texture": _resolve(self.base, sample["texture"]),
            "disparity": _resolve(self.base, sample["disparity"]),
            "mono_depth": _resolve(self.base, sample.get("mono_depth")),
            "calibration": calibration,
            "rt": rt,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        paths = self._sample_paths(sample)
        left = _load_rgb(paths["left"])
        right = _load_rgb(paths["right"])
        texture = _load_rgb(paths["texture"])
        disparity = _load_disparity(paths["disparity"])
        if left.shape[:2] != right.shape[:2] or left.shape[:2] != disparity.shape[:2]:
            raise ValueError(f"left/right/disparity shapes differ for sample {index}")

        full_height, full_width = left.shape[:2]
        crop_y = crop_x = 0
        if self.crop_size is not None:
            crop_height, crop_width = self.crop_size
            if crop_height > full_height or crop_width > full_width:
                raise ValueError(
                    f"crop {self.crop_size} exceeds image {(full_height, full_width)}"
                )
            if self.random_crop:
                crop_y = random.randint(0, full_height - crop_height)
                crop_x = random.randint(0, full_width - crop_width)
            else:
                crop_y = (full_height - crop_height) // 2
                crop_x = (full_width - crop_width) // 2
            left = left[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
            right = right[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
            disparity = disparity[
                crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
            ]

        texture_height, texture_width = texture.shape[:2]
        target_texture_hw = self.texture_size or (texture_height, texture_width)
        if target_texture_hw != (texture_height, texture_width):
            texture = np.asarray(
                Image.fromarray(texture).resize(
                    (target_texture_hw[1], target_texture_hw[0]), Image.Resampling.BILINEAR
                )
            )

        direction = sample.get("rt_direction", self.rt_direction)
        calibration = _cached_calibration(
            str(paths["calibration"]),
            None if paths["rt"] is None else str(paths["rt"]),
            direction,
        )
        runtime = calibration.runtime_dict(
            left_image_size=(full_width, full_height),
            texture_image_size=(target_texture_hw[1], target_texture_hw[0]),
        )
        runtime = {key: value.squeeze(0) for key, value in runtime.items()}
        runtime["K_left"] = runtime["K_left"].clone()
        runtime["K_left"][0, 2] -= float(crop_x)
        runtime["K_left"][1, 2] -= float(crop_y)

        output: Dict[str, Any] = {
            "left": torch.from_numpy(left.copy()).permute(2, 0, 1).float(),
            "right": torch.from_numpy(right.copy()).permute(2, 0, 1).float(),
            "texture": torch.from_numpy(texture.copy()).permute(2, 0, 1).float(),
            "disparity": torch.from_numpy(disparity.copy()).unsqueeze(0).float(),
            "valid": torch.from_numpy(
                np.isfinite(disparity) & (np.abs(disparity) > 0)
            ).float(),
            "calibration": runtime,
            "paths": {key: str(value) for key, value in paths.items() if value is not None},
        }
        if paths["mono_depth"] is not None:
            mono = _load_disparity(paths["mono_depth"])
            mono_tensor = torch.from_numpy(mono).unsqueeze(0).unsqueeze(0).float()
            if mono.shape != target_texture_hw:
                mono_tensor = F.interpolate(
                    mono_tensor, size=target_texture_hw, mode="bilinear", align_corners=False
                )
            output["mono_depth"] = mono_tensor.squeeze(0)
        return output
