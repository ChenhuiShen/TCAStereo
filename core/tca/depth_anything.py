"""Frozen Depth Anything V2 adapter for online monocular priors."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


class DepthAnythingV2Prior(nn.Module):
    """Run the official Depth Anything V2 model on a batched RGB tensor.

    The checkpoint is frozen by design, as described in the TCAStereo paper.
    A model object may be injected for tests; production normally supplies the
    official repository and checkpoint paths.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        encoder: str = "vitl",
        repository_path: Optional[str] = None,
        input_size: int = 518,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"encoder must be one of {sorted(MODEL_CONFIGS)}, got {encoder}")
        self.encoder = encoder
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")

        if model is None:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required for Depth Anything V2")
            if repository_path is not None:
                repository = Path(repository_path).expanduser().resolve()
                if not (repository / "depth_anything_v2" / "dpt.py").is_file():
                    raise FileNotFoundError(
                        f"official Depth Anything V2 source was not found under {repository}"
                    )
                repository_string = str(repository)
                if repository_string not in sys.path:
                    sys.path.insert(0, repository_string)
            try:
                module = importlib.import_module("depth_anything_v2.dpt")
                model_class = getattr(module, "DepthAnythingV2")
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "Depth Anything V2 is unavailable. Clone the official repository "
                    "and pass --depth-anything-repo, or install its Python package."
                ) from exc
            model = model_class(**MODEL_CONFIGS[encoder])
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=True)

        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "DepthAnythingV2Prior":
        super().train(mode)
        self.model.eval()
        return self

    def _resize_shape(self, height: int, width: int) -> tuple[int, int]:
        # Mirrors the official lower-bound resize: the shorter side reaches the
        # configured input size, while both dimensions remain DINOv2/14 aligned.
        scale = max(self.input_size / float(height), self.input_size / float(width))
        new_height = max(14, int(round(height * scale / 14.0)) * 14)
        new_width = max(14, int(round(width * scale / 14.0)) * 14)
        return new_height, new_width

    def forward(self, texture_image: torch.Tensor) -> torch.Tensor:
        if texture_image.ndim != 4 or texture_image.shape[1] != 3:
            raise ValueError("texture_image must be [B, 3, H, W]")
        original_height, original_width = texture_image.shape[-2:]
        image = texture_image.float()
        if image.detach().amax() > 1.5:
            image = image / 255.0
        target_shape = self._resize_shape(original_height, original_width)
        image = F.interpolate(image, size=target_shape, mode="bicubic", align_corners=False)
        image = (image - self.image_mean) / self.image_std
        with torch.no_grad():
            depth = self.model(image)
            if isinstance(depth, (tuple, list)):
                depth = depth[0]
            if isinstance(depth, dict):
                depth = depth.get("predicted_depth", depth.get("depth"))
            if depth is None:
                raise RuntimeError("Depth Anything V2 returned no depth tensor")
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            if depth.ndim != 4 or depth.shape[1] != 1:
                raise RuntimeError(f"unexpected Depth Anything output shape: {tuple(depth.shape)}")
            depth = F.interpolate(
                depth.float(),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            )
        return depth
