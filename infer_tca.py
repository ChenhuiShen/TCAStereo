"""Calibrated three-camera inference for TCAStereo."""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
import time
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from core.TCAStereo import TCAStereo
from core.tca.calibration import load_calibration
from core.utils.frame_utils import readPFM, savePFM
from core.utils.utils import InputPadder


def expand_images(pattern: Optional[str]) -> List[str]:
    if pattern is None:
        return []
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths and Path(pattern).is_file():
        paths = [str(Path(pattern))]
    return paths


def load_image(path: str, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image[..., :3].astype(np.uint8)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device)


def load_mono_depth(path: str, device: torch.device) -> torch.Tensor:
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".pfm":
        value = readPFM(path)
        if isinstance(value, tuple):
            value = value[0]
    else:
        value = np.asarray(Image.open(path), dtype=np.float32)
    value = np.asarray(value, dtype=np.float32).squeeze()
    return torch.from_numpy(value.copy()).unsqueeze(0).unsqueeze(0).to(device)


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    state = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    logging.info("checkpoint: %d missing and %d unexpected keys", len(missing), len(unexpected))
    if any(key.startswith(("gem.", "gam.", "fiu.")) for key in missing):
        logging.warning("GEM/GAM/FIU contain untrained parameters in this checkpoint")


def save_visualization(path: Path, value: np.ndarray, valid: Optional[np.ndarray] = None) -> None:
    mask = np.isfinite(value)
    if valid is not None:
        mask &= valid
    if not mask.any():
        image = np.zeros(value.shape, dtype=np.uint16)
    else:
        low, high = np.percentile(value[mask], [1.0, 99.0])
        normalized = np.clip((value - low) / max(high - low, 1e-6), 0.0, 1.0)
        image = (normalized * 65535.0).astype(np.uint16)
        image[~mask] = 0
    Image.fromarray(image).save(path)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    left_paths = expand_images(args.left)
    right_paths = expand_images(args.right)
    texture_paths = expand_images(args.texture)
    mono_paths = expand_images(args.mono_depth)
    if not left_paths or not (len(left_paths) == len(right_paths) == len(texture_paths)):
        raise ValueError("left/right/texture patterns must resolve to equal non-zero counts")
    if mono_paths and len(mono_paths) != len(left_paths):
        raise ValueError("cached mono-depth count must match image count")
    if not mono_paths and not args.depth_anything_checkpoint:
        raise ValueError("provide Depth Anything V2 checkpoint or --mono-depth")

    calibration = load_calibration(args.calibration, args.rt, args.rt_direction)
    model = TCAStereo(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    elapsed = []
    with torch.inference_mode():
        for index, (left_path, right_path, texture_path) in enumerate(
            zip(left_paths, right_paths, texture_paths)
        ):
            left = load_image(left_path, device)
            right = load_image(right_path, device)
            texture = load_image(texture_path, device)
            if left.shape != right.shape:
                raise ValueError(f"left/right shapes differ: {left_path}, {right_path}")
            original_h, original_w = left.shape[-2:]
            texture_h, texture_w = texture.shape[-2:]
            padder = InputPadder(left.shape, divis_by=32)
            left_padded, right_padded = padder.pad(left, right)
            runtime_calibration = calibration.runtime_dict(
                left_image_size=(original_w, original_h),
                texture_image_size=(texture_w, texture_h),
                left_padding=tuple(padder._pad),
                batch_size=1,
                device=device,
            )
            mono_depth = load_mono_depth(mono_paths[index], device) if mono_paths else None
            if mono_depth is not None and mono_depth.shape[-2:] != texture.shape[-2:]:
                mono_depth = F.interpolate(
                    mono_depth,
                    size=texture.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = model(
                left_padded,
                right_padded,
                texture_image=texture,
                mono_depth=mono_depth,
                calibration=runtime_calibration,
                iters=args.valid_iters,
                test_mode=True,
                return_details=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - start)

            disparity = padder.unpad(result["disparity"]).squeeze().float().cpu().numpy()
            stem = Path(left_path).stem
            np.save(output_directory / f"{stem}_disparity.npy", disparity)
            savePFM(str(output_directory / f"{stem}_disparity.pfm"), disparity.astype(np.float32))
            save_visualization(output_directory / f"{stem}_disparity.png", disparity)

            aligned_low = result["aligned_depth_left"].float()
            aligned_full = F.interpolate(
                aligned_low, size=left_padded.shape[-2:], mode="bilinear", align_corners=False
            )
            aligned_full = padder.unpad(aligned_full).squeeze().cpu().numpy()
            valid_low = result["gam_valid_left"].float()
            valid_full = F.interpolate(valid_low, size=left_padded.shape[-2:], mode="nearest")
            valid_full = padder.unpad(valid_full).squeeze().bool().cpu().numpy()
            np.save(output_directory / f"{stem}_aligned_depth.npy", aligned_full)
            save_visualization(
                output_directory / f"{stem}_aligned_depth.png", aligned_full, valid_full
            )
            diagnostics = {
                "left": left_path,
                "right": right_path,
                "texture": texture_path,
                "gam_alpha": float(result["gam_alpha"][0].cpu()),
                "gam_beta": float(result["gam_beta"][0].cpu()),
                "gam_left_coverage": float(valid_full.mean()),
                "inference_seconds": elapsed[-1],
            }
            with (output_directory / f"{stem}_alignment.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
            logging.info(
                "%s: %.3fs alpha=%.6g beta=%.6g coverage=%.1f%%",
                stem,
                elapsed[-1],
                diagnostics["gam_alpha"],
                diagnostics["gam_beta"],
                diagnostics["gam_left_coverage"] * 100.0,
            )
    logging.info("processed %d samples; mean %.3fs", len(elapsed), float(np.mean(elapsed)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="left IR path or glob")
    parser.add_argument("--right", required=True, help="right IR path or glob")
    parser.add_argument("--texture", required=True, help="third-camera RGB path or glob")
    parser.add_argument("--mono-depth", help="optional cached Depth Anything NPY/PFM glob")
    parser.add_argument("--calibration", required=True, help="intrinsics/baseline JSON or YAML")
    parser.add_argument("--rt", required=True, help="external RT JSON/YAML/NPY/TXT")
    parser.add_argument(
        "--rt-direction",
        default="left_to_texture",
        choices=["left_to_texture", "texture_to_left"],
    )
    parser.add_argument("--checkpoint", required=True, help="trained TCAStereo checkpoint")
    parser.add_argument("--output-directory", default="out/tca")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--mixed-precision", action="store_true")

    parser.add_argument("--depth-anything-repo", default="third_party/Depth-Anything-V2")
    parser.add_argument("--depth-anything-checkpoint")
    parser.add_argument(
        "--depth-anything-encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
    )
    parser.add_argument("--depth-anything-input-size", type=int, default=518)
    parser.add_argument("--use-gem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gam", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fiu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gam-ransac-iterations", type=int, default=128)
    parser.add_argument("--gam-ransac-threshold", type=float, default=0.05)
    parser.add_argument("--gam-min-points", type=int, default=32)
    parser.add_argument("--fiu-hidden-dim", type=int, default=32)

    parser.add_argument("--shining", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden-dims", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--corr-levels", type=int, default=2)
    parser.add_argument("--corr-radius", type=int, default=4)
    parser.add_argument("--n-downsample", type=int, default=2)
    parser.add_argument("--slow-fast-gru", action="store_true")
    parser.add_argument("--n-gru-layers", type=int, default=3)
    parser.add_argument("--max-disp", type=int, default=256)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_parser().parse_args())
