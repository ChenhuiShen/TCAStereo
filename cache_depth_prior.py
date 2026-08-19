"""Precompute frozen Depth Anything V2 relative-depth priors as NPY files."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from core.tca.depth_anything import DepthAnythingV2Prior


def load_rgb(path: str, device: torch.device) -> torch.Tensor:
    image = np.asarray(Image.open(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    return (
        torch.from_numpy(image[..., :3].astype(np.uint8).copy())
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
    )


def main(args: argparse.Namespace) -> None:
    images = sorted(glob.glob(args.images, recursive=True))
    if not images and Path(args.images).is_file():
        images = [args.images]
    if not images:
        raise ValueError("the image pattern matched no files")
    device = torch.device(args.device)
    model = DepthAnythingV2Prior(
        checkpoint_path=args.checkpoint,
        encoder=args.encoder,
        repository_path=args.repository,
        input_size=args.input_size,
    ).to(device).eval()
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for path in images:
            depth = model(load_rgb(path, device))[0, 0].float().cpu().numpy()
            destination = output / f"{Path(path).stem}_depth_anything_v2.npy"
            np.save(destination, depth)
            print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="RGB image path or glob")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repository", default="third_party/Depth-Anything-V2")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--output-directory", default="cache/depth_anything_v2")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
