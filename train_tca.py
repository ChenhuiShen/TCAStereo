"""Training entry point for the complete TCAStereo tri-camera model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.TCAStereo import TCAStereo
from core.tri_camera_dataset import TriCameraDataset


def sequence_loss(
    disparity_predictions: Iterable[torch.Tensor],
    initial_prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = 0.9,
    max_disparity: float = 256.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    predictions = list(disparity_predictions)
    if not predictions:
        raise ValueError("at least one iterative prediction is required")
    valid = valid.unsqueeze(1) if valid.ndim == 3 else valid
    valid = valid.bool() & torch.isfinite(ground_truth) & (ground_truth.abs() < max_disparity)
    if not valid.any():
        raise ValueError("the batch contains no valid disparity pixels")

    loss = F.smooth_l1_loss(initial_prediction[valid], ground_truth[valid])
    denominator = max(len(predictions) - 1, 1)
    adjusted_gamma = gamma ** (15.0 / denominator)
    for index, prediction in enumerate(predictions):
        weight = adjusted_gamma ** (len(predictions) - index - 1)
        loss = loss + weight * (prediction - ground_truth).abs()[valid].mean()

    error = (predictions[-1] - ground_truth).abs()[valid]
    metrics = {
        "epe": float(error.mean().detach()),
        "bad1": float((error > 1.0).float().mean().detach()),
        "bad3": float((error > 3.0).float().mean().detach()),
    }
    return loss, metrics


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    checkpoint = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    logging.info("checkpoint loaded: %d missing, %d unexpected keys", len(missing), len(unexpected))


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dataset = TriCameraDataset(
        args.manifest,
        calibration_path=args.calibration,
        rt_path=args.rt,
        rt_direction=args.rt_direction,
        crop_size=args.image_size,
        texture_size=args.texture_size,
        random_crop=True,
    )
    if args.depth_anything_checkpoint is None and not all(
        sample.get("mono_depth") for sample in dataset.samples
    ):
        raise ValueError(
            "provide --depth-anything-checkpoint or mono_depth for every manifest sample"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size,
    )

    model = TCAStereo(args).to(device)
    if args.restore_checkpoint:
        load_checkpoint(model, args.restore_checkpoint)
    model.train()
    model.freeze_bn()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - step / float(args.num_steps)) ** 0.9,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.mixed_precision and device.type == "cuda"
    )
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.num_steps:
        for batch in loader:
            if step >= args.num_steps:
                break
            left = batch["left"].to(device, non_blocking=True)
            right = batch["right"].to(device, non_blocking=True)
            texture = batch["texture"].to(device, non_blocking=True)
            disparity = batch["disparity"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            calibration = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["calibration"].items()
            }
            mono_depth = batch.get("mono_depth")
            if mono_depth is not None:
                mono_depth = mono_depth.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=args.mixed_precision and device.type == "cuda"
            ):
                initial, predictions = model(
                    left,
                    right,
                    texture_image=texture,
                    mono_depth=mono_depth,
                    calibration=calibration,
                    iters=args.train_iters,
                    test_mode=False,
                )
                loss, metrics = sequence_loss(
                    predictions,
                    initial,
                    disparity,
                    valid,
                    gamma=args.loss_gamma,
                    max_disparity=args.max_disp,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step == 1 or step % args.log_frequency == 0:
                logging.info(
                    "step=%d loss=%.5f epe=%.4f bad1=%.4f lr=%.3e",
                    step,
                    float(loss.detach()),
                    metrics["epe"],
                    metrics["bad1"],
                    scheduler.get_last_lr()[0],
                )
            if step % args.save_frequency == 0 or step == args.num_steps:
                destination = output_directory / f"tca_step_{step:07d}.pth"
                torch.save(
                    {"model": model.state_dict(), "step": step, "args": vars(args)},
                    destination,
                )
                logging.info("saved %s", destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="tri-camera JSONL manifest")
    parser.add_argument("--calibration", help="global intrinsics JSON/YAML (optional per sample)")
    parser.add_argument("--rt", help="external left-to-texture RT (optional per sample)")
    parser.add_argument(
        "--rt-direction",
        default="left_to_texture",
        choices=["left_to_texture", "texture_to_left"],
    )
    parser.add_argument("--depth-anything-repo", default="third_party/Depth-Anything-V2")
    parser.add_argument("--depth-anything-checkpoint")
    parser.add_argument(
        "--depth-anything-encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
    )
    parser.add_argument("--depth-anything-input-size", type=int, default=518)
    parser.add_argument("--restore-checkpoint")
    parser.add_argument("--output-directory", default="checkpoints/tca")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=400000)
    parser.add_argument("--train-iters", type=int, default=22)
    parser.add_argument("--image-size", nargs=2, type=int, default=[256, 512])
    parser.add_argument("--texture-size", nargs=2, type=int)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-gamma", type=float, default=0.9)
    parser.add_argument("--clip-gradient", type=float, default=1.0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--log-frequency", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=2000)

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
    arguments = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    random.seed(666)
    np.random.seed(666)
    torch.manual_seed(666)
    train(arguments)
