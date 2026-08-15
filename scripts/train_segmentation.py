#!/usr/bin/env python3
"""Train either released U-Net/ResNet-50 segmentation task from scratch."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from normvein.models.segmentation import (
    FingerVeinSegmentationDataset,
    UNetBCEDiceLoss,
    build_model,
)


TASKS = {
    "finger-shape": {
        "model_id": "seg_finger_shape_unet_r50",
        "images": "raw_images",
        "masks": "shape_masks",
        "image_size": (300, 600),
        "percentile_normalize": False,
        "percentile_low": 1.0,
        "percentile_high": 99.0,
    },
    "vein-pattern": {
        "model_id": "seg_vein_pattern_unet_r50",
        "images": "roi_images",
        "masks": "pattern_masks",
        "image_size": (100, 300),
        "percentile_normalize": True,
        "percentile_low": 10.0,
        "percentile_high": 90.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-subset-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/segmentation"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = TASKS[args.task]
    device = torch.device(args.device)
    dataset = FingerVeinSegmentationDataset(
        args.data_root / config["images"],
        args.data_root / config["masks"],
        image_size=config["image_size"],
        train=True,
        percentile_normalize=config["percentile_normalize"],
        percentile_low=config["percentile_low"],
        percentile_high=config["percentile_high"],
    )
    subset_count = max(1, round(len(dataset) * args.train_subset_ratio))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_count].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        "unet_resnet50",
        encoder_weights="none",
        use_smp=False,
        allow_weight_download=False,
        in_channels=1,
    ).to(device)
    criterion = UNetBCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        print(f"epoch={epoch}/{args.epochs} loss={total_loss / max(len(loader), 1):.6f}")
        artifact = {
            "format_version": 1,
            "model_id": config["model_id"],
            "task": config["model_id"].replace("seg_", "").replace("_unet_r50", "_segmentation"),
            "architecture": "U-Net (ResNet-50 encoder)",
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": {
                "format_version": 1,
                "model_id": config["model_id"],
                "dataset": "FingerVeinSyn-5M",
                "source_epoch": epoch,
                "input_size": list(config["image_size"]),
                "train_subset_ratio": args.train_subset_ratio,
                "training_args": vars(args),
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.output_dir / f"{config['model_id']}_e{epoch:03d}.pth")


if __name__ == "__main__":
    main()

