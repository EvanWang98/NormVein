#!/usr/bin/env python3
"""Train a NormVein ROI detector on FingerVeinSyn-5M."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from normvein.data import FVSynRotatedCocoDataset, collate_fn
from normvein.models import build_model


MODELS = (
    "roi_rotated_faster_rcnn_r50_fpn",
    "roi_rotated_ssd_r50_fpn",
    "roi_rotated_yolo_r50_fpn",
)


def save_model(path: Path, model_id: str, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_id": model_id,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": {"dataset": "FingerVeinSyn-5M"},
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--ann-file", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/detection"))
    args = parser.parse_args()

    torch.manual_seed(2048)
    device = torch.device(args.device)
    dataset = FVSynRotatedCocoDataset(
        ann_file=args.ann_file,
        image_root=args.image_root,
        image_size=(600, 300),
        train=True,
        root_dir=Path.cwd(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    model = build_model(args.model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    output = args.output_dir / f"normvein_{args.model}.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        scheduler.step()
        print(f"epoch={epoch}/{args.epochs} loss={total_loss / len(loader):.6f}")
        save_model(output, args.model, model)


if __name__ == "__main__":
    main()
