#!/usr/bin/env python3
"""Train a NormVein recognition backbone from scratch with ArcFace loss."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from normvein.models import build_model, model_spec


RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
DEFAULT_EPOCHS = {
    "rec_mobilefacenet_large": 200,
    "rec_iresnet100": 100,
    "rec_convnext_small": 300,
}


class ManifestDataset(Dataset):
    def __init__(self, image_root: Path, manifest: Path, model_id: str):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.samples = payload["samples"]
        self.image_root = image_root
        self.size_hw = tuple(model_spec(model_id)["input_size"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        height, width = self.size_hw
        with Image.open(self.image_root / sample["path"]) as image:
            image = image.convert("L").resize((width, height), RESAMPLE_BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array.copy()).unsqueeze(0), int(sample["label"])


class CombinedMarginClassifier(nn.Module):
    """Full-class additive angular-margin head with (m1, m2, m3)."""

    def __init__(self, features: int, classes: int, margin=(1.0, 0.7, 0.0), scale=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, features))
        nn.init.normal_(self.weight, std=0.01)
        self.m1, self.m2, self.m3 = [float(value) for value in margin]
        self.scale = float(scale)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = cosine.gather(1, labels[:, None])
        target_margin = torch.cos(torch.acos(target) * self.m1 + self.m2) - self.m3
        logits = cosine.scatter(1, labels[:, None], target_margin)
        return logits * self.scale


def save_artifact(path: Path, model_id: str, model: nn.Module, epoch: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": 1,
        "model_id": model_id,
        "task": "recognition",
        "architecture": model_spec(model_id)["architecture"],
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "format_version": 1,
            "model_id": model_id,
            "task": "recognition",
            "dataset": "FingerVeinSyn-5M",
            "source_epoch": epoch,
            "input_size": list(model_spec(model_id)["input_size"]),
            "embedding_dim": 512,
            "training_args": vars(args),
        },
    }
    torch.save(artifact, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Root corresponding to manifest paths")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=sorted(DEFAULT_EPOCHS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/recognition"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    epochs = args.epochs or DEFAULT_EPOCHS[args.model]
    is_convnext = args.model == "rec_convnext_small"
    learning_rate = args.lr if args.lr is not None else (0.001 if is_convnext else 0.01)
    weight_decay = args.weight_decay if args.weight_decay is not None else (0.1 if is_convnext else 5e-4)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    num_classes = int(payload["num_identities"])
    dataset = ManifestDataset(args.data_root, args.manifest, args.model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    backbone = build_model(args.model).to(device)
    classifier = CombinedMarginClassifier(512, num_classes).to(device)
    parameters = list(backbone.parameters()) + list(classifier.parameters())
    if is_convnext:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(
            parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        backbone.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                embeddings = backbone(images)
                logits = classifier(embeddings, labels)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
        scheduler.step()
        mean_loss = total_loss / max(len(loader), 1)
        print(f"epoch={epoch}/{epochs} loss={mean_loss:.6f} lr={scheduler.get_last_lr()[0]:.6g}")
        output = args.output_dir / f"{args.model}_e{epoch:03d}.pth"
        save_artifact(output, args.model, backbone, epoch, args)


if __name__ == "__main__":
    main()
