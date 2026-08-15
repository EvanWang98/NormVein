#!/usr/bin/env python3
"""Extract the evaluated model state dictionaries into uniform release files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


RELEASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[3]

SPECS = (
    {
        "model_id": "roi_rotated_faster_rcnn_r50_fpn",
        "task": "roi_detection",
        "architecture": "Rotated Faster R-CNN (ResNet-50-FPN)",
        "source": "ckpt/Det/epoch_6.pth",
        "state_key": "model",
        "output": "detection/normvein_roi_rotated_faster_rcnn_r50_fpn_e006.pth",
        "expected_epoch": 6,
        "input_size": [300, 600],
        "pretraining_samples": 500000,
        "source_train_subset_ratio": 0.1,
    },
    {
        "model_id": "roi_rotated_ssd_r50_fpn",
        "task": "roi_detection",
        "architecture": "Rotated SSD (ResNet-50-FPN)",
        "source": "ckpt/Det/ssd_latest.pth",
        "state_key": "model",
        "output": "detection/normvein_roi_rotated_ssd_r50_fpn_e010.pth",
        "expected_epoch": 10,
        "input_size": [300, 600],
        "pretraining_samples": 500000,
        "source_train_subset_ratio": 0.1,
    },
    {
        "model_id": "roi_rotated_yolo_r50_fpn",
        "task": "roi_detection",
        "architecture": "Rotated YOLO-style detector (ResNet-50-FPN)",
        "source": "ckpt/Det/yolo_latest.pth",
        "state_key": "model",
        "output": "detection/normvein_roi_rotated_yolo_r50_fpn_e010.pth",
        "expected_epoch": 10,
        "input_size": [300, 600],
        "pretraining_samples": 500000,
        "source_train_subset_ratio": 0.1,
    },
    {
        "model_id": "seg_finger_shape_unet_r50",
        "task": "finger_shape_segmentation",
        "architecture": "U-Net (ResNet-50 encoder)",
        "source": "ckpt/Seg/FingerShape.pt",
        "state_key": "model",
        "output": "segmentation/normvein_seg_finger_shape_unet_r50_e010.pth",
        "expected_epoch": 10,
        "input_size": [300, 600],
        "percentile_normalization": None,
        "paper_pretraining_samples": 500000,
        "source_train_subset_ratio": 0.01,
        "pretraining_samples": None,
        "sample_count_status": "unverified; 1% of the published 5M dataset would be 50,000",
    },
    {
        "model_id": "seg_vein_pattern_unet_r50",
        "task": "vein_pattern_segmentation",
        "architecture": "U-Net (ResNet-50 encoder)",
        "source": "ckpt/Seg/FingerPattern.pt",
        "state_key": "model",
        "output": "segmentation/normvein_seg_vein_pattern_unet_r50_e010.pth",
        "expected_epoch": 10,
        "input_size": [100, 300],
        "percentile_normalization": [10, 90],
        "paper_pretraining_samples": 500000,
        "source_train_subset_ratio": 0.01,
        "pretraining_samples": None,
        "sample_count_status": "unverified; 1% of the published 5M dataset would be 50,000",
    },
    {
        "model_id": "rec_mobilefacenet_large",
        "task": "recognition",
        "architecture": "MobileFaceNet-Large",
        "source": "ckpt/Rec/MB/checkpoint_gpu_0MB_500k.pt",
        "state_key": "state_dict_backbone",
        "output": "recognition/normvein_rec_mobilefacenet_large_e050.pth",
        "expected_epoch": 50,
        "input_size": [100, 300],
        "embedding_dim": 512,
        "paper_epoch": 200,
        "pretraining_samples": 500000,
    },
    {
        "model_id": "rec_iresnet100",
        "task": "recognition",
        "architecture": "iResNet-100",
        "source": "ckpt/Rec/RS/checkpoint_gpu_0R100_500k.pt",
        "state_key": "state_dict_backbone",
        "output": "recognition/normvein_rec_iresnet100_e044.pth",
        "expected_epoch": 44,
        "input_size": [100, 300],
        "embedding_dim": 512,
        "paper_epoch": 100,
        "pretraining_samples": 500000,
    },
    {
        "model_id": "rec_convnext_small",
        "task": "recognition",
        "architecture": "ConvNeXt-Small",
        "source": "ckpt/Rec/VIT/checkpoint_gpu_0vit_500k.pt",
        "state_key": "state_dict_backbone",
        "output": "recognition/normvein_rec_convnext_small_e100.pth",
        "expected_epoch": 100,
        "input_size": [224, 224],
        "embedding_dim": 512,
        "paper_epoch": 300,
        "pretraining_samples": 500000,
    },
)


def sha256sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_state_dict(value: Any) -> OrderedDict:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("checkpoint state_dict is missing or empty")
    state_dict = OrderedDict()
    for key, tensor in value.items():
        if not isinstance(key, str) or not torch.is_tensor(tensor):
            raise TypeError("checkpoint state_dict must map string keys to tensors")
        clean_key = key[7:] if key.startswith("module.") else key
        state_dict[clean_key] = tensor.detach().cpu()
    return state_dict


def extract_one(spec: Dict[str, Any], source_root: Path, output_root: Path) -> Dict[str, Any]:
    source_path = source_root / spec["source"]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    checkpoint = torch.load(str(source_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{source_path}: expected a checkpoint mapping")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != int(spec["expected_epoch"]):
        raise ValueError(
            f"{source_path}: expected epoch {spec['expected_epoch']}, found {epoch}"
        )
    state_dict = normalize_state_dict(checkpoint.get(spec["state_key"]))

    metadata = {
        key: value
        for key, value in spec.items()
        if key
        not in {
            "source",
            "state_key",
            "output",
            "expected_epoch",
        }
    }
    metadata.update(
        {
            "format_version": 1,
            "dataset": "FingerVeinSyn-5M",
            "source_epoch": epoch,
            "source_checkpoint": spec["source"],
        }
    )
    artifact = {
        "format_version": 1,
        "model_id": spec["model_id"],
        "task": spec["task"],
        "architecture": spec["architecture"],
        "state_dict": state_dict,
        "metadata": metadata,
    }

    output_path = output_root / spec["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(artifact, str(temporary_path))
    os.replace(str(temporary_path), str(output_path))

    result = {
        **metadata,
        "filename": output_path.name,
        "relative_path": output_path.relative_to(output_root).as_posix(),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256sum(output_path),
        "tensor_count": len(state_dict),
        "parameter_count": sum(int(tensor.numel()) for tensor in state_dict.values()),
    }
    print(
        f"prepared {spec['model_id']}: {result['relative_path']} "
        f"({result['size_bytes'] / 1024**2:.2f} MiB)"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=RELEASE_DIR / "weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    records = [extract_one(spec, source_root, output_root) for spec in SPECS]
    manifest = {
        "format_version": 1,
        "release": "candidate-v1",
        "dataset": "FingerVeinSyn-5M",
        "models": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
