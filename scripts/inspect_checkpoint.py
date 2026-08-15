#!/usr/bin/env python3
"""Inspect trusted PyTorch checkpoints before preparing a model release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch


def sha256sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    )


def find_state_dict(checkpoint: Any) -> Tuple[Optional[str], Optional[Mapping[str, torch.Tensor]]]:
    if is_tensor_mapping(checkpoint):
        return "root", checkpoint
    if not isinstance(checkpoint, Mapping):
        return None, None
    for key in ("state_dict_backbone", "model", "state_dict"):
        value = checkpoint.get(key)
        if is_tensor_mapping(value):
            return key, value
    return None, None


def tensor_summary(state_dict: Optional[Mapping[str, torch.Tensor]]) -> Dict[str, Any]:
    if state_dict is None:
        return {
            "tensor_count": 0,
            "parameter_count": 0,
            "first_keys": [],
            "last_keys": [],
        }
    keys = list(state_dict.keys())
    return {
        "tensor_count": len(keys),
        "parameter_count": sum(int(tensor.numel()) for tensor in state_dict.values()),
        "first_keys": keys[:5],
        "last_keys": keys[-5:],
    }


def checkpoint_args(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}
    value = checkpoint.get("args")
    if not isinstance(value, Mapping):
        return {}
    allowlist = {
        "model",
        "epochs",
        "image_width",
        "image_height",
        "image_size",
        "subset_ratio",
        "train_subset_ratio",
        "checkpoint_name",
    }
    return {str(key): value[key] for key in sorted(value) if key in allowlist}


def inspect(path: Path, include_hash: bool) -> Dict[str, Any]:
    checkpoint = torch.load(str(path), map_location="cpu")
    state_key, state_dict = find_state_dict(checkpoint)
    result: Dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "top_level_type": type(checkpoint).__name__,
        "top_level_keys": list(checkpoint.keys()) if isinstance(checkpoint, Mapping) else [],
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
        "state_dict_key": state_key,
        "args": checkpoint_args(checkpoint),
        **tensor_summary(state_dict),
    }
    if include_hash:
        result["sha256"] = sha256sum(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--sha256", action="store_true", help="Also calculate SHA-256 digests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for path in args.checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
        reports.append(inspect(path, include_hash=args.sha256))
    print(json.dumps(reports, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
