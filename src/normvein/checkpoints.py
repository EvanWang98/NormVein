"""Helpers for the model-only NormVein release artifact format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch


def load_artifact(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a trusted NormVein artifact and validate its public schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        artifact = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        artifact = torch.load(str(path), map_location=map_location)
    if not isinstance(artifact, Mapping):
        raise TypeError(f"{path}: expected a mapping, found {type(artifact).__name__}")
    required = {"format_version", "model_id", "state_dict", "metadata"}
    missing = sorted(required.difference(artifact))
    if missing:
        raise ValueError(f"{path}: missing artifact fields: {', '.join(missing)}")
    if artifact["format_version"] != 1:
        raise ValueError(f"{path}: unsupported format_version={artifact['format_version']}")
    state_dict = artifact["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{path}: state_dict is missing or empty")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items()):
        raise TypeError(f"{path}: state_dict must map string names to tensors")
    return dict(artifact)


def load_pretrained(
    model: torch.nn.Module,
    path: str | Path,
    *,
    expected_model_id: str | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a release artifact into an already constructed model."""
    artifact = load_artifact(path, map_location=map_location)
    if expected_model_id and artifact["model_id"] != expected_model_id:
        raise ValueError(
            f"Expected checkpoint for {expected_model_id}, found {artifact['model_id']}"
        )
    model.load_state_dict(artifact["state_dict"], strict=strict)
    return model, artifact


def load_manifest(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))

