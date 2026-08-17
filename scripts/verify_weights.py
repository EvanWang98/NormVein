#!/usr/bin/env python3
"""Verify downloaded NormVein model files."""

import hashlib
import json
from pathlib import Path


WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((WEIGHTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        path = WEIGHTS_DIR / model["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != model["size_bytes"] or sha256sum(path) != model["sha256"]:
            raise RuntimeError(f"Invalid model file: {path.name}")
    print(f"Verified {len(manifest['models'])} pretrained models.")


if __name__ == "__main__":
    main()
