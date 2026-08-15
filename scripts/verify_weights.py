#!/usr/bin/env python3
"""Verify all local release assets against weights/manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "weights" / "manifest.json")
    parser.add_argument("--weights-root", type=Path, default=REPO_ROOT / "weights")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for record in manifest["models"]:
        path = args.weights_root / record["relative_path"]
        if not path.is_file():
            failures.append(f"MISSING {record['model_id']}: {path}")
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256sum(path)
        if actual_size != record["size_bytes"]:
            failures.append(
                f"SIZE {record['model_id']}: expected {record['size_bytes']}, got {actual_size}"
            )
        if actual_sha != record["sha256"]:
            failures.append(
                f"SHA256 {record['model_id']}: expected {record['sha256']}, got {actual_sha}"
            )
        if not failures or not failures[-1].startswith(("SIZE ", "SHA256 ")):
            print(f"OK {record['model_id']} {actual_sha}")
        elif actual_size == record["size_bytes"] and actual_sha == record["sha256"]:
            print(f"OK {record['model_id']} {actual_sha}")
    if failures:
        raise SystemExit("Weight verification failed:\n" + "\n".join(failures))
    print(f"Verified {len(manifest['models'])} release assets.")


if __name__ == "__main__":
    main()

