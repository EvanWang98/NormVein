#!/usr/bin/env python3
"""Create the deterministic 10K-identity/500K-image recognition manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def images_under(path: Path):
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image-subdir", default="roi_images")
    parser.add_argument("--identities", type=int, default=10000)
    parser.add_argument("--images-per-identity", type=int, default=50)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    image_root = args.data_root / args.image_subdir
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    identity_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    if len(identity_dirs) < args.identities:
        raise RuntimeError(
            f"Requested {args.identities} identities but found only {len(identity_dirs)} in {image_root}"
        )

    samples = []
    identities = []
    for label, identity_dir in enumerate(identity_dirs[: args.identities]):
        identity_images = images_under(identity_dir)
        if len(identity_images) < args.images_per_identity:
            raise RuntimeError(
                f"Identity {identity_dir.name} contains {len(identity_images)} images; "
                f"{args.images_per_identity} are required"
            )
        identities.append({"label": label, "identity": identity_dir.name})
        for path in identity_images[: args.images_per_identity]:
            samples.append(
                {
                    "path": path.relative_to(image_root).as_posix(),
                    "label": label,
                    "identity": identity_dir.name,
                }
            )

    payload = {
        "format_version": 1,
        "image_root": args.image_subdir,
        "selection": "lexicographically first identities and images",
        "num_identities": len(identities),
        "images_per_identity": args.images_per_identity,
        "num_images": len(samples),
        "identities": identities,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} samples from {len(identities)} identities to {args.output}")


if __name__ == "__main__":
    main()

