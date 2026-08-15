import os
from pathlib import Path


def as_path(path):
    return Path(path)


def project_path(path):
    text = Path(path).as_posix()
    if text == ".":
        return "."
    if text.startswith(("./", "../", "/")) or Path(path).is_absolute():
        return text
    return f"./{text}"


def normalize_prefix(prefix):
    prefix = str(prefix).replace("\\", "/").rstrip("/")
    if prefix and not prefix.startswith(("./", "../", "/")):
        prefix = f"./{prefix}"
    return prefix


def with_prefix(prefix, relative_path):
    prefix = normalize_prefix(prefix)
    rel = Path(relative_path).as_posix().lstrip("/")
    return f"{prefix}/{rel}" if prefix else rel


def find_child_dir(root, names, xml_required=False):
    candidates = [root / name for name in names if (root / name).is_dir()]
    if xml_required:
        for candidate in candidates:
            if any(path.is_file() and path.suffix.lower() == ".xml" for path in candidate.rglob("*")):
                return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing one of {names} under {root}")


def find_dataset_dirs(dataset_root):
    root = as_path(dataset_root)
    images_dir = find_child_dir(root, ["raw_images", "Images", "images"])
    annotations_dir = find_child_dir(root, ["Annotations", "annotations"], xml_required=True)
    return root, images_dir, annotations_dir


def custom_import_pythonpath():
    parts = []
    if Path("detector_pretrain").is_dir():
        parts.append(".")
    if Path("datasets").is_dir() and Path("models").is_dir():
        parts.append("..")
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    seen = set()
    unique = []
    for part in parts:
        if part and part not in seen:
            unique.append(part)
            seen.add(part)
    return os.pathsep.join(unique)