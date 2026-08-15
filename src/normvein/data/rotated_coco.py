import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SIZE = (600, 300)  # width, height
ANGLE_SIGN = -1.0


class FVSynRotatedCocoDataset(Dataset):
    """Read the prepared COCO json with one extra rotation_angle per annotation."""

    def __init__(
        self,
        ann_file="./tools/data/fvsyn50k_coco/annotations/train.json",
        image_root=None,
        image_size=IMAGE_SIZE,
        angle_sign=ANGLE_SIGN,
        train=True,
        random_flip=True,
        minmax_normalize=True,
        root_dir=None,
    ):
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]
        self.ann_file = self.resolve_path(ann_file)
        self.image_root = self.resolve_path(image_root) if image_root else None
        self.image_size = tuple(image_size)
        self.angle_sign = float(angle_sign)
        self.train = bool(train)
        self.random_flip = bool(random_flip)
        self.minmax_normalize = bool(minmax_normalize)

        with open(self.ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        anns_by_image = {}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for image in coco.get("images", []):
            anns = anns_by_image.get(image["id"], [])
            if not anns:
                continue
            self.samples.append({"image": image, "annotations": anns})

        if not self.samples:
            raise RuntimeError(f"No valid annotated images found in {self.ann_file}")

    def resolve_path(self, path):
        if path is None:
            return None
        path = Path(path)
        candidates = [path]
        if not path.is_absolute():
            candidates.extend([self.root_dir / path, self.root_dir.parent / path])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    def resolve_image_path(self, file_name):
        raw = str(file_name).replace("\\", "/")
        path = Path(raw)
        candidates = [path]
        if raw.startswith("./"):
            candidates.extend([Path(raw[2:]), self.root_dir / raw[2:], self.root_dir.parent / raw[2:]])
        elif not path.is_absolute():
            candidates.extend([self.root_dir / path, self.root_dir.parent / path])

        if self.image_root is not None:
            candidates.append(self.image_root / path.name)
            parts = [part for part in Path(raw).parts if part not in {".", ""}]
            lowered = [part.lower() for part in parts]
            if "images" in lowered:
                index = lowered.index("images")
                candidates.append(self.image_root / Path(*parts[index + 1 :]))

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return path

    @staticmethod
    def minmax_to_uint8(array, eps=1e-6):
        array = array.astype(np.float32, copy=False)
        finite = np.isfinite(array)
        if not finite.any():
            return np.zeros_like(array, dtype=np.uint8)
        valid = array[finite]
        min_value = float(valid.min())
        max_value = float(valid.max())
        value_range = max_value - min_value
        if value_range <= eps:
            return np.zeros_like(array, dtype=np.uint8)
        safe_array = np.where(finite, array, min_value)
        normalized = (safe_array - min_value) * (255.0 / value_range)
        return np.clip(np.rint(normalized), 0, 255).astype(np.uint8)

    def load_image(self, image_path):
        with Image.open(image_path) as img:
            gray = img.convert("L")
            original_size = gray.size
            gray = gray.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(gray)
        if self.minmax_normalize:
            array = self.minmax_to_uint8(array)
        tensor = torch.from_numpy(array.copy()).float().unsqueeze(0) / 127.5 - 1.0
        return tensor, original_size

    def build_target(self, sample, original_size):
        image_info = sample["image"]
        orig_w = float(image_info.get("width", original_size[0]))
        orig_h = float(image_info.get("height", original_size[1]))
        dst_w, dst_h = self.image_size
        scale_x = float(dst_w) / max(orig_w, 1.0)
        scale_y = float(dst_h) / max(orig_h, 1.0)

        boxes = []
        angles = []
        areas = []
        for ann in sample["annotations"]:
            x, y, w, h = [float(v) for v in ann["bbox"][:4]]
            x1 = max(0.0, min(float(dst_w), x * scale_x))
            y1 = max(0.0, min(float(dst_h), y * scale_y))
            x2 = max(0.0, min(float(dst_w), (x + w) * scale_x))
            y2 = max(0.0, min(float(dst_h), (y + h) * scale_y))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            angles.append(self.angle_sign * float(ann.get("rotation_angle", ann.get("angle", 0.0))))
            areas.append((x2 - x1) * (y2 - y1))

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            angles = torch.zeros((0,), dtype=torch.float32)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            angles = torch.as_tensor(angles, dtype=torch.float32)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        bbox_cxcywh = self.boxes_to_cxcywh(boxes)
        rboxes = torch.cat([bbox_cxcywh, angles.reshape(-1, 1)], dim=1)

        target = {
            "boxes": boxes,
            "rboxes": rboxes,
            "labels": labels,
            "angles": angles,
            "image_id": torch.tensor([int(image_info["id"])]),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return target

    @staticmethod
    def boxes_to_cxcywh(boxes):
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)
    @staticmethod
    def horizontal_flip(image, target):
        width = image.shape[-1]
        image = torch.flip(image, dims=[-1])
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = width - x2
            boxes[:, 2] = width - x1
            target = dict(target)
            target["boxes"] = boxes
            target["angles"] = -target["angles"]
            target["rboxes"] = torch.cat([FVSynRotatedCocoDataset.boxes_to_cxcywh(boxes), target["angles"].reshape(-1, 1)], dim=1)
        return image, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.resolve_image_path(sample["image"]["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {sample['image']['file_name']} -> {image_path}")
        image, original_size = self.load_image(image_path)
        target = self.build_target(sample, original_size)
        if self.train and self.random_flip and random.random() < 0.5:
            image, target = self.horizontal_flip(image, target)
        target["image_path"] = str(image_path)
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)