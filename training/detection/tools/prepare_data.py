import argparse
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from common import find_dataset_dirs, project_path, with_prefix

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
CATEGORY = {"id": 1, "name": "finger", "supercategory": "finger"}


def iter_xml_files(ann_dir):
    return sorted(
        path for path in ann_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".xml"
    )


def describe_annotation_layout(ann_dir):
    child_dirs = sorted(path for path in ann_dir.iterdir() if path.is_dir()) if ann_dir.is_dir() else []
    examples = [project_path(path) for path in child_dirs[:5]]
    return {
        "annotations_dir": project_path(ann_dir),
        "num_child_dirs": len(child_dirs),
        "child_dir_examples": examples,
    }


def text_float(parent, name):
    node = parent.find(name)
    return None if node is None or node.text is None else float(node.text.strip())


def image_size(xml_root, image_path):
    image_node = xml_root.find(".//image")
    if image_node is not None and image_node.get("width") and image_node.get("height"):
        return int(float(image_node.get("width"))), int(float(image_node.get("height")))
    with Image.open(image_path) as img:
        return img.size


def candidate_images(xml_root, xml_path, ann_dir, images_dir):
    rel_xml = xml_path.relative_to(ann_dir)
    for ext in IMAGE_EXTS:
        yield (images_dir / rel_xml).with_suffix(ext)

    image_node = xml_root.find(".//image")
    if image_node is not None and image_node.get("file"):
        image_file = Path(image_node.get("file"))
        yield images_dir / rel_xml.parent / image_file.name
        yield images_dir / image_file
        yield images_dir / image_file.name


def find_image(xml_root, xml_path, ann_dir, images_dir):
    seen = set()
    for path in candidate_images(xml_root, xml_path, ann_dir, images_dir):
        key = path.as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def parse_bbox(xml_root):
    bbox = xml_root.find(".//bbox")
    if bbox is None:
        return None
    cx = text_float(bbox, "center_x")
    cy = text_float(bbox, "center_y")
    bw = text_float(bbox, "width")
    bh = text_float(bbox, "height")
    if None in (cx, cy, bw, bh):
        return None
    return cx - bw / 2.0, cy - bh / 2.0, bw, bh


def parse_joint_points(xml_root):
    joint_points = xml_root.find(".//joint_points")
    if joint_points is None:
        return None

    points = []
    for point in joint_points.findall(".//point"):
        x = text_float(point, "x")
        y = text_float(point, "y")
        if None in (x, y):
            return None
        points.append({"x": round(x, 3), "y": round(y, 3)})

    return points if len(points) == 4 else None


def parse_rotation_angle(xml_root):
    angle = xml_root.find(".//rotation/angle")
    if angle is None or angle.text is None:
        return None
    return round(float(angle.text.strip()), 3)


def clip_xywh(box, width, height):
    x, y, w, h = box
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return [round(x1, 3), round(y1, 3), round(clipped_w, 3), round(clipped_h, 3)]


def convert_one(task):
    xml_path, ann_dir, images_dir, image_prefix = task
    xml_path = Path(xml_path)
    ann_dir = Path(ann_dir)
    images_dir = Path(images_dir)
    try:
        xml_root = ET.parse(xml_path).getroot()
        image_path = find_image(xml_root, xml_path, ann_dir, images_dir)
        if image_path is None:
            return None, {"xml": project_path(xml_path), "reason": "missing image"}

        width, height = image_size(xml_root, image_path)
        bbox = parse_bbox(xml_root)
        if bbox is None:
            return None, {"xml": project_path(xml_path), "reason": "missing bbox"}

        bbox = clip_xywh(bbox, width, height)
        if bbox is None:
            return None, {"xml": project_path(xml_path), "reason": "invalid bbox"}

        joint_points = parse_joint_points(xml_root)

        rotation_angle = parse_rotation_angle(xml_root)
        if rotation_angle is None:
            return None, {"xml": project_path(xml_path), "reason": "missing rotation angle"}

        rel_img = image_path.relative_to(images_dir)
        return {
            "file_name": with_prefix(image_prefix, rel_img),
            "width": width,
            "height": height,
            "bbox": bbox,
            "joint_points": joint_points,
            "rotation_angle": rotation_angle,
        }, None
    except Exception as exc:
        return None, {"xml": project_path(xml_path), "reason": repr(exc)}


def build_coco(items):
    images = []
    annotations = []
    for idx, item in enumerate(items, start=1):
        images.append(
            {
                "id": idx,
                "file_name": item["file_name"],
                "width": item["width"],
                "height": item["height"],
            }
        )
        bbox = item["bbox"]
        annotation = {
            "id": idx,
            "image_id": idx,
            "category_id": 1,
            "bbox": bbox,
            "rotation_angle": item["rotation_angle"],
            "area": round(bbox[2] * bbox[3], 3),
            "iscrowd": 0,
            "segmentation": [],
        }
        if item.get("joint_points") is not None:
            annotation["joint_points"] = item["joint_points"]
        annotations.append(annotation)
    return {
        "info": {"description": "FingerVeinSyn50K finger detection train set"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [CATEGORY],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="../../dataset/FingerVeinSyn50K")
    parser.add_argument("--out-dir", default="./data/fvsyn50k_coco")
    parser.add_argument("--coco-image-prefix", default="./dataset/FingerVeinSyn50K/Images")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    _, images_dir, ann_dir = find_dataset_dirs(args.dataset_root)
    xml_files = iter_xml_files(ann_dir)
    if args.limit:
        xml_files = xml_files[: args.limit]
    if not xml_files:
        raise RuntimeError(f"No XML files found: {describe_annotation_layout(ann_dir)}")

    tasks = [(str(path), str(ann_dir), str(images_dir), args.coco_image_prefix) for path in xml_files]
    if args.workers <= 1:
        results = map(convert_one, tasks)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(convert_one, tasks, chunksize=128)

    items = []
    skipped = []
    for index, (item, error) in enumerate(results, start=1):
        if item is not None:
            items.append(item)
        if error is not None:
            skipped.append(error)
        if index % 10000 == 0 or index == len(tasks):
            print(f"processed={index}/{len(tasks)} converted={len(items)} skipped={len(skipped)}", flush=True)

    out_dir = Path(args.out_dir)
    ann_dir_out = out_dir / "annotations"
    ann_dir_out.mkdir(parents=True, exist_ok=True)
    train_json = ann_dir_out / "train.json"
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(build_coco(items), f, ensure_ascii=False, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": project_path(args.dataset_root),
                "images_dir": project_path(images_dir),
                "annotations_dir": project_path(ann_dir),
                "images": len(items),
                "skipped": len(skipped),
                "annotation_file": project_path(train_json),
                "skipped_examples": skipped[:20],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print({"annotation_file": project_path(train_json), "images": len(items), "skipped": len(skipped)})


if __name__ == "__main__":
    main()
