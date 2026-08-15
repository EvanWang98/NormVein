"""Command-line inference for NormVein release models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .checkpoints import load_pretrained
from .models import MODEL_SPECS, build_model, model_spec
from .preprocessing import preprocess_image


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def load_model(model_id: str, checkpoint: Path, device: torch.device):
    model = build_model(model_id)
    model, artifact = load_pretrained(
        model,
        checkpoint,
        expected_model_id=model_id,
        map_location="cpu",
    )
    model.to(device).eval()
    return model, artifact


def recognize(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, artifact = load_model(args.model, args.checkpoint, device)
    tensor = preprocess_image(args.input, args.model).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor).float()
        embedding = torch.nn.functional.normalize(embedding, dim=1)
    output = embedding[0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(
        json.dumps(
            {
                "model_id": artifact["model_id"],
                "input": str(args.input),
                "output": str(args.output),
                "shape": list(output.shape),
                "l2_norm": float(np.linalg.norm(output)),
            },
            indent=2,
        )
    )


def segment(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, artifact = load_model(args.model, args.checkpoint, device)
    tensor = preprocess_image(args.input, args.model).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask = (probability >= args.threshold).astype(np.uint8) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(args.output)
    print(
        json.dumps(
            {
                "model_id": artifact["model_id"],
                "input": str(args.input),
                "output": str(args.output),
                "threshold": args.threshold,
                "foreground_ratio": float((mask > 0).mean()),
            },
            indent=2,
        )
    )


def _prediction_records(output: dict, score_threshold: float, top_k: int):
    scores = output.get("scores", torch.empty(0))
    order = torch.argsort(scores, descending=True)[:top_k]
    records = []
    for tensor_index in order:
        index = int(tensor_index.item())
        score = float(scores[index].item())
        if score < score_threshold:
            continue
        rboxes = output.get("bbox_cxcywha", output.get("rboxes"))
        record = {
            "score": score,
            "label": int(output.get("labels", torch.ones_like(scores, dtype=torch.long))[index].item()),
            "bbox_xyxy": [float(v) for v in output["boxes"][index].tolist()],
        }
        if rboxes is not None and len(rboxes) > index:
            record["bbox_cxcywha"] = [float(v) for v in rboxes[index].tolist()]
        records.append(record)
    return records


def detect(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, artifact = load_model(args.model, args.checkpoint, device)
    tensor = preprocess_image(args.input, args.model).to(device)
    with torch.inference_mode():
        output = model([tensor])[0]
    records = _prediction_records(output, args.score_threshold, args.top_k)
    result = {
        "model_id": artifact["model_id"],
        "input": str(args.input),
        "input_size": list(model_spec(args.model)["input_size"]),
        "detections": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def add_common(parser: argparse.ArgumentParser, models: list[str]) -> None:
    parser.add_argument("--model", required=True, choices=models)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="normvein", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    recognition_models = [key for key, value in MODEL_SPECS.items() if value["task"] == "recognition"]
    segmentation_models = [key for key, value in MODEL_SPECS.items() if "segmentation" in value["task"]]
    detection_models = [key for key, value in MODEL_SPECS.items() if value["task"] == "roi_detection"]

    rec = subparsers.add_parser("recognize", help="extract a normalized 512-D embedding")
    add_common(rec, recognition_models)
    rec.set_defaults(func=recognize)

    seg = subparsers.add_parser("segment", help="write a binary segmentation mask")
    add_common(seg, segmentation_models)
    seg.add_argument("--threshold", type=float, default=0.5)
    seg.set_defaults(func=segment)

    det = subparsers.add_parser("detect", help="write rotated ROI detections as JSON")
    add_common(det, detection_models)
    det.add_argument("--score-threshold", type=float, default=0.25)
    det.add_argument("--top-k", type=int, default=5)
    det.set_defaults(func=detect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

