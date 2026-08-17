"""Model registry matching the eight published release artifacts."""

from __future__ import annotations

from copy import deepcopy

from .detection import build_rotated_faster_rcnn, build_rotated_ssd, build_rotated_yolo
from .recognition import get_convnext, get_mbf_large, iresnet100
from .segmentation import build_model as build_segmentation_model


MODEL_SPECS = {
    "roi_rotated_faster_rcnn_r50_fpn": {
        "task": "roi_detection",
        "input_size": (300, 600),
        "architecture": "Rotated Faster R-CNN (ResNet-50-FPN)",
    },
    "roi_rotated_ssd_r50_fpn": {
        "task": "roi_detection",
        "input_size": (300, 600),
        "architecture": "Rotated SSD (ResNet-50-FPN)",
    },
    "roi_rotated_yolo_r50_fpn": {
        "task": "roi_detection",
        "input_size": (300, 600),
        "architecture": "Rotated YOLO-style detector (ResNet-50-FPN)",
    },
    "seg_finger_shape_unet_r50": {
        "task": "finger_shape_segmentation",
        "input_size": (300, 600),
        "architecture": "U-Net (ResNet-50 encoder)",
        "percentile_normalization": None,
    },
    "seg_vein_pattern_unet_r50": {
        "task": "vein_pattern_segmentation",
        "input_size": (100, 300),
        "architecture": "U-Net (ResNet-50 encoder)",
        "percentile_normalization": (10, 90),
    },
    "rec_mobilefacenet_large": {
        "task": "recognition",
        "input_size": (100, 300),
        "architecture": "MobileFaceNet-Large",
        "embedding_dim": 512,
    },
    "rec_iresnet100": {
        "task": "recognition",
        "input_size": (100, 300),
        "architecture": "iResNet-100",
        "embedding_dim": 512,
    },
    "rec_convnext_small": {
        "task": "recognition",
        "input_size": (224, 224),
        "architecture": "ConvNeXt-Small",
        "embedding_dim": 512,
    },
}


def model_spec(model_id: str):
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model_id={model_id!r}. Available: {', '.join(MODEL_SPECS)}")
    return deepcopy(MODEL_SPECS[model_id])


def build_model(model_id: str, **kwargs):
    """Build a model with the architecture expected by a release artifact."""
    model_spec(model_id)
    if model_id == "rec_mobilefacenet_large":
        return get_mbf_large(
            fp16=bool(kwargs.pop("fp16", False)),
            num_features=int(kwargs.pop("num_features", 512)),
        )
    if model_id == "rec_iresnet100":
        return iresnet100(
            fp16=bool(kwargs.pop("fp16", False)),
            num_features=int(kwargs.pop("num_features", 512)),
            **kwargs,
        )
    if model_id == "rec_convnext_small":
        return get_convnext(
            model_name="convnext_small",
            fp16=bool(kwargs.pop("fp16", False)),
            num_features=int(kwargs.pop("num_features", 512)),
            pretrained=bool(kwargs.pop("pretrained", False)),
            drop_path_rate=float(kwargs.pop("drop_path_rate", 0.1)),
        )
    if model_id in {"seg_finger_shape_unet_r50", "seg_vein_pattern_unet_r50"}:
        return build_segmentation_model(
            "unet_resnet50",
            encoder_weights=kwargs.pop("encoder_weights", "none"),
            use_smp=False,
            allow_weight_download=False,
            in_channels=1,
            **kwargs,
        )

    score_thresh = float(kwargs.pop("score_thresh", 0.05))
    nms_thresh = float(kwargs.pop("nms_thresh", 0.5))
    detections_per_img = int(kwargs.pop("detections_per_img", 100))
    common = {
        "image_size": (600, 300),
        "pretrained_backbone_path": None,
    }
    common.update(kwargs)
    if model_id == "roi_rotated_faster_rcnn_r50_fpn":
        return build_rotated_faster_rcnn(
            **common,
            box_detections_per_img=detections_per_img,
        )
    if model_id == "roi_rotated_ssd_r50_fpn":
        return build_rotated_ssd(
            **common,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )
    if model_id == "roi_rotated_yolo_r50_fpn":
        return build_rotated_yolo(
            **common,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )
    raise AssertionError(model_id)
