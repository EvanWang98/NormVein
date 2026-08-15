from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import boxes as box_ops


DEFAULT_IMAGE_SIZE = (600, 300)  # width, height
ANCHOR_SIZES = ((32,), (64,), (128,), (256,), (512,))
ANCHOR_ASPECT_RATIOS = ((0.5, 0.4, 1.0 / 3.0, 0.3, 0.25),) * len(ANCHOR_SIZES)


def build_torchvision_faster_rcnn(
    num_classes=2,
    image_size=DEFAULT_IMAGE_SIZE,
    trainable_backbone_layers=None,
    rpn_pre_nms_top_n_train=2000,
    rpn_pre_nms_top_n_test=1000,
    rpn_post_nms_top_n_train=2000,
    rpn_post_nms_top_n_test=1000,
    box_batch_size_per_image=512,
    box_positive_fraction=0.25,
    box_detections_per_img=100,
):
    min_size = min(image_size)
    max_size = max(image_size)
    kwargs = dict(
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        image_mean=[0.0],
        image_std=[1.0],
        rpn_pre_nms_top_n_train=int(rpn_pre_nms_top_n_train),
        rpn_pre_nms_top_n_test=int(rpn_pre_nms_top_n_test),
        rpn_post_nms_top_n_train=int(rpn_post_nms_top_n_train),
        rpn_post_nms_top_n_test=int(rpn_post_nms_top_n_test),
        box_batch_size_per_image=int(box_batch_size_per_image),
        box_positive_fraction=float(box_positive_fraction),
        box_detections_per_img=int(box_detections_per_img),
        rpn_anchor_generator=AnchorGenerator(
            sizes=ANCHOR_SIZES,
            aspect_ratios=ANCHOR_ASPECT_RATIOS,
        ),
    )
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers
    try:
        return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, **kwargs)
    except TypeError:
        return fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, **kwargs)


def load_local_resnet50_weights(detector, weights_path):
    if not weights_path:
        return None
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        return None
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = detector.backbone.body.load_state_dict(state, strict=False)
    return {"path": str(weights_path), "missing": missing, "unexpected": unexpected}


def patch_first_conv_to_single_channel(detector):
    conv = detector.backbone.body.conv1
    if conv.in_channels == 1:
        return
    new_conv = nn.Conv2d(
        1,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    detector.backbone.body.conv1 = new_conv


def xyxy_to_cxcywh(boxes):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1)


def xyxy_to_rboxes(boxes, angle=0.0):
    cxcywh = xyxy_to_cxcywh(boxes)
    angles = cxcywh.new_full((cxcywh.shape[0], 1), float(angle))
    return torch.cat([cxcywh, angles], dim=1)


def normalize_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0


def encode_rboxes(reference_boxes, target_boxes):
    wx, wy, ww, wh, wa = 10.0, 10.0, 5.0, 5.0, 1.0
    ex_ctr_x, ex_ctr_y, ex_w, ex_h, ex_a = reference_boxes.unbind(dim=1)
    gt_ctr_x, gt_ctr_y, gt_w, gt_h, gt_a = target_boxes.unbind(dim=1)
    ex_w = ex_w.clamp(min=1e-6)
    ex_h = ex_h.clamp(min=1e-6)
    gt_w = gt_w.clamp(min=1e-6)
    gt_h = gt_h.clamp(min=1e-6)
    dx = wx * (gt_ctr_x - ex_ctr_x) / ex_w
    dy = wy * (gt_ctr_y - ex_ctr_y) / ex_h
    dw = ww * torch.log(gt_w / ex_w)
    dh = wh * torch.log(gt_h / ex_h)
    da = wa * normalize_angle(gt_a - ex_a)
    return torch.stack((dx, dy, dw, dh, da), dim=1)


def decode_rboxes(reference_boxes, deltas):
    wx, wy, ww, wh, wa = 10.0, 10.0, 5.0, 5.0, 1.0
    boxes = reference_boxes.to(deltas.dtype)
    ctr_x, ctr_y, widths, heights, angles = boxes.unbind(dim=1)
    widths = widths.clamp(min=1e-6)
    heights = heights.clamp(min=1e-6)
    dx = deltas[:, 0::5] / wx
    dy = deltas[:, 1::5] / wy
    dw = (deltas[:, 2::5] / ww).clamp(max=4.135)
    dh = (deltas[:, 3::5] / wh).clamp(max=4.135)
    da = deltas[:, 4::5] / wa
    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]
    pred_a = normalize_angle(da + angles[:, None])
    return torch.stack((pred_ctr_x, pred_ctr_y, pred_w, pred_h, pred_a), dim=2).flatten(1)


def rboxes_to_enclosing_xyxy(rboxes):
    cx, cy, w, h, angle = rboxes.unbind(dim=-1)
    theta = angle * torch.pi / 180.0
    cos_t = torch.abs(torch.cos(theta))
    sin_t = torch.abs(torch.sin(theta))
    out_w = w * cos_t + h * sin_t
    out_h = w * sin_t + h * cos_t
    return torch.stack((cx - out_w * 0.5, cy - out_h * 0.5, cx + out_w * 0.5, cy + out_h * 0.5), dim=-1)


def clip_rboxes_to_image(rboxes, image_shape):
    height, width = image_shape
    clipped = rboxes.clone()
    clipped[:, 0] = clipped[:, 0].clamp(min=0.0, max=float(width))
    clipped[:, 1] = clipped[:, 1].clamp(min=0.0, max=float(height))
    clipped[:, 2] = clipped[:, 2].clamp(min=1e-3, max=float(width))
    clipped[:, 3] = clipped[:, 3].clamp(min=1e-3, max=float(height))
    clipped[:, 4] = normalize_angle(clipped[:, 4])
    return clipped


def rbox_to_polygon_np(rbox):
    import math
    cx, cy, w, h, angle = [float(v) for v in rbox]
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points = []
    for dx, dy in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
        points.append((cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t))
    return points


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def polygon_clip(subject, clip):
    def inside(point, edge_start, edge_end):
        return (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]) >= 0

    def intersection(p1, p2, e1, e2):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = e1
        x4, y4 = e2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p2
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (px, py)

    output = subject
    for i in range(len(clip)):
        input_list = output
        output = []
        if not input_list:
            break
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        s = input_list[-1]
        for e in input_list:
            if inside(e, edge_start, edge_end):
                if not inside(s, edge_start, edge_end):
                    output.append(intersection(s, e, edge_start, edge_end))
                output.append(e)
            elif inside(s, edge_start, edge_end):
                output.append(intersection(s, e, edge_start, edge_end))
            s = e
    return output


def rotated_iou_np(box_a, box_b):
    poly_a = rbox_to_polygon_np(box_a)
    poly_b = rbox_to_polygon_np(box_b)
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter = polygon_area(polygon_clip(poly_a, poly_b))
    return inter / max(area_a + area_b - inter, 1e-12)


def rotated_nms(rboxes, scores, iou_threshold):
    if rboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=rboxes.device)
    order = torch.argsort(scores, descending=True).detach().cpu().tolist()
    boxes_cpu = rboxes.detach().cpu().tolist()
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining = []
        for idx in order:
            if rotated_iou_np(boxes_cpu[current], boxes_cpu[idx]) <= iou_threshold:
                remaining.append(idx)
        order = remaining
    return torch.as_tensor(keep, dtype=torch.long, device=rboxes.device)


class RotatedRoIAlign(nn.Module):
    def __init__(self, output_size=7, sampling_ratio=2, canonical_scale=224, canonical_level=4, roi_chunk_size=128):
        super().__init__()
        self.output_size = int(output_size)
        self.sampling_ratio = int(sampling_ratio)
        self.canonical_scale = int(canonical_scale)
        self.canonical_level = int(canonical_level)
        self.roi_chunk_size = int(roi_chunk_size)

    def map_levels(self, rboxes):
        if rboxes.numel() == 0:
            return rboxes.new_zeros((0,), dtype=torch.long)
        scale = torch.sqrt((rboxes[:, 2] * rboxes[:, 3]).clamp(min=1e-6))
        target = torch.floor(self.canonical_level + torch.log2(scale / self.canonical_scale + 1e-6))
        return (target.clamp(min=2, max=5) - 2).to(dtype=torch.long)

    def make_grid(self, rboxes, feature_shape, image_shape):
        device = rboxes.device
        dtype = rboxes.dtype
        feat_h, feat_w = feature_shape
        img_h, img_w = image_shape
        scale_x = float(feat_w) / float(img_w)
        scale_y = float(feat_h) / float(img_h)
        cx = rboxes[:, 0] * scale_x
        cy = rboxes[:, 1] * scale_y
        w = rboxes[:, 2].clamp(min=1e-6) * scale_x
        h = rboxes[:, 3].clamp(min=1e-6) * scale_y
        theta = rboxes[:, 4] * torch.pi / 180.0
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.output_size, device=device, dtype=dtype),
            torch.arange(self.output_size, device=device, dtype=dtype),
            indexing="ij",
        )
        base_x = (grid_x + 0.5) / self.output_size - 0.5
        base_y = (grid_y + 0.5) / self.output_size - 0.5
        local_x = base_x[None, :, :] * w[:, None, None]
        local_y = base_y[None, :, :] * h[:, None, None]
        sample_x = cx[:, None, None] + local_x * cos_t[:, None, None] - local_y * sin_t[:, None, None]
        sample_y = cy[:, None, None] + local_x * sin_t[:, None, None] + local_y * cos_t[:, None, None]
        norm_x = ((sample_x + 0.5) / max(float(feat_w), 1.0)) * 2.0 - 1.0
        norm_y = ((sample_y + 0.5) / max(float(feat_h), 1.0)) * 2.0 - 1.0
        return torch.stack((norm_x, norm_y), dim=-1)

    def forward(self, features, rboxes, image_shapes):
        feature_list = [features[name] for name in sorted(features.keys())[:4]]
        channels = feature_list[0].shape[1]
        total = sum(int(boxes.shape[0]) for boxes in rboxes)
        if total == 0:
            return feature_list[0].new_zeros((0, channels, self.output_size, self.output_size))

        all_boxes = torch.cat(rboxes, dim=0)
        batch_indices = torch.cat([
            boxes.new_full((boxes.shape[0],), image_idx, dtype=torch.long)
            for image_idx, boxes in enumerate(rboxes)
        ], dim=0)
        levels = self.map_levels(all_boxes)
        output = feature_list[0].new_zeros((total, channels, self.output_size, self.output_size))

        chunk_size = max(self.roi_chunk_size, 1)
        for level_idx, feature in enumerate(feature_list):
            for image_idx, image_shape in enumerate(image_shapes):
                roi_indices = torch.where((levels == level_idx) & (batch_indices == image_idx))[0]
                if roi_indices.numel() == 0:
                    continue
                image_feature = feature[image_idx:image_idx + 1]
                for chunk_indices in roi_indices.split(chunk_size):
                    boxes_chunk = all_boxes[chunk_indices]
                    grids = self.make_grid(boxes_chunk, feature.shape[-2:], image_shape)
                    expanded_feature = image_feature.expand(grids.shape[0], -1, -1, -1)
                    pooled = F.grid_sample(
                        expanded_feature,
                        grids,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                    output[chunk_indices] = pooled
        return output


class RotatedFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 5)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class RotatedFasterRCNN(nn.Module):
    """Standard-style Rotated Faster R-CNN with rotated RoIAlign and 5-delta box head."""

    def __init__(
        self,
        num_classes=2,
        image_size=DEFAULT_IMAGE_SIZE,
        roi_pool_size=7,
        roi_chunk_size=128,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        box_detections_per_img=100,
        bbox_loss_weight=1.0,
        bbox_loss_beta=1.0 / 9.0,
        min_area_ratio=0.25,
        area_reg_weight=0.1,
        pretrained_backbone_path="./weight/resnet50-0676ba61.pth",
        trainable_backbone_layers=None,
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.num_classes = int(num_classes)
        self.bbox_loss_weight = float(bbox_loss_weight)
        self.bbox_loss_beta = float(bbox_loss_beta)
        self.min_area_ratio = float(min_area_ratio)
        self.area_reg_weight = float(area_reg_weight)
        self.detector = build_torchvision_faster_rcnn(
            num_classes=num_classes,
            image_size=self.image_size,
            trainable_backbone_layers=trainable_backbone_layers,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            box_batch_size_per_image=box_batch_size_per_image,
            box_positive_fraction=box_positive_fraction,
            box_detections_per_img=box_detections_per_img,
        )
        self.pretrained_info = load_local_resnet50_weights(self.detector, pretrained_backbone_path)
        patch_first_conv_to_single_channel(self.detector)
        predictor_in = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = RotatedFastRCNNPredictor(predictor_in, num_classes)
        self.rotated_roi_align = RotatedRoIAlign(output_size=roi_pool_size, roi_chunk_size=roi_chunk_size)

    @property
    def pool_size(self):
        return self.rotated_roi_align.output_size

    def check_targets(self, targets):
        if targets is None:
            raise ValueError("targets should not be None in training mode")
        for target in targets:
            if "boxes" not in target or "rboxes" not in target or "labels" not in target:
                raise ValueError("Each target must contain boxes, rboxes and labels")

    def prepare_training_samples(self, proposals, targets, image_sizes):
        gt_boxes = [target["boxes"] for target in targets]
        gt_labels = [target["labels"] for target in targets]
        gt_rboxes = [target["rboxes"] for target in targets]
        proposals = self.detector.roi_heads.add_gt_proposals(proposals, gt_boxes)
        proposal_rboxes = [
            torch.cat([xyxy_to_rboxes(prop[:-len(gt)] if len(gt) > 0 else prop), gt], dim=0)
            if len(gt) > 0 else xyxy_to_rboxes(prop)
            for prop, gt in zip(proposals, gt_rboxes)
        ]
        matched_idxs, labels = self.detector.roi_heads.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        sampled_inds = self.detector.roi_heads.subsample(labels)
        sampled_props = []
        sampled_rprops = []
        sampled_labels = []
        regression_targets = []
        sampled_target_rboxes = []
        sampled_image_areas = []
        for img_id, inds in enumerate(sampled_inds):
            props_per_img = proposals[img_id][inds]
            rprops_per_img = proposal_rboxes[img_id][inds]
            labels_per_img = labels[img_id][inds]
            matched_per_img = matched_idxs[img_id][inds]
            target_rboxes = gt_rboxes[img_id][matched_per_img]
            image_h, image_w = image_sizes[img_id]
            image_area = float(image_h * image_w)
            regression_targets.append(encode_rboxes(rprops_per_img, target_rboxes))
            sampled_props.append(props_per_img)
            sampled_rprops.append(rprops_per_img)
            sampled_labels.append(labels_per_img)
            sampled_target_rboxes.append(target_rboxes)
            sampled_image_areas.append(rprops_per_img.new_full((rprops_per_img.shape[0],), image_area))
        return sampled_props, sampled_rprops, sampled_labels, regression_targets, sampled_target_rboxes, sampled_image_areas

    def rotated_fastrcnn_loss(
        self,
        class_logits,
        box_regression,
        labels,
        regression_targets,
        proposal_rboxes,
        target_rboxes,
        image_areas,
    ):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        proposal_rboxes = torch.cat(proposal_rboxes, dim=0)
        target_rboxes = torch.cat(target_rboxes, dim=0)
        image_areas = torch.cat(image_areas, dim=0).clamp(min=1.0)
        classification_loss = F.cross_entropy(class_logits, labels)
        sampled_pos_inds = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds]
        box_regression = box_regression.reshape(box_regression.shape[0], self.num_classes, 5)
        area_reg_loss = box_regression.sum() * 0.0
        if sampled_pos_inds.numel() == 0:
            box_loss = box_regression.sum() * 0.0
        else:
            pred_deltas = box_regression[sampled_pos_inds, labels_pos]
            box_loss = F.smooth_l1_loss(
                pred_deltas,
                regression_targets[sampled_pos_inds],
                beta=self.bbox_loss_beta,
                reduction="sum",
            )
            box_loss = box_loss / labels.numel()
            if self.area_reg_weight > 0.0 and self.min_area_ratio > 0.0:
                pos_target_rboxes = target_rboxes[sampled_pos_inds]
                pos_image_areas = image_areas[sampled_pos_inds]
                target_area_ratio = (pos_target_rboxes[:, 2] * pos_target_rboxes[:, 3]) / pos_image_areas
                valid = target_area_ratio >= self.min_area_ratio
                if bool(valid.any().item()):
                    pred_rboxes = decode_rboxes(proposal_rboxes[sampled_pos_inds], pred_deltas)
                    pred_area_ratio = (pred_rboxes[:, 2] * pred_rboxes[:, 3]) / pos_image_areas
                    deficit = F.relu(self.min_area_ratio - pred_area_ratio[valid]) / max(self.min_area_ratio, 1e-6)
                    area_reg_loss = deficit.pow(2).mean() * self.area_reg_weight
        return classification_loss, box_loss * self.bbox_loss_weight, area_reg_loss

    def run_box_head_training(self, features, proposals, image_sizes, targets):
        _, sampled_rprops, labels, regression_targets, target_rboxes, image_areas = self.prepare_training_samples(proposals, targets, image_sizes)
        box_features = self.rotated_roi_align(features, sampled_rprops, image_sizes)
        box_features = self.detector.roi_heads.box_head(box_features)
        class_logits, box_regression = self.detector.roi_heads.box_predictor(box_features)
        loss_classifier, loss_rbox_reg, loss_area_reg = self.rotated_fastrcnn_loss(
            class_logits,
            box_regression,
            labels,
            regression_targets,
            sampled_rprops,
            target_rboxes,
            image_areas,
        )
        return {"loss_classifier": loss_classifier, "loss_rbox_reg": loss_rbox_reg, "loss_area_reg": loss_area_reg}

    def postprocess_rotated_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes.shape[0] for boxes in proposals]
        proposal_rboxes = [xyxy_to_rboxes(boxes) for boxes in proposals]
        concat_props = torch.cat(proposal_rboxes, dim=0)
        pred_boxes = decode_rboxes(concat_props, box_regression)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        detections = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            labels = torch.arange(num_classes, device=device).view(1, -1).expand_as(scores)
            boxes = boxes[:, 5:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            boxes = boxes.reshape(-1, 5)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            boxes = clip_rboxes_to_image(boxes, image_shape)
            keep = torch.where(scores > self.detector.roi_heads.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = torch.where((boxes[:, 2] > 1e-2) & (boxes[:, 3] > 1e-2))[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep = rotated_nms(boxes, scores, self.detector.roi_heads.nms_thresh)
            keep = keep[: self.detector.roi_heads.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            detections.append(
                {
                    "rboxes": boxes,
                    "bbox_cxcywha": boxes,
                    "boxes": rboxes_to_enclosing_xyxy(boxes),
                    "scores": scores,
                    "labels": labels,
                    "angles": boxes[:, 4] if boxes.numel() > 0 else boxes.new_zeros((0,)),
                }
            )
        return detections

    def run_box_head_inference(self, features, proposals, image_sizes):
        proposal_rboxes = [xyxy_to_rboxes(boxes) for boxes in proposals]
        box_features = self.rotated_roi_align(features, proposal_rboxes, image_sizes)
        box_features = self.detector.roi_heads.box_head(box_features)
        class_logits, box_regression = self.detector.roi_heads.box_predictor(box_features)
        return self.postprocess_rotated_detections(class_logits, box_regression, proposals, image_sizes)

    def forward(self, images, targets=None):
        if self.training:
            self.check_targets(targets)
        original_image_sizes = []
        for image in images:
            if image.dim() != 3:
                raise ValueError(f"Expected each image to be [C, H, W], got {tuple(image.shape)}")
            original_image_sizes.append((image.shape[-2], image.shape[-1]))

        images, targets = self.detector.transform(images, targets)
        features = self.detector.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.detector.rpn(images, features, targets)

        if self.training:
            detector_losses = self.run_box_head_training(features, proposals, images.image_sizes, targets)
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        detections = self.run_box_head_inference(features, proposals, images.image_sizes)
        detections = self.detector.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections


def build_rotated_faster_rcnn(**kwargs):
    return RotatedFasterRCNN(**kwargs)