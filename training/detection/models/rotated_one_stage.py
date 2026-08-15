from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotated_faster_rcnn import (
    DEFAULT_IMAGE_SIZE,
    ANCHOR_SIZES,
    ANCHOR_ASPECT_RATIOS,
    build_torchvision_faster_rcnn,
    load_local_resnet50_weights,
    patch_first_conv_to_single_channel,
    encode_rboxes,
    decode_rboxes,
    clip_rboxes_to_image,
    rboxes_to_enclosing_xyxy,
    rotated_nms,
)


FCOS_STRIDES = (4, 8, 16, 32)
FCOS_REGRESS_RANGES = ((0, 96), (64, 192), (128, 384), (256, 1e8))
ONE_STAGE_PRE_NMS_TOPK = 100


def build_backbone(image_size=DEFAULT_IMAGE_SIZE, pretrained_backbone_path="./weight/resnet50-0676ba61.pth", trainable_backbone_layers=None):
    detector = build_torchvision_faster_rcnn(
        num_classes=2,
        image_size=image_size,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    load_local_resnet50_weights(detector, pretrained_backbone_path)
    patch_first_conv_to_single_channel(detector)
    return detector.backbone


def feature_list_from_backbone(features, max_levels=4):
    if isinstance(features, torch.Tensor):
        return [features]
    if isinstance(features, OrderedDict):
        return [features[key] for key in sorted(features.keys())[:max_levels]]
    return list(features)[:max_levels]


def stack_images(images):
    return torch.stack(images, dim=0)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def xyxy_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


def make_grid_points(feature, stride):
    h, w = feature.shape[-2:]
    ys = (torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5) * float(stride)
    xs = (torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5) * float(stride)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def make_anchors(feature, stride, image_size, ratios):
    points = make_grid_points(feature, stride)
    anchors = []
    base = float(stride) * 8.0
    for ratio in ratios:
        ratio = float(ratio)
        w = base / (ratio ** 0.5)
        h = base * (ratio ** 0.5)
        cx = points[:, 0]
        cy = points[:, 1]
        angle = points.new_zeros((points.shape[0],))
        anchors.append(torch.stack((cx, cy, cx.new_full(cx.shape, w), cy.new_full(cy.shape, h), angle), dim=1))
    return torch.cat(anchors, dim=0)


def flatten_head_output(tensor, channels):
    n = tensor.shape[0]
    return tensor.permute(0, 2, 3, 1).reshape(n, -1, channels)


def postprocess_rboxes(rboxes, scores, image_shape, score_thresh, nms_thresh, detections_per_img):
    if rboxes.numel() == 0:
        empty = rboxes.new_zeros((0,))
        return {
            "rboxes": rboxes.new_zeros((0, 5)),
            "bbox_cxcywha": rboxes.new_zeros((0, 5)),
            "boxes": rboxes.new_zeros((0, 4)),
            "scores": empty,
            "labels": empty.to(dtype=torch.long),
            "angles": empty,
        }
    rboxes = clip_rboxes_to_image(rboxes, image_shape)
    keep = torch.where(scores > float(score_thresh))[0]
    rboxes = rboxes[keep]
    scores = scores[keep]
    if rboxes.numel() == 0:
        return postprocess_rboxes(rboxes, scores, image_shape, -1.0, nms_thresh, detections_per_img)
    keep = torch.where((rboxes[:, 2] > 1e-2) & (rboxes[:, 3] > 1e-2))[0]
    rboxes = rboxes[keep]
    scores = scores[keep]
    if rboxes.numel() == 0:
        return postprocess_rboxes(rboxes, scores, image_shape, -1.0, nms_thresh, detections_per_img)
    if int(detections_per_img) <= 1:
        top = int(torch.argmax(scores).item())
        rboxes = rboxes[top: top + 1]
        scores = scores[top: top + 1]
        return {
            "rboxes": rboxes,
            "bbox_cxcywha": rboxes,
            "boxes": rboxes_to_enclosing_xyxy(rboxes),
            "scores": scores,
            "labels": torch.ones((1,), dtype=torch.long, device=rboxes.device),
            "angles": rboxes[:, 4],
        }
    if scores.numel() > ONE_STAGE_PRE_NMS_TOPK:
        scores, order = scores.topk(ONE_STAGE_PRE_NMS_TOPK, sorted=True)
        rboxes = rboxes[order]
    keep = rotated_nms(rboxes, scores, nms_thresh)
    keep = keep[: int(detections_per_img)]
    rboxes = rboxes[keep]
    scores = scores[keep]
    return {
        "rboxes": rboxes,
        "bbox_cxcywha": rboxes,
        "boxes": rboxes_to_enclosing_xyxy(rboxes),
        "scores": scores,
        "labels": torch.ones((rboxes.shape[0],), dtype=torch.long, device=rboxes.device),
        "angles": rboxes[:, 4] if rboxes.numel() > 0 else rboxes.new_zeros((0,)),
    }


class ConvTower(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=256, num_convs=4):
        super().__init__()
        layers = []
        for idx in range(num_convs):
            channels = in_channels if idx == 0 else hidden_channels
            layers.extend([
                nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_channels),
                nn.ReLU(inplace=True),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RotatedFCOS(nn.Module):
    def __init__(
        self,
        num_classes=1,
        image_size=DEFAULT_IMAGE_SIZE,
        pretrained_backbone_path="./weight/resnet50-0676ba61.pth",
        trainable_backbone_layers=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        bbox_loss_weight=1.0,
        centerness_loss_weight=1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = tuple(image_size)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.detections_per_img = int(detections_per_img)
        self.bbox_loss_weight = float(bbox_loss_weight)
        self.centerness_loss_weight = float(centerness_loss_weight)
        self.backbone = build_backbone(self.image_size, pretrained_backbone_path, trainable_backbone_layers)
        self.cls_tower = ConvTower()
        self.reg_tower = ConvTower()
        self.cls_logits = nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(256, 5, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.cls_logits.bias, -4.595)

    def forward_features(self, images):
        tensors = stack_images(images)
        return feature_list_from_backbone(self.backbone(tensors), max_levels=4)

    def head_forward(self, features):
        cls_outputs = []
        reg_outputs = []
        centerness_outputs = []
        for feature in features:
            cls_feat = self.cls_tower(feature)
            reg_feat = self.reg_tower(feature)
            cls_outputs.append(self.cls_logits(cls_feat))
            reg_outputs.append(self.bbox_pred(reg_feat))
            centerness_outputs.append(self.centerness(reg_feat))
        return cls_outputs, reg_outputs, centerness_outputs

    def decode_points(self, points, strides, deltas):
        dx, dy, dw, dh, da = deltas.unbind(dim=1)
        stride = strides.to(deltas.dtype)
        cx = points[:, 0] + dx * stride
        cy = points[:, 1] + dy * stride
        w = torch.exp(dw.clamp(max=4.135)) * stride
        h = torch.exp(dh.clamp(max=4.135)) * stride
        return torch.stack((cx, cy, w, h, da), dim=1)

    def encode_points(self, points, strides, target_rboxes):
        stride = strides.to(target_rboxes.dtype)
        cx, cy, w, h, angle = target_rboxes.unbind(dim=1)
        dx = (cx - points[:, 0]) / stride
        dy = (cy - points[:, 1]) / stride
        dw = torch.log(w.clamp(min=1e-6) / stride)
        dh = torch.log(h.clamp(min=1e-6) / stride)
        return torch.stack((dx, dy, dw, dh, angle), dim=1)

    def assign_targets(self, points, strides, level_ids, target):
        boxes = target["boxes"]
        rboxes = target["rboxes"]
        labels = points.new_zeros((points.shape[0],))
        bbox_targets = points.new_zeros((points.shape[0], 5))
        centerness_targets = points.new_zeros((points.shape[0],))
        if boxes.numel() == 0:
            return labels, bbox_targets, centerness_targets
        xs, ys = points[:, 0], points[:, 1]
        l = xs[:, None] - boxes[None, :, 0]
        t = ys[:, None] - boxes[None, :, 1]
        r = boxes[None, :, 2] - xs[:, None]
        b = boxes[None, :, 3] - ys[:, None]
        ltrb = torch.stack((l, t, r, b), dim=2)
        inside = ltrb.min(dim=2).values > 0
        max_regress = ltrb.max(dim=2).values
        ranges = points.new_tensor(FCOS_REGRESS_RANGES)[level_ids]
        in_level = (max_regress >= ranges[:, None, 0]) & (max_regress <= ranges[:, None, 1])
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[None, :].repeat(points.shape[0], 1)
        areas[~(inside & in_level)] = float("inf")
        matched_area, matched_idx = areas.min(dim=1)
        pos = torch.isfinite(matched_area)
        labels[pos] = 1.0
        if pos.any():
            bbox_targets[pos] = self.encode_points(points[pos], strides[pos], rboxes[matched_idx[pos]])
            matched_ltrb = ltrb[pos, matched_idx[pos]]
            lr = matched_ltrb[:, [0, 2]]
            tb = matched_ltrb[:, [1, 3]]
            centerness_targets[pos] = torch.sqrt(
                (lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-6))
                * (tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-6))
            )
        return labels, bbox_targets, centerness_targets

    def compute_losses(self, cls_outputs, reg_outputs, centerness_outputs, features, targets):
        batch_size = cls_outputs[0].shape[0]
        all_points = []
        all_strides = []
        all_level_ids = []
        cls_flat = []
        reg_flat = []
        cent_flat = []
        for level, (cls_out, reg_out, cent_out, feature, stride) in enumerate(zip(cls_outputs, reg_outputs, centerness_outputs, features, FCOS_STRIDES)):
            points = make_grid_points(feature, stride)
            num_points = points.shape[0]
            all_points.append(points)
            all_strides.append(points.new_full((num_points,), float(stride)))
            all_level_ids.append(torch.full((num_points,), level, dtype=torch.long, device=points.device))
            cls_flat.append(flatten_head_output(cls_out, self.num_classes))
            reg_flat.append(flatten_head_output(reg_out, 5))
            cent_flat.append(flatten_head_output(cent_out, 1))
        points = torch.cat(all_points, dim=0)
        strides = torch.cat(all_strides, dim=0)
        level_ids = torch.cat(all_level_ids, dim=0)
        cls_pred = torch.cat(cls_flat, dim=1).reshape(-1)
        reg_pred = torch.cat(reg_flat, dim=1).reshape(-1, 5)
        cent_pred = torch.cat(cent_flat, dim=1).reshape(-1)
        labels = []
        bbox_targets = []
        cent_targets = []
        for target in targets:
            label, bbox_target, cent_target = self.assign_targets(points, strides, level_ids, target)
            labels.append(label)
            bbox_targets.append(bbox_target)
            cent_targets.append(cent_target)
        labels = torch.cat(labels, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cent_targets = torch.cat(cent_targets, dim=0)
        normalizer = labels.sum().clamp(min=1.0)
        loss_cls = sigmoid_focal_loss(cls_pred, labels, reduction="sum") / normalizer
        pos = labels > 0
        if pos.any():
            loss_bbox = F.smooth_l1_loss(reg_pred[pos], bbox_targets[pos], beta=1.0 / 9.0, reduction="sum") / normalizer
            loss_centerness = F.binary_cross_entropy_with_logits(cent_pred[pos], cent_targets[pos], reduction="sum") / normalizer
        else:
            loss_bbox = reg_pred.sum() * 0.0
            loss_centerness = cent_pred.sum() * 0.0
        return {
            "loss_cls": loss_cls,
            "loss_rbox_reg": loss_bbox * self.bbox_loss_weight,
            "loss_centerness": loss_centerness * self.centerness_loss_weight,
        }

    def inference(self, cls_outputs, reg_outputs, centerness_outputs, features):
        batch_size = cls_outputs[0].shape[0]
        image_shape = (self.image_size[1], self.image_size[0])
        results = []
        per_image_rboxes = [[] for _ in range(batch_size)]
        per_image_scores = [[] for _ in range(batch_size)]
        for cls_out, reg_out, cent_out, feature, stride in zip(cls_outputs, reg_outputs, centerness_outputs, features, FCOS_STRIDES):
            points = make_grid_points(feature, stride)
            strides = points.new_full((points.shape[0],), float(stride))
            cls_score = torch.sigmoid(flatten_head_output(cls_out, self.num_classes).squeeze(-1))
            cent_score = torch.sigmoid(flatten_head_output(cent_out, 1).squeeze(-1))
            reg_pred = flatten_head_output(reg_out, 5)
            for img_id in range(batch_size):
                score = torch.sqrt(cls_score[img_id] * cent_score[img_id])
                rboxes = self.decode_points(points, strides, reg_pred[img_id])
                per_image_rboxes[img_id].append(rboxes)
                per_image_scores[img_id].append(score)
        for img_id in range(batch_size):
            rboxes = torch.cat(per_image_rboxes[img_id], dim=0)
            scores = torch.cat(per_image_scores[img_id], dim=0)
            results.append(postprocess_rboxes(rboxes, scores, image_shape, self.score_thresh, self.nms_thresh, self.detections_per_img))
        return results

    def forward(self, images, targets=None):
        features = self.forward_features(images)
        cls_outputs, reg_outputs, centerness_outputs = self.head_forward(features)
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None in training mode")
            return self.compute_losses(cls_outputs, reg_outputs, centerness_outputs, features, targets)
        return self.inference(cls_outputs, reg_outputs, centerness_outputs, features)


class RotatedSSD(nn.Module):
    def __init__(
        self,
        num_classes=1,
        image_size=DEFAULT_IMAGE_SIZE,
        pretrained_backbone_path="./weight/resnet50-0676ba61.pth",
        trainable_backbone_layers=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        neg_pos_ratio=3,
        bbox_loss_weight=1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = tuple(image_size)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.detections_per_img = int(detections_per_img)
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.bbox_loss_weight = float(bbox_loss_weight)
        self.backbone = build_backbone(self.image_size, pretrained_backbone_path, trainable_backbone_layers)
        num_anchors = len(ANCHOR_ASPECT_RATIOS[0])
        self.cls_heads = nn.ModuleList([nn.Conv2d(256, num_anchors * 2, kernel_size=3, padding=1) for _ in FCOS_STRIDES])
        self.reg_heads = nn.ModuleList([nn.Conv2d(256, num_anchors * 5, kernel_size=3, padding=1) for _ in FCOS_STRIDES])

    def forward_features(self, images):
        tensors = stack_images(images)
        return feature_list_from_backbone(self.backbone(tensors), max_levels=4)

    def head_forward(self, features):
        cls_outputs = []
        reg_outputs = []
        for feature, cls_head, reg_head in zip(features, self.cls_heads, self.reg_heads):
            cls_outputs.append(cls_head(feature))
            reg_outputs.append(reg_head(feature))
        return cls_outputs, reg_outputs

    def make_all_anchors(self, features):
        anchors = []
        ratios = ANCHOR_ASPECT_RATIOS[0]
        for feature, stride in zip(features, FCOS_STRIDES):
            anchors.append(make_anchors(feature, stride, self.image_size, ratios))
        return torch.cat(anchors, dim=0)

    def flatten_outputs(self, cls_outputs, reg_outputs):
        cls_flat = []
        reg_flat = []
        for cls_out, reg_out in zip(cls_outputs, reg_outputs):
            n = cls_out.shape[0]
            cls_flat.append(cls_out.permute(0, 2, 3, 1).reshape(n, -1, 2))
            reg_flat.append(reg_out.permute(0, 2, 3, 1).reshape(n, -1, 5))
        return torch.cat(cls_flat, dim=1), torch.cat(reg_flat, dim=1)

    def assign_targets(self, anchors, target):
        boxes = target["boxes"]
        rboxes = target["rboxes"]
        labels = torch.zeros((anchors.shape[0],), dtype=torch.long, device=anchors.device)
        bbox_targets = anchors.new_zeros((anchors.shape[0], 5))
        if boxes.numel() == 0:
            return labels, bbox_targets
        anchor_xyxy = rboxes_to_enclosing_xyxy(anchors)
        ious = xyxy_iou(anchor_xyxy, boxes)
        max_iou, matched_idx = ious.max(dim=1)
        labels[max_iou >= 0.5] = 1
        best_anchor_per_gt = ious.argmax(dim=0)
        labels[best_anchor_per_gt] = 1
        matched_idx[best_anchor_per_gt] = torch.arange(boxes.shape[0], device=anchors.device)
        pos = labels > 0
        if pos.any():
            bbox_targets[pos] = encode_rboxes(anchors[pos], rboxes[matched_idx[pos]])
        return labels, bbox_targets

    def compute_losses(self, cls_outputs, reg_outputs, features, targets):
        anchors = self.make_all_anchors(features)
        cls_pred, reg_pred = self.flatten_outputs(cls_outputs, reg_outputs)
        all_labels = []
        all_bbox_targets = []
        for target in targets:
            labels, bbox_targets = self.assign_targets(anchors, target)
            all_labels.append(labels)
            all_bbox_targets.append(bbox_targets)
        labels = torch.stack(all_labels, dim=0)
        bbox_targets = torch.stack(all_bbox_targets, dim=0)
        cls_loss_all = F.cross_entropy(cls_pred.reshape(-1, 2), labels.reshape(-1), reduction="none").reshape_as(labels)
        pos = labels > 0
        num_pos = pos.sum().clamp(min=1)
        neg = labels == 0
        neg_loss = cls_loss_all.clone()
        neg_loss[~neg] = -1.0
        num_neg = torch.clamp(pos.sum(dim=1) * self.neg_pos_ratio, min=1, max=neg.shape[1]).to(torch.long)
        keep_neg = torch.zeros_like(neg)
        for img_id in range(labels.shape[0]):
            _, idx = neg_loss[img_id].sort(descending=True)
            keep_neg[img_id, idx[: num_neg[img_id]]] = True
        cls_loss = cls_loss_all[pos | keep_neg].sum() / num_pos
        if pos.any():
            reg_loss = F.smooth_l1_loss(reg_pred[pos], bbox_targets[pos], beta=1.0 / 9.0, reduction="sum") / num_pos
        else:
            reg_loss = reg_pred.sum() * 0.0
        return {"loss_cls": cls_loss, "loss_rbox_reg": reg_loss * self.bbox_loss_weight}

    def inference(self, cls_outputs, reg_outputs, features):
        anchors = self.make_all_anchors(features)
        cls_pred, reg_pred = self.flatten_outputs(cls_outputs, reg_outputs)
        probs = F.softmax(cls_pred, dim=-1)[..., 1]
        image_shape = (self.image_size[1], self.image_size[0])
        results = []
        for img_id in range(cls_pred.shape[0]):
            rboxes = decode_rboxes(anchors, reg_pred[img_id])
            scores = probs[img_id]
            results.append(postprocess_rboxes(rboxes, scores, image_shape, self.score_thresh, self.nms_thresh, self.detections_per_img))
        return results

    def forward(self, images, targets=None):
        features = self.forward_features(images)
        cls_outputs, reg_outputs = self.head_forward(features)
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None in training mode")
            return self.compute_losses(cls_outputs, reg_outputs, features, targets)
        return self.inference(cls_outputs, reg_outputs, features)


def build_rotated_fcos(**kwargs):
    return RotatedFCOS(**kwargs)


def build_rotated_ssd(**kwargs):
    return RotatedSSD(**kwargs)