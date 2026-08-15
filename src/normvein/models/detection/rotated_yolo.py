from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotated_faster_rcnn import (
    DEFAULT_IMAGE_SIZE,
    build_torchvision_faster_rcnn,
    load_local_resnet50_weights,
    patch_first_conv_to_single_channel,
    normalize_angle,
    clip_rboxes_to_image,
    rboxes_to_enclosing_xyxy,
    rotated_nms,
)


YOLO_STRIDES = (4, 8, 16, 32)
YOLO_SIZE_THRESHOLDS = (64.0, 128.0, 192.0)


def build_backbone(image_size=DEFAULT_IMAGE_SIZE, pretrained_backbone_path="./weight/resnet50-0676ba61.pth", trainable_backbone_layers=None):
    detector = build_torchvision_faster_rcnn(
        num_classes=2,
        image_size=image_size,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    load_local_resnet50_weights(detector, pretrained_backbone_path)
    patch_first_conv_to_single_channel(detector)
    return detector.backbone


def feature_list_from_backbone(features):
    if isinstance(features, torch.Tensor):
        return [features]
    if isinstance(features, OrderedDict):
        values = [features[key] for key in sorted(features.keys())]
    else:
        values = list(features)
    return values[:len(YOLO_STRIDES)]


def stack_images(images):
    return torch.stack(images, dim=0)


def make_grid_points(feature, stride):
    h, w = feature.shape[-2:]
    ys = (torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5) * float(stride)
    xs = (torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5) * float(stride)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def flatten_output(tensor, channels):
    n = tensor.shape[0]
    return tensor.permute(0, 2, 3, 1).reshape(n, -1, channels)


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


def postprocess_rboxes(rboxes, scores, image_shape, score_thresh, nms_thresh, detections_per_img, pre_nms_topk):
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
    valid = (rboxes[:, 2] > 1e-2) & (rboxes[:, 3] > 1e-2)
    valid = valid & (scores > float(score_thresh))
    if not valid.any():
        valid = scores == scores.max()
    rboxes = rboxes[valid]
    scores = scores[valid]

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

    if scores.numel() > int(pre_nms_topk):
        scores, order = scores.topk(int(pre_nms_topk), sorted=True)
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


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class YOLODecoupledHead(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=256, num_classes=1):
        super().__init__()
        self.reg_tower = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )
        self.cls_tower = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels),
            ConvBNAct(hidden_channels, hidden_channels),
        )
        self.reg_pred = nn.Conv2d(hidden_channels, 5, kernel_size=1)
        self.obj_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        nn.init.constant_(self.obj_pred.bias, -4.595)
        nn.init.constant_(self.cls_pred.bias, -4.595)

    def forward(self, x):
        reg_feat = self.reg_tower(x)
        cls_feat = self.cls_tower(x)
        return self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)


class RotatedYOLO(nn.Module):
    def __init__(
        self,
        num_classes=1,
        image_size=DEFAULT_IMAGE_SIZE,
        pretrained_backbone_path="./weight/resnet50-0676ba61.pth",
        trainable_backbone_layers=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
        pre_nms_topk=1000,
        bbox_loss_weight=1.0,
        obj_loss_weight=1.0,
        cls_loss_weight=0.5,
        center_radius=1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = tuple(image_size)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.detections_per_img = int(detections_per_img)
        self.pre_nms_topk = int(pre_nms_topk)
        self.bbox_loss_weight = float(bbox_loss_weight)
        self.obj_loss_weight = float(obj_loss_weight)
        self.cls_loss_weight = float(cls_loss_weight)
        self.center_radius = int(center_radius)
        self.backbone = build_backbone(self.image_size, pretrained_backbone_path, trainable_backbone_layers)
        self.heads = nn.ModuleList([YOLODecoupledHead(num_classes=self.num_classes) for _ in YOLO_STRIDES])

    def forward_features(self, images):
        tensors = stack_images(images)
        return feature_list_from_backbone(self.backbone(tensors))

    def head_forward(self, features):
        reg_outputs = []
        obj_outputs = []
        cls_outputs = []
        for feature, head in zip(features, self.heads):
            reg, obj, cls = head(feature)
            reg_outputs.append(reg)
            obj_outputs.append(obj)
            cls_outputs.append(cls)
        return reg_outputs, obj_outputs, cls_outputs

    def encode_points(self, points, stride, target_rboxes):
        cx, cy, w, h, angle = target_rboxes.unbind(dim=1)
        stride_value = float(stride)
        dx = (cx - points[:, 0]) / stride_value
        dy = (cy - points[:, 1]) / stride_value
        dw = torch.log(w.clamp(min=1e-6) / stride_value)
        dh = torch.log(h.clamp(min=1e-6) / stride_value)
        return torch.stack((dx, dy, dw, dh, angle), dim=1)

    def decode_points(self, points, stride, deltas):
        dx, dy, dw, dh, angle = deltas.unbind(dim=1)
        stride_value = float(stride)
        cx = points[:, 0] + dx * stride_value
        cy = points[:, 1] + dy * stride_value
        w = torch.exp(dw.clamp(max=4.135)) * stride_value
        h = torch.exp(dh.clamp(max=4.135)) * stride_value
        return torch.stack((cx, cy, w, h, normalize_angle(angle)), dim=1)

    def choose_level(self, rbox):
        max_side = float(torch.maximum(rbox[2], rbox[3]).detach().cpu())
        if max_side < YOLO_SIZE_THRESHOLDS[0]:
            return 0
        if max_side < YOLO_SIZE_THRESHOLDS[1]:
            return 1
        if max_side < YOLO_SIZE_THRESHOLDS[2]:
            return 2
        return 3

    def assign_single_level(self, feature, stride, target, level_id):
        _, _, h, w = feature.shape
        device = feature.device
        obj_target = feature.new_zeros((h, w))
        cls_target = feature.new_zeros((h, w, self.num_classes))
        reg_target = feature.new_zeros((h, w, 5))
        pos_mask = torch.zeros((h, w), dtype=torch.bool, device=device)
        if target["rboxes"].numel() == 0:
            return obj_target, cls_target, reg_target, pos_mask

        for rbox in target["rboxes"]:
            if self.choose_level(rbox) != level_id:
                continue
            gx = float(rbox[0] / float(stride) - 0.5)
            gy = float(rbox[1] / float(stride) - 0.5)
            cx = int(round(gx))
            cy = int(round(gy))
            for oy in range(-self.center_radius, self.center_radius + 1):
                for ox in range(-self.center_radius, self.center_radius + 1):
                    ix = cx + ox
                    iy = cy + oy
                    if ix < 0 or iy < 0 or ix >= w or iy >= h:
                        continue
                    point = rbox.new_tensor([[(ix + 0.5) * float(stride), (iy + 0.5) * float(stride)]])
                    obj_target[iy, ix] = 1.0
                    cls_target[iy, ix, 0] = 1.0
                    reg_target[iy, ix] = self.encode_points(point, stride, rbox.reshape(1, 5))[0]
                    pos_mask[iy, ix] = True
        return obj_target, cls_target, reg_target, pos_mask

    def compute_losses(self, reg_outputs, obj_outputs, cls_outputs, features, targets):
        obj_losses = []
        cls_losses = []
        reg_losses = []
        total_pos = 0
        for img_id, target in enumerate(targets):
            for level_id, (reg_out, obj_out, cls_out, feature, stride) in enumerate(zip(reg_outputs, obj_outputs, cls_outputs, features, YOLO_STRIDES)):
                obj_target, cls_target, reg_target, pos_mask = self.assign_single_level(feature, stride, target, level_id)
                obj_logit = obj_out[img_id, 0]
                cls_logit = cls_out[img_id].permute(1, 2, 0)
                reg_pred = reg_out[img_id].permute(1, 2, 0)
                obj_losses.append(sigmoid_focal_loss(obj_logit, obj_target, reduction="sum"))
                num_pos = int(pos_mask.sum().item())
                total_pos += num_pos
                if num_pos > 0:
                    cls_losses.append(F.binary_cross_entropy_with_logits(cls_logit[pos_mask], cls_target[pos_mask], reduction="sum"))
                    reg_losses.append(F.smooth_l1_loss(reg_pred[pos_mask], reg_target[pos_mask], beta=1.0 / 9.0, reduction="sum"))
                else:
                    cls_losses.append(cls_logit.sum() * 0.0)
                    reg_losses.append(reg_pred.sum() * 0.0)

        normalizer = max(float(total_pos), 1.0)
        loss_obj = torch.stack(obj_losses).sum() / normalizer
        loss_cls = torch.stack(cls_losses).sum() / normalizer
        loss_reg = torch.stack(reg_losses).sum() / normalizer
        return {
            "loss_obj": loss_obj * self.obj_loss_weight,
            "loss_cls": loss_cls * self.cls_loss_weight,
            "loss_rbox_reg": loss_reg * self.bbox_loss_weight,
        }

    def inference(self, reg_outputs, obj_outputs, cls_outputs, features):
        batch_size = reg_outputs[0].shape[0]
        image_shape = (self.image_size[1], self.image_size[0])
        per_image_rboxes = [[] for _ in range(batch_size)]
        per_image_scores = [[] for _ in range(batch_size)]

        for reg_out, obj_out, cls_out, feature, stride in zip(reg_outputs, obj_outputs, cls_outputs, features, YOLO_STRIDES):
            points = make_grid_points(feature, stride).reshape(-1, 2)
            reg_pred = flatten_output(reg_out, 5)
            obj_score = torch.sigmoid(flatten_output(obj_out, 1).squeeze(-1))
            cls_score = torch.sigmoid(flatten_output(cls_out, self.num_classes).amax(dim=-1))
            score = obj_score * cls_score
            for img_id in range(batch_size):
                rboxes = self.decode_points(points, stride, reg_pred[img_id])
                per_image_rboxes[img_id].append(rboxes)
                per_image_scores[img_id].append(score[img_id])

        results = []
        for img_id in range(batch_size):
            rboxes = torch.cat(per_image_rboxes[img_id], dim=0)
            scores = torch.cat(per_image_scores[img_id], dim=0)
            results.append(
                postprocess_rboxes(
                    rboxes,
                    scores,
                    image_shape,
                    self.score_thresh,
                    self.nms_thresh,
                    self.detections_per_img,
                    self.pre_nms_topk,
                )
            )
        return results

    def forward(self, images, targets=None):
        features = self.forward_features(images)
        reg_outputs, obj_outputs, cls_outputs = self.head_forward(features)
        if self.training:
            if targets is None:
                raise ValueError("targets should not be None in training mode")
            return self.compute_losses(reg_outputs, obj_outputs, cls_outputs, features, targets)
        return self.inference(reg_outputs, obj_outputs, cls_outputs, features)


def build_rotated_yolo(**kwargs):
    return RotatedYOLO(**kwargs)
