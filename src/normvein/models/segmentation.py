import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def percentile_normalize_l_image(image, low_pct=1.0, high_pct=99.0):
    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, [low_pct, high_pct])
    if high <= low:
        return image
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)



def build_mask_index(mask_dir):
    mask_dir = Path(mask_dir)
    by_relative_stem = {}
    by_name_stem = {}
    for path in list_images(mask_dir):
        relative_stem = path.relative_to(mask_dir).with_suffix("").as_posix()
        by_relative_stem[relative_stem] = path
        by_name_stem.setdefault(path.stem, path)
    return by_relative_stem, by_name_stem


class FingerVeinSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=(256, 512),
        train=True,
        mask_threshold=127,
        percentile_normalize=True,
        percentile_low=1.0,
        percentile_high=99.0,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = tuple(image_size)
        self.train = train
        self.mask_threshold = mask_threshold
        self.percentile_normalize = bool(percentile_normalize)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        image_paths = list_images(self.image_dir)
        relative_mask_index, name_mask_index = build_mask_index(self.mask_dir)
        self.samples = []
        missing = []
        for image_path in image_paths:
            relative_stem = image_path.relative_to(self.image_dir).with_suffix("").as_posix()
            mask_path = relative_mask_index.get(relative_stem)
            if mask_path is None:
                mask_path = name_mask_index.get(image_path.stem)
            if mask_path is None:
                missing.append(str(image_path))
                continue
            self.samples.append((image_path, mask_path))

        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"Found {len(missing)} images without matching masks. First missing images:\n{preview}"
            )
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found in {self.image_dir} and {self.mask_dir}")

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        height, width = self.image_size
        image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [height, width], interpolation=InterpolationMode.NEAREST)
        if self.percentile_normalize:
            image = percentile_normalize_l_image(image, self.percentile_low, self.percentile_high)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask * 255.0 > self.mask_threshold).float()
        return image, mask



class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, subset_ratio=0.1, seed=2048):
        self.data_source = data_source
        self.subset_ratio = min(max(float(subset_ratio), 0.0), 1.0)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return max(1, int(math.ceil(len(self.data_source) * self.subset_ratio)))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter(indices[: len(self)])


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=None, dilation=1):
        if padding is None:
            padding = dilation if kernel_size == 3 else 0
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBNReLU(in_channels, out_channels, kernel_size=3),
            ConvBNReLU(out_channels, out_channels, kernel_size=3),
        )


def adapt_first_conv_to_grayscale(model, state_dict):
    model_state = model.state_dict()
    for key, value in list(state_dict.items()):
        target = model_state.get(key)
        if target is None or not torch.is_tensor(value):
            continue
        if value.ndim == 4 and target.ndim == 4 and value.shape[1] == 3 and target.shape[1] == 1:
            if value.shape[0] == target.shape[0] and value.shape[2:] == target.shape[2:]:
                state_dict[key] = value.mean(dim=1, keepdim=True)
                print(f"[info] Adapted pretrained RGB conv to grayscale: {key}")
    return state_dict


def load_local_torchvision_weights(model, pretrained_dir, filenames, model_label):
    pretrained_dir = Path(pretrained_dir)
    candidates = [pretrained_dir / name for name in filenames]
    for path in candidates:
        if path.is_file():
            state_dict = torch.load(str(path), map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if isinstance(state_dict, dict):
                state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
            state_dict = adapt_first_conv_to_grayscale(model, state_dict)
            model.load_state_dict(state_dict, strict=True)
            print(f"[info] Loaded {model_label} pretrained weights: {path}")
            return model
    expected = ", ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(
        f"Missing local {model_label} pretrained weights. Expected one of: {expected}. "
        "Download the file first, pass --allow-weight-download, or use --encoder-weights none."
    )


def safe_resnet50(weights_mode, pretrained_dir, allow_weight_download=False, in_channels=1):
    def replace_input_conv(model):
        if in_channels == 3:
            return model
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if old.weight.shape[1] == 3 and in_channels == 1:
            model.conv1.weight.data.copy_(old.weight.data.mean(dim=1, keepdim=True))
        return model

    if weights_mode == "none":
        return replace_input_conv(models.resnet50(weights=None))
    model = replace_input_conv(models.resnet50(weights=None))
    try:
        return load_local_torchvision_weights(
            model,
            pretrained_dir,
            ("resnet50-0676ba61", "resnet50-0676ba61.pth", "resnet50-11ad3fa6", "resnet50-11ad3fa6.pth"),
            "ResNet50",
        )
    except FileNotFoundError:
        if not allow_weight_download:
            raise
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    return replace_input_conv(model)


def safe_resnet34(weights_mode, pretrained_dir, allow_weight_download=False, in_channels=1):
    def replace_input_conv(model):
        if in_channels == 3:
            return model
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if old.weight.shape[1] == 3 and in_channels == 1:
            model.conv1.weight.data.copy_(old.weight.data.mean(dim=1, keepdim=True))
        return model

    if weights_mode == "none":
        return replace_input_conv(models.resnet34(weights=None))
    model = replace_input_conv(models.resnet34(weights=None))
    try:
        return load_local_torchvision_weights(
            model,
            pretrained_dir,
            ("resnet34-b627a593", "resnet34-b627a593.pth"),
            "ResNet34",
        )
    except FileNotFoundError:
        if not allow_weight_download:
            raise
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    return replace_input_conv(model)


def replace_first_conv(module, in_channels):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=1,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            if child.weight.shape[1] == 3 and in_channels == 1:
                new_conv.weight.data.copy_(child.weight.data.mean(dim=1, keepdim=True))
            if child.bias is not None:
                new_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, new_conv)
            return True
        if replace_first_conv(child, in_channels):
            return True
    return False


def safe_mobilenet_v3_large(weights_mode, pretrained_dir, dilated=True, allow_weight_download=False, in_channels=1):
    kwargs = {"dilated": dilated}
    if weights_mode == "none":
        model = models.mobilenet_v3_large(weights=None, **kwargs)
        if in_channels != 3:
            replace_first_conv(model, in_channels)
        return model
    model = models.mobilenet_v3_large(weights=None, **kwargs)
    if in_channels != 3:
        replace_first_conv(model, in_channels)
    try:
        return load_local_torchvision_weights(
            model,
            pretrained_dir,
            ("mobilenet_v3_large-5c1a4163", "mobilenet_v3_large-5c1a4163.pth"),
            "MobileNetV3-Large",
        )
    except FileNotFoundError:
        if not allow_weight_download:
            raise
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, **kwargs)
    if in_channels != 3:
        replace_first_conv(model, in_channels)
    return model


class ResNet50Encoder(nn.Module):
    def __init__(self, weights_mode="imagenet", pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
        super().__init__()
        resnet = safe_resnet50(weights_mode, pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class ResNet34Encoder(nn.Module):
    def __init__(self, weights_mode="imagenet", pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
        super().__init__()
        resnet = safe_resnet34(weights_mode, pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class UNetResNet50(nn.Module):
    def __init__(self, num_classes=1, weights_mode="imagenet", decoder_channels=(512, 256, 128, 64), pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
        super().__init__()
        self.encoder = ResNet50Encoder(weights_mode, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
        d3, d2, d1, d0 = decoder_channels
        self.decoder3 = DecoderBlock(2048 + 1024, d3)
        self.decoder2 = DecoderBlock(d3 + 512, d2)
        self.decoder1 = DecoderBlock(d2 + 256, d1)
        self.decoder0 = DecoderBlock(d1 + 64, d0)
        self.segmentation_head = nn.Conv2d(d0, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)
        d3 = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat([d3, x3], dim=1))
        d2 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([d2, x2], dim=1))
        d1 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x1], dim=1))
        d0 = F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.decoder0(torch.cat([d0, x0], dim=1))
        logits = self.segmentation_head(d0)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class UNetResNet34(nn.Module):
    def __init__(self, num_classes=1, weights_mode="imagenet", decoder_channels=(256, 128, 64, 32), pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
        super().__init__()
        self.encoder = ResNet34Encoder(weights_mode, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
        d3, d2, d1, d0 = decoder_channels
        self.decoder3 = DecoderBlock(512 + 256, d3)
        self.decoder2 = DecoderBlock(d3 + 128, d2)
        self.decoder1 = DecoderBlock(d2 + 64, d1)
        self.decoder0 = DecoderBlock(d1 + 64, d0)
        self.segmentation_head = nn.Conv2d(d0, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)
        d3 = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.decoder3(torch.cat([d3, x3], dim=1))
        d2 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.decoder2(torch.cat([d2, x2], dim=1))
        d1 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.decoder1(torch.cat([d1, x1], dim=1))
        d0 = F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.decoder0(torch.cat([d0, x0], dim=1))
        logits = self.segmentation_head(d0)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class UNetPlusPlusResNet50(nn.Module):
    def __init__(self, num_classes=1, weights_mode="imagenet", decoder_channels=(64, 128, 256, 512), pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
        super().__init__()
        self.encoder = ResNet50Encoder(weights_mode, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
        encoder_channels = [64, 256, 512, 1024, 2048]
        decoder_channels = list(decoder_channels)
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four values.")

        self.blocks = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                skip_channels = encoder_channels[i] + max(0, j - 1) * decoder_channels[i]
                up_channels = encoder_channels[i + 1] if j == 1 else decoder_channels[i + 1]
                self.blocks[f"x{i}_{j}"] = DecoderBlock(skip_channels + up_channels, decoder_channels[i])
        self.segmentation_head = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        nodes = {}
        for i, feature in enumerate(self.encoder(x)):
            nodes[(i, 0)] = feature

        for j in range(1, 5):
            for i in range(0, 5 - j):
                same_level = [nodes[(i, k)] for k in range(0, j)]
                up = F.interpolate(
                    nodes[(i + 1, j - 1)],
                    size=nodes[(i, 0)].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                node = torch.cat(same_level + [up], dim=1)
                nodes[(i, j)] = self.blocks[f"x{i}_{j}"](node)

        logits = self.segmentation_head(nodes[(0, 4)])
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


try:
    from torchvision.models.segmentation.deeplabv3 import ASPP
except Exception:
    ASPP = None


class SimpleASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = [ConvBNReLU(in_channels, out_channels, kernel_size=1)]
        modules.extend(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=rate)
            for rate in atrous_rates
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            ConvBNReLU(out_channels * len(modules), out_channels, kernel_size=1),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))


class MobileNetV3LargeEncoder(nn.Module):
    def __init__(self, weights_mode="imagenet", pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
        super().__init__()
        backbone = safe_mobilenet_v3_large(weights_mode, pretrained_dir, dilated=True, allow_weight_download=allow_weight_download, in_channels=in_channels).features
        stage_indices = [0] + [i for i, block in enumerate(backbone) if getattr(block, "_is_cn", False)]
        stage_indices.append(len(backbone) - 1)
        self.low_index = stage_indices[-4]
        self.out_index = stage_indices[-1]
        self.features = backbone

        with torch.no_grad():
            was_training = self.features.training
            self.features.eval()
            low, out = self.forward(torch.zeros(1, in_channels, 224, 224))
            self.low_channels = low.shape[1]
            self.out_channels = out.shape[1]
            self.features.train(was_training)

    def forward(self, x):
        low = None
        out = None
        for index, block in enumerate(self.features):
            x = block(x)
            if index == self.low_index:
                low = x
            if index == self.out_index:
                out = x
        return low, out


class DeepLabV3PlusMobileNetV3(nn.Module):
    def __init__(self, num_classes=1, weights_mode="imagenet", aspp_channels=256, low_channels=48, pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
        super().__init__()
        self.encoder = MobileNetV3LargeEncoder(weights_mode, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
        aspp_class = ASPP if ASPP is not None else SimpleASPP
        self.aspp = aspp_class(self.encoder.out_channels, [12, 24, 36], out_channels=aspp_channels)
        self.low_project = ConvBNReLU(self.encoder.low_channels, low_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_channels + low_channels, 256, kernel_size=3),
            ConvBNReLU(256, 256, kernel_size=3),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        low, out = self.encoder(x)
        out = self.aspp(out)
        out = F.interpolate(out, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_project(low)
        logits = self.decoder(torch.cat([out, low], dim=1))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def build_smp_model(model_name, encoder_weights, in_channels=1):
    try:
        import segmentation_models_pytorch as smp
    except Exception:
        return None

    weights = "imagenet" if encoder_weights == "imagenet" else None
    if model_name == "unet_resnet34":
        try:
            print("[info] Using segmentation_models_pytorch.Unet(resnet34).")
            return smp.Unet(
                encoder_name="resnet34",
                encoder_weights=weights,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        except Exception as exc:
            print(f"[warn] SMP Unet ResNet34 construction failed: {exc}")
            return None

    if model_name == "unetpp_resnet50":
        try:
            print("[info] Using segmentation_models_pytorch.UnetPlusPlus(resnet50).")
            return smp.UnetPlusPlus(
                encoder_name="resnet50",
                encoder_weights=weights,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        except Exception as exc:
            print(f"[warn] SMP UnetPlusPlus ResNet50 construction failed: {exc}")
            return None

    if model_name == "unet_resnet50":
        try:
            print("[info] Using segmentation_models_pytorch.Unet(resnet50).")
            return smp.Unet(
                encoder_name="resnet50",
                encoder_weights=weights,
                in_channels=in_channels,
                classes=1,
                activation=None,
            )
        except Exception as exc:
            print(f"[warn] SMP Unet ResNet50 construction failed: {exc}")
            return None

    if model_name == "deeplabv3plus_mobilenetv3":
        candidates = ["mobilenet_v3_large", "timm-mobilenetv3_large_100", "tu-mobilenetv3_large_100"]
        last_error = None
        for encoder_name in candidates:
            try:
                print(f"[info] Using segmentation_models_pytorch.DeepLabV3Plus({encoder_name}).")
                return smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=weights,
                    in_channels=in_channels,
                    classes=1,
                    activation=None,
                )
            except Exception as exc:
                last_error = exc
        print(f"[warn] SMP DeepLabV3Plus MobileNetV3 construction failed: {last_error}")
    return None


def build_model(model_name, encoder_weights, use_smp=False, pretrained_dir="./weights/backbones", allow_weight_download=False, in_channels=1):
    model = build_smp_model(model_name, encoder_weights, in_channels=in_channels) if use_smp and allow_weight_download else None
    if model is not None:
        return model
    if model_name == "unetpp_resnet50":
        return UNetPlusPlusResNet50(num_classes=1, weights_mode=encoder_weights, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
    if model_name == "unet_resnet50":
        return UNetResNet50(num_classes=1, weights_mode=encoder_weights, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
    if model_name == "unet_resnet34":
        return UNetResNet34(num_classes=1, weights_mode=encoder_weights, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
    if model_name == "deeplabv3plus_mobilenetv3":
        return DeepLabV3PlusMobileNetV3(num_classes=1, weights_mode=encoder_weights, pretrained_dir=pretrained_dir, allow_weight_download=allow_weight_download, in_channels=in_channels)
    raise ValueError(f"Unsupported model: {model_name}")



class UNetBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


@torch.no_grad()
def segmentation_metrics(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    pred_sum = torch.sum(preds, dims)
    target_sum = torch.sum(targets, dims)
    union = pred_sum + target_sum - intersection
    return {
        "dice": ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).mean().item(),
        "iou": ((intersection + eps) / (union + eps)).mean().item(),
        "precision": ((intersection + eps) / (pred_sum + eps)).mean().item(),
        "recall": ((intersection + eps) / (target_sum + eps)).mean().item(),
    }


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self):
        return self.total / max(1, self.count)


def make_scheduler(optimizer, total_steps, warmup_steps):
    total_steps = max(1, total_steps)
    warmup_steps = max(0, warmup_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, (1.0 - progress) ** 0.9)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, args):
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def make_epoch_checkpoint_path(run_dir, checkpoint_name, epoch):
    checkpoint_path = Path(checkpoint_name)
    suffix = checkpoint_path.suffix or ".pt"
    return Path(run_dir) / f"{checkpoint_path.stem}_epoch{epoch:03d}{suffix}"


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(str(path), map_location=device)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0)) + 1


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, args):
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    model.train()
    loss_meter = AverageMeter()
    metric_meters = {name: AverageMeter() for name in ["dice", "iou", "precision", "recall"]}
    start = time.time()

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
        scheduler.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        for name, value in segmentation_metrics(logits.detach(), masks, threshold=args.threshold).items():
            metric_meters[name].update(value, batch_size)

        if step % args.log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                f"epoch={epoch:03d} step={step:05d}/{len(loader):05d} "
                f"lr={lr:.3e} loss={loss_meter.avg:.4f} "
                f"dice={metric_meters['dice'].avg:.4f} iou={metric_meters['iou'].avg:.4f} "
                f"time={elapsed:.1f}s"
            )

    return {"loss": loss_meter.avg, **{name: meter.avg for name, meter in metric_meters.items()}}



def build_loaders(args):
    image_dir = Path(args.data_root) / args.images_subdir
    mask_dir = Path(args.data_root) / args.masks_subdir
    train_dataset = FingerVeinSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_size=args.image_size,
        train=True,
        mask_threshold=args.mask_threshold,
        percentile_normalize=args.percentile_normalize,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )

    train_sampler = RandomSubsetSampler(
        train_dataset,
        subset_ratio=args.train_subset_ratio,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, len(train_dataset)


def train_model(model_name, args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_dir = Path(args.output_dir) / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] model={model_name}")
    print(f"[info] data_root={args.data_root}")
    print(f"[info] output_dir={run_dir}")

    print(
        f"[info] percentile_normalize={args.percentile_normalize} "
        f"low={args.percentile_low} high={args.percentile_high}"
    )
    train_loader, dataset_size = build_loaders(args)
    print(f"[info] dataset_size={dataset_size} train={len(train_loader.dataset)} train_epoch_samples={len(train_loader.sampler)}")

    model = build_model(
        model_name,
        args.encoder_weights,
        use_smp=args.use_smp,
        pretrained_dir=args.pretrained_dir,
        allow_weight_download=args.allow_weight_download,
        in_channels=args.in_channels,
    ).to(device)
    criterion = UNetBCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = make_scheduler(optimizer, total_steps=total_steps, warmup_steps=int(total_steps * args.warmup_ratio))
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        print(f"[info] resumed from {args.resume}, start_epoch={start_epoch}")

    final_epoch = start_epoch - 1
    final_train_dice = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, args)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_stats['loss']:.4f} train_dice={train_stats['dice']:.4f} "
            f"train_iou={train_stats['iou']:.4f} train_precision={train_stats['precision']:.4f} "
            f"train_recall={train_stats['recall']:.4f}"
        )
        final_epoch = epoch
        final_train_dice = train_stats["dice"]
        epoch_path = make_epoch_checkpoint_path(run_dir, args.checkpoint_name, epoch)
        save_checkpoint(epoch_path, model, optimizer, scheduler, scaler, epoch, args)
        print(f"[info] saved epoch checkpoint: {epoch_path}")

    if final_epoch >= start_epoch:
        final_path = run_dir / args.checkpoint_name
        save_checkpoint(final_path, model, optimizer, scheduler, scaler, final_epoch, args)
        print(f"[info] saved latest checkpoint: {final_path}")
    return final_train_dice


def parse_args():
    parser = argparse.ArgumentParser(description="Finger vein texture segmentation training.")
    parser.add_argument(
        "--model",
        default="unetpp_resnet50",
        choices=["unetpp_resnet50", "unet_resnet50", "unet_resnet34", "deeplabv3plus_mobilenetv3", "all"],
        help="Train one model or train all registered models sequentially.",
    )
    parser.add_argument("--data-root", default="./data/FingerVeinSyn-5M")
    parser.add_argument("--images-subdir", default="Images")
    parser.add_argument("--masks-subdir", default="Masks")
    parser.add_argument("--output-dir", default="./runs/segmentation")
    parser.add_argument("--checkpoint-name", default="FingerPattern.pt")
    parser.add_argument("--image-size", nargs=2, type=int, default=[300, 600], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-subset-ratio", type=float, default=0.01, help="Random training subset ratio used every epoch.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=2048)
    parser.add_argument("--mask-threshold", type=int, default=127)

    parser.add_argument("--percentile-normalize", dest="percentile_normalize", action="store_true", default=True)
    parser.add_argument("--no-percentile-normalize", dest="percentile_normalize", action="store_false")
    parser.add_argument("--percentile-low", type=float, default=10.0)
    parser.add_argument("--percentile-high", type=float, default=90.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE in BCE+Dice loss. 0.5 means BCE and Dice are equally weighted.")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--pretrained-dir", default="./weights/backbones")
    parser.add_argument("--allow-weight-download", action="store_true", help="Allow torchvision/SMP to download weights if local files are missing.")
    parser.add_argument("--use-smp", action="store_true", help="Use segmentation_models_pytorch if installed. Requires --allow-weight-download for ImageNet weights.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training.")
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--resume", default="")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 <= args.percentile_low < args.percentile_high <= 100.0):
        raise ValueError("--percentile-low and --percentile-high must satisfy 0 <= low < high <= 100")
    if args.model == "all":
        results = {}
        for model_name in ["unetpp_resnet50", "unet_resnet50", "unet_resnet34", "deeplabv3plus_mobilenetv3"]:
            args.resume = ""
            results[model_name] = train_model(model_name, args)
        print("[info] finished all models")
        for model_name, final_train_dice in results.items():
            print(f"{model_name}: final_train_dice={final_train_dice:.4f}")
    else:
        train_model(args.model, args)


if __name__ == "__main__":
    main()
















