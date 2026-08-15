import torch
import torch.nn as nn


class ConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        model_name="convnext_base",
        fp16=False,
        num_features=512,
        pretrained=False,
        drop_path_rate=0.05,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("ConvNeXt backbone requires timm to be installed.") from exc

        self.fp16 = fp16
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        body_features = self.body.num_features
        self.fc = nn.Linear(body_features, num_features, bias=False)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.features.weight, 1.0)
        nn.init.constant_(self.features.bias, 0.0)
        self.features.weight.requires_grad = False

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=self.fp16):
            x = self.body(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def get_convnext(
    model_name="convnext_base",
    fp16=False,
    num_features=512,
    pretrained=False,
    drop_path_rate=0.05,
):
    return ConvNeXtBackbone(
        model_name=model_name,
        fp16=fp16,
        num_features=num_features,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
    )
