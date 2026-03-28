"""Model definition compatible with ISIC training checkpoints.

Backbones available:
  - efficientnet_v2_s / m / l
  - convnext_base / convnext_small
  - swin_v2_b / swin_v2_s
  - resnet50 / resnet18
"""

import sys
import torch
import torch.nn as nn
import torchvision.models as tv_models


class Network(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        num_classes: int = 7,
        input_channel: int = 3,
        pretrained=True,
    ):
        super().__init__()

        # Keep compatibility with training code where pretrained may be bool or string.
        use_pretrained = (pretrained == "pretrained") or (pretrained is True)

        if input_channel != 3:
            raise ValueError("This Network currently supports RGB input only (input_channel=3).")

        if backbone == "efficientnet_v2_s":
            weights = tv_models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.efficientnet_v2_s(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        elif backbone == "efficientnet_v2_m":
            weights = tv_models.EfficientNet_V2_M_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.efficientnet_v2_m(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        elif backbone == "efficientnet_v2_l":
            weights = tv_models.EfficientNet_V2_L_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.efficientnet_v2_l(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        elif backbone == "convnext_base":
            weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.convnext_base(weights=weights)
            in_features = base.classifier[2].in_features
            base.classifier[2] = nn.Linear(in_features, num_classes)
            self.model = base

        elif backbone == "convnext_small":
            weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.convnext_small(weights=weights)
            in_features = base.classifier[2].in_features
            base.classifier[2] = nn.Linear(in_features, num_classes)
            self.model = base

        elif backbone == "swin_v2_b":
            weights = tv_models.Swin_V2_B_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.swin_v2_b(weights=weights)
            in_features = base.head.in_features
            base.head = nn.Linear(in_features, num_classes)
            self.model = base

        elif backbone == "swin_v2_s":
            weights = tv_models.Swin_V2_S_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.swin_v2_s(weights=weights)
            in_features = base.head.in_features
            base.head = nn.Linear(in_features, num_classes)
            self.model = base

        elif backbone == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.resnet50(weights=weights)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            self.model = base

        elif backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
            base = tv_models.resnet18(weights=weights)
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            self.model = base

        else:
            print(f"Unknown backbone: {backbone}")
            sys.exit(-1)

    def forward(self, x):
        return self.model(x)


# Input size and normalization by backbone family.
BACKBONE_CONFIGS = {
    "efficientnet_v2_s": (384, 384, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "efficientnet_v2_m": (480, 480, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "efficientnet_v2_l": (480, 480, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "convnext_base": (232, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "convnext_small": (230, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "swin_v2_b": (272, 256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "swin_v2_s": (260, 256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "resnet50": (256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "resnet18": (256, 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


if __name__ == "__main__":
    net = Network(backbone="efficientnet_v2_s", num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    print(net(x).shape)
