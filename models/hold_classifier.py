import torch
from torch import nn
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnet18, ResNet18_Weights,
)

from common.config import cfg
from data.crop_dataset import get_dataset_info
from models import register


def _build_convnext(use_mask: bool, num_classes: int, dropout: float):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)

    if use_mask:
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = 0.0
            new_conv.bias.copy_(old_conv.bias)
        model.features[0][0] = new_conv

    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_resnet18(use_mask: bool, num_classes: int, dropout: float):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    if use_mask:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = 0.0
        model.conv1 = new_conv

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


BACKBONES = {
    "convnext_tiny": _build_convnext,
    "resnet18": _build_resnet18,
}


def _build_classifier(model_name: str):
    mcfg = cfg.model_cfg(model_name)
    use_mask = mcfg["use_mask_channel"]
    backbone = mcfg.get("backbone", "convnext_tiny")
    dropout = mcfg.get("dropout", 0.3)
    info = get_dataset_info(model_name)

    builder = BACKBONES[backbone]
    return builder(use_mask, info.num_classes, dropout)


@register("hold_color_classifier")
class HoldColorClassifier:
    def __new__(cls, **kwargs):
        return _build_classifier("hold_color_classifier")


@register("hold_type_classifier")
class HoldTypeClassifier:
    def __new__(cls, **kwargs):
        return _build_classifier("hold_type_classifier")
