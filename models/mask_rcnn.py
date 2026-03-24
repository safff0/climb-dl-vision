from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from common.config import cfg
from models import register


def _build_mask_rcnn(num_classes: int):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


@register("mask_rcnn_hold")
class MaskRCNNHold:
    def __new__(cls, **kwargs):
        num_classes = cfg.models.get("mask_rcnn_hold", {}).get("num_classes", 3)
        return _build_mask_rcnn(num_classes)


@register("mask_rcnn_holdtype")
class MaskRCNNHoldType:
    def __new__(cls, **kwargs):
        num_classes = cfg.models.get("mask_rcnn_holdtype", {}).get("num_classes", 7)
        return _build_mask_rcnn(num_classes)
