from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from common.config import cfg
from models import register


def _build_mask_rcnn(model_name: str):
    mcfg = cfg.model_cfg(model_name)
    num_classes = mcfg["num_classes"]
    min_size = mcfg.get("min_size", 800)
    max_size = mcfg.get("max_size", 1333)
    anchor_sizes = mcfg.get("anchor_sizes", [[32, 64, 128, 256, 512]])
    anchor_ratios = mcfg.get("anchor_ratios", [[0.5, 1.0, 2.0]])
    detections_per_img = mcfg.get("box_detections_per_img", 100)
    trainable_layers = mcfg.get("trainable_backbone_layers", 3)

    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=min_size,
        max_size=max_size,
        box_detections_per_img=detections_per_img,
        trainable_backbone_layers=trainable_layers,
    )

    flat_sizes = anchor_sizes[0]
    sizes_tuple = tuple((s,) for s in flat_sizes)
    ratios_tuple = tuple(tuple(anchor_ratios[0]) for _ in range(len(flat_sizes)))
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=sizes_tuple,
        aspect_ratios=ratios_tuple
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


@register("mask_rcnn_hold")
class MaskRCNNHold:
    def __new__(cls, **kwargs):
        return _build_mask_rcnn("mask_rcnn_hold")


@register("mask_rcnn_holdtype")
class MaskRCNNHoldType:
    def __new__(cls, **kwargs):
        return _build_mask_rcnn("mask_rcnn_holdtype")
