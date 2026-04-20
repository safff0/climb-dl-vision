import json
from pathlib import Path

from common.config import cfg
from models import register


def _build_eva02_type(model_name: str):
    import timm

    mcfg = cfg.model_cfg(model_name)
    backbone = mcfg.get("backbone", "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")
    pretrained = mcfg.get("pretrained", True)

    if "num_classes" in mcfg:
        num_classes = mcfg["num_classes"]
    else:
        ann_path = Path(mcfg["dataset"]) / "train" / "_annotations.coco.json"
        with open(ann_path) as f:
            data = json.load(f)
        num_classes = len(data["categories"])

    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


@register("eva02_type")
class EVA02Type:
    def __new__(cls, **kwargs):
        return _build_eva02_type("eva02_type")
