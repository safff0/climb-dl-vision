from common.config import cfg
from models import register


def _build_eva02(model_name: str):
    import timm

    mcfg = cfg.model_cfg(model_name)
    backbone = mcfg.get("backbone", "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")
    num_classes = mcfg["num_classes"]
    pretrained = mcfg.get("pretrained", True)

    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model


@register("eva02_color")
class EVA02Color:
    def __new__(cls, **kwargs):
        return _build_eva02("eva02_color")
