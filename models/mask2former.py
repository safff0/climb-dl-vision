from common.config import cfg
from models import register


def _build_mask2former(model_name: str):
    from transformers import Mask2FormerForUniversalSegmentation

    mcfg = cfg.model_cfg(model_name)
    num_classes = mcfg["num_classes"]
    id2label = {int(k): v for k, v in mcfg["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}
    pretrained = mcfg["model_name_or_path"]

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if mcfg.get("gradient_checkpointing", True):
        backbone = model.model.pixel_level_module.encoder
        backbone.gradient_checkpointing_enable()
    return model


@register("mask2former_hold")
class Mask2FormerHold:
    def __new__(cls, **kwargs):
        return _build_mask2former("mask2former_hold")
