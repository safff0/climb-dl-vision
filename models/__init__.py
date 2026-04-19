from torch import nn

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def create_model(name: str, **kwargs) -> nn.Module:
    return MODEL_REGISTRY[name](**kwargs)


import models.color_gnn
import models.eva02_color
import models.hold_classifier
import models.mask2former
import models.mask_rcnn
