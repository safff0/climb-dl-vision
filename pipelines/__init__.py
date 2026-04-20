from typing import Callable

from common.types import PipelineMode

PIPELINE_REGISTRY: dict[str, dict[PipelineMode, Callable]] = {}


def register_pipeline(name: str, mode: PipelineMode):
    def decorator(fn):
        PIPELINE_REGISTRY.setdefault(name, {})[mode] = fn
        return fn
    return decorator


def get_pipeline(name: str, mode: PipelineMode) -> Callable:
    return PIPELINE_REGISTRY[name][mode]


import pipelines.climb
import pipelines.color_gnn
import pipelines.color_handcrafted
import pipelines.eva02_color
import pipelines.eva02_type
import pipelines.hold_classifier
import pipelines.mask2former
import pipelines.mask_rcnn
