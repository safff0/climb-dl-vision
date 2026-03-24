from typing import Callable

PIPELINE_REGISTRY: dict[str, dict[str, Callable]] = {}


def register_pipeline(name: str, mode: str):
    def decorator(fn):
        PIPELINE_REGISTRY.setdefault(name, {})[mode] = fn
        return fn
    return decorator


def get_pipeline(name: str, mode: str) -> Callable:
    return PIPELINE_REGISTRY[name][mode]


import pipelines.mask_rcnn
