from dataclasses import dataclass, field
from enum import StrEnum


class Split(StrEnum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class SchedulerType(StrEnum):
    COSINE = "cosine"
    MULTISTEP = "multistep"


class AugmentMode(StrEnum):
    COLOR = "color"
    TYPE = "type"


class PipelineMode(StrEnum):
    TRAIN = "train"
    VALIDATE = "validate"
    INFERENCE = "inference"


@dataclass
class EvalMetrics:
    bbox_ap: float = 0.0
    segm_ap: float = 0.0


@dataclass
class DatasetInfo:
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    cat_id_to_name: dict[int, str] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    dir: str = ""
    val_split: float = 0.15


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    epochs: int
    scheduler: SchedulerType
    lr_milestones: list[int] = field(default_factory=list)
    lr_gamma: float = 0.1
    freeze_backbone_epochs: int = 0
    freeze_backbone_lr: float = 0.01


@dataclass
class ValidateConfig:
    batch_size: int = 1
