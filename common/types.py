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


class SegClass(StrEnum):
    HOLD = "hold"
    VOLUME = "volume"


@dataclass
class Detection:
    box: object = None
    mask: object = None
    seg_label: str = ""
    score: float = 0.0
    color: str | None = None
    color_probs: dict[str, float] | None = None
    color_cluster: int | None = None
    color_clustered: str | None = None
    hold_type: str | None = None
    type_probs: dict[str, float] | None = None

    def display_label(self) -> str:
        parts = [self.seg_label]
        color = self.color_clustered or self.color
        if color:
            parts.append(color)
        if self.hold_type:
            parts.append(self.hold_type)
        return " | ".join(parts)

    def to_dict(self) -> dict:
        d = {
            "bbox": self.box.cpu().tolist() if self.box is not None else None,
            "seg_class": self.seg_label,
            "score": round(self.score, 4),
        }
        if self.color is not None:
            d["color"] = self.color_clustered or self.color
            d["color_raw"] = self.color
            d["color_probs"] = self.color_probs
            if self.color_cluster is not None:
                d["color_cluster"] = self.color_cluster
        if self.hold_type is not None:
            d["type"] = self.hold_type
            d["type_probs"] = self.type_probs
        return d


@dataclass
class ImagePredictions:
    image: str = ""
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class CropRecord:
    file: str = ""
    label: int = 0
    source_image: str = ""
    pred_box: list[int] = field(default_factory=list)
    mask_file: str | None = None

    def to_dict(self) -> dict:
        d = {
            "file": self.file,
            "label": self.label,
            "source_image": self.source_image,
            "pred_box": self.pred_box,
        }
        if self.mask_file is not None:
            d["mask_file"] = self.mask_file
        return d


@dataclass
class CropMeta:
    class_names: list[str] = field(default_factory=list)
    num_classes: int = 0
    crops: list[CropRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "crops": [c.to_dict() for c in self.crops],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CropMeta":
        return cls(
            class_names=d["class_names"],
            num_classes=d["num_classes"],
            crops=[CropRecord(
                file=c["file"], label=c["label"],
                source_image=c["source_image"], pred_box=c["pred_box"],
                mask_file=c.get("mask_file"),
            ) for c in d["crops"]],
        )
