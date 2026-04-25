from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


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


class RouteState(StrEnum):
    CORE = "core"
    POSSIBLE = "possible"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_list(cls, xs: list[float] | tuple[float, ...]) -> "BBox":
        return cls(float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3]))


@dataclass
class PhysicalHold:
    physical_track_id: str
    bbox: BBox
    center: tuple[float, float]
    area: float
    seg_class: str

    color_label_raw: str
    color_conf_raw: float
    color_probs_raw: dict[str, float]

    color_label_temporal: str
    color_conf_temporal: float
    color_probs_temporal: dict[str, float]
    color_entropy: float

    type_label: str
    type_conf: float
    type_probs_raw: dict[str, float] = field(default_factory=dict)
    type_probs_temporal: dict[str, float] = field(default_factory=dict)

    frames_seen: list[int] = field(default_factory=list)
    det_conf_mean: float = 0.0
    det_conf_max: float = 0.0

    route_state: str = RouteState.UNKNOWN.value
    route_score: float = 0.0
    color_score: float = 0.0
    graph_score: float = 0.0
    track_score: float = 0.0
    det_score: float = 0.0

    usage_score: float = 0.0
    usage_by_limb: str | None = None
    usage_per_limb: dict[str, float] = field(default_factory=dict)

    mask_rle: dict[str, Any] | None = None
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = self.bbox.to_list()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhysicalHold":
        d = dict(d)
        d["bbox"] = BBox.from_list(d["bbox"])
        return cls(**d)


@dataclass
class Route:
    target_color: str
    holds: list[PhysicalHold]

    def core_holds(self) -> list[PhysicalHold]:
        return [h for h in self.holds if h.route_state == RouteState.CORE.value]

    def possible_holds(self) -> list[PhysicalHold]:
        return [h for h in self.holds if h.route_state == RouteState.POSSIBLE.value]

    def active_holds(self) -> list[PhysicalHold]:
        active = {RouteState.CORE.value, RouteState.POSSIBLE.value}
        return [h for h in self.holds if h.route_state in active]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_color": self.target_color,
            "holds": [h.to_dict() for h in self.holds],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Route":
        return cls(
            target_color=d["target_color"],
            holds=[PhysicalHold.from_dict(x) for x in d["holds"]],
        )


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
