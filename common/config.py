from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os
import yaml

from common.types import DatasetConfig, SchedulerType, TrainConfig, ValidateConfig


@dataclass
class TorchConfig:
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4


@dataclass
class Config:
    torch: TorchConfig = field(default_factory=TorchConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)

    def model_cfg(self, model_name: str) -> dict[str, Any]:
        return self.models[model_name]

    def train_cfg(self, model_name: str) -> TrainConfig:
        return self.models[model_name]["_train"]

    def validate_cfg(self, model_name: str) -> ValidateConfig:
        return self.models[model_name]["_validate"]


def _parse_train_config(raw: dict) -> TrainConfig:
    return TrainConfig(
        batch_size=raw["batch_size"],
        lr=raw["lr"],
        epochs=raw["epochs"],
        scheduler=SchedulerType(raw["scheduler"]),
        lr_milestones=raw.get("lr_milestones", []),
        lr_gamma=raw.get("lr_gamma", 0.1),
        freeze_backbone_epochs=raw.get("freeze_backbone_epochs", 0),
        freeze_backbone_lr=raw.get("freeze_backbone_lr", 0.01),
    )


def _parse_validate_config(raw: dict) -> ValidateConfig:
    return ValidateConfig(batch_size=raw.get("batch_size", 1))


def _load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    torch_cfg = TorchConfig(**raw.get("torch", {}))

    datasets = {}
    for name, ds_raw in raw.get("datasets", {}).items():
        datasets[name] = DatasetConfig(**ds_raw)

    models = {}
    for name, model_raw in raw.get("models", {}).items():
        model_dict = dict(model_raw)
        if "train" in model_dict:
            model_dict["_train"] = _parse_train_config(model_dict.pop("train"))
        if "validate" in model_dict:
            model_dict["_validate"] = _parse_validate_config(model_dict.pop("validate"))
        models[name] = model_dict

    return Config(torch=torch_cfg, models=models, datasets=datasets)


_DEFAULT_PATH = "config.yaml"
_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(_DEFAULT_PATH)))


if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Config file not found: {_CONFIG_PATH}. "
        f"Set CONFIG_PATH env var or place config.yaml next to common/config.py"
    )


cfg: Config = _load_config(_CONFIG_PATH)
