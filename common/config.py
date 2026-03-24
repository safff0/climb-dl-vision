from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import os
import yaml


@dataclass
class TorchConfig:
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4


@dataclass
class Config:
    torch: TorchConfig = field(default_factory=TorchConfig)
    models: Dict[str, Any] = field(default_factory=dict)

    def model_cfg(self, model_name: str, mode: str) -> Dict[str, Any]:
        return self.models.get(model_name, {}).get(mode, {})


def _load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    torch_cfg = TorchConfig(**raw.get("torch", {}))
    return Config(torch=torch_cfg, models=raw.get("models", {}))


_DEFAULT_PATH = "config.yaml"
_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(_DEFAULT_PATH)))


if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Config file not found: {_CONFIG_PATH}. "
        f"Set CONFIG_PATH env var or place config.yaml next to common/config.py"
    )


cfg: Config = _load_config(_CONFIG_PATH)
