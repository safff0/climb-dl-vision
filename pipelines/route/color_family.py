import json
from pathlib import Path


_CALIBRATED: dict[str, dict[str, float]] = {
    "Green": {
        "Green": 0.584, "Black": 0.101, "Blue": 0.055, "Pink": 0.031,
        "Red": 0.046, "White": 0.047, "Purple": 0.036, "Gray": 0.040,
        "Yellow": 0.031, "Orange": 0.029,
    },
    "Orange": {
        "Black": 0.233, "Orange": 0.162, "Red": 0.121, "Gray": 0.107,
        "Yellow": 0.100, "Purple": 0.069, "Blue": 0.065, "White": 0.056,
        "Green": 0.047, "Pink": 0.042,
    },
}


_FAMILIES: dict[str, dict[str, float]] = {
    "Orange": {"Orange": 1.0, "Red": 0.55, "Yellow": 0.30, "Pink": 0.15},
    "Red":    {"Red": 1.0, "Orange": 0.55, "Pink": 0.40, "Purple": 0.20},
    "Yellow": {"Yellow": 1.0, "Orange": 0.35, "Green": 0.20, "White": 0.15},
    "Green":  {"Green": 1.0, "Yellow": 0.25, "Blue": 0.15},
    "Blue":   {"Blue": 1.0, "Purple": 0.45, "Gray": 0.15},
    "Purple": {"Purple": 1.0, "Blue": 0.40, "Pink": 0.30, "Red": 0.20},
    "Pink":   {"Pink": 1.0, "Red": 0.40, "Purple": 0.30, "Orange": 0.20},
    "Black":  {"Black": 1.0, "Gray": 0.35},
    "White":  {"White": 1.0, "Gray": 0.30, "Yellow": 0.15},
    "Gray":   {"Gray": 1.0, "Black": 0.35, "White": 0.30},
}


def _maybe_load_override() -> None:
    for p in (
        Path("runs/color_calibration.json"),
        Path(__file__).resolve().parents[2] / "runs" / "color_calibration.json",
    ):
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            cm = data.get("confusion_matrix", {})
            for tgt, row in cm.items():
                _CALIBRATED[tgt] = {c: float(v) for c, v in row.items()}
            return
        except Exception:
            continue


_maybe_load_override()


def _weights_for(target: str) -> dict[str, float]:
    return _FAMILIES.get(target, {target: 1.0})


def family_prob(
    target: str, color_probs: dict[str, float], enabled: bool = True,
) -> float:
    if not enabled:
        return float(color_probs.get(target, 0.0))
    w = _weights_for(target)
    s = 0.0
    for c, ww in w.items():
        s += ww * float(color_probs.get(c, 0.0))
    return float(s)


def dominant_non_family(
    target: str, color_probs: dict[str, float],
) -> tuple[str, float]:
    w = _weights_for(target)
    best = ("", 0.0)
    in_family = {c for c, ww in w.items() if ww >= 0.10}
    for c, p in color_probs.items():
        if c in in_family:
            continue
        if p > best[1]:
            best = (c, float(p))
    return best
