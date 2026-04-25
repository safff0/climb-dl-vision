from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline.common.schemas import (
    AttemptAnalysis,
    HAND_MAIN_IDX_LEFT,
    HAND_MAIN_IDX_RIGHT,
    LIMBS,
    Keypoint,
    Limb,
    LimbPointConfig,
    limb_point,
)

PointFn = Callable[[dict[str, Keypoint], str], tuple[float, float, float]]

def _main_frames(analysis: AttemptAnalysis):
    main = next((t for t in analysis.pose_tracks if t.is_main_climber), None)
    if main is None and analysis.pose_tracks:
        main = analysis.pose_tracks[0]
    return [] if main is None else sorted(main.frames, key=lambda f: f.frame)

def _weighted(points: list[Keypoint]) -> tuple[float, float, float]:
    weights = [p.conf for p in points if p.conf > 0]
    pts = [p for p in points if p.conf > 0]
    if not pts or not weights:
        return 0.0, 0.0, 0.0
    wsum = float(sum(weights))
    x = sum(p.x * w for p, w in zip(pts, weights)) / wsum
    y = sum(p.y * w for p, w in zip(pts, weights)) / wsum
    return float(x), float(y), float(wsum / len(weights))

def _legacy_pivot(kp: dict[str, Keypoint], limb: str) -> tuple[float, float, float]:
    if limb == Limb.LEFT_HAND.value:
        m = HAND_MAIN_IDX_LEFT
        return _weighted([kp.get(m[k], Keypoint(0, 0, 0)) for k in ("wrist", "mcp_index", "mcp_middle", "tip_index")])
    if limb == Limb.RIGHT_HAND.value:
        m = HAND_MAIN_IDX_RIGHT
        return _weighted([kp.get(m[k], Keypoint(0, 0, 0)) for k in ("wrist", "mcp_index", "mcp_middle", "tip_index")])
    return limb_point(kp, limb, LimbPointConfig(min_conf=0.0, foot_min_conf=0.0, use_precomputed_pivot=False))

def _hand_mode(mode: str) -> PointFn:
    def fn(kp: dict[str, Keypoint], limb: str) -> tuple[float, float, float]:
        if limb not in (Limb.LEFT_HAND.value, Limb.RIGHT_HAND.value):
            return limb_point(kp, limb)
        m = HAND_MAIN_IDX_LEFT if limb == Limb.LEFT_HAND.value else HAND_MAIN_IDX_RIGHT
        names = ["wrist"]
        if mode in ("wrist_mcp", "wrist_mcp_tip"):
            names += ["mcp_index", "mcp_middle"]
        if mode == "wrist_mcp_tip":
            names += ["tip_index"]
        return _weighted([kp.get(m[n], Keypoint(0, 0, 0)) for n in names])
    return fn

def _metrics(frames, fps: float, point_fn: PointFn) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for limb in LIMBS:
        pts = []
        frame_ids = []
        conf = []
        for f in frames:
            kp = f.keypoints_smooth or f.keypoints_raw
            x, y, c = point_fn(kp, limb)
            if c > 0:
                pts.append((x, y))
                frame_ids.append(f.frame)
            conf.append(c)
        arr = np.asarray(pts, dtype=np.float64)
        farr = np.asarray(frame_ids, dtype=np.int64)
        carr = np.asarray(conf, dtype=np.float64)
        if len(arr) >= 2:
            contiguous = np.diff(farr) == 1
            all_step = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
            step = all_step[contiguous]
        else:
            step = np.zeros(0, dtype=np.float64)
        jerks = []
        start = 0
        while start < len(arr):
            end = start + 1
            while end < len(arr) and farr[end] - farr[end - 1] == 1:
                end += 1
            run = arr[start:end]
            if len(run) >= 4:
                vel = np.diff(run, axis=0) * fps
                acc = np.diff(vel, axis=0) * fps
                jerk = np.diff(acc, axis=0) * fps
                jerks.append(np.hypot(jerk[:, 0], jerk[:, 1]))
            start = end
        jmag = np.concatenate(jerks) if jerks else np.zeros(0, dtype=np.float64)
        out[limb] = {
            "coverage": float((carr > 0).mean()) if carr.size else 0.0,
            "mean_conf": float(carr.mean()) if carr.size else 0.0,
            "conf_lt_0_3": float((carr < 0.3).mean()) if carr.size else 0.0,
            "p95_step": float(np.percentile(step, 95)) if step.size else 0.0,
            "p99_step": float(np.percentile(step, 99)) if step.size else 0.0,
            "max_step": float(step.max()) if step.size else 0.0,
            "count_step_gt_40": int((step > 40).sum()),
            "p95_jerk": float(np.percentile(jmag, 95)) if jmag.size else 0.0,
        }
    return out

def _flatten(metrics: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ablation, by_limb in metrics.items():
        for limb, vals in by_limb.items():
            row: dict[str, object] = {"ablation": ablation, "limb": limb}
            row.update(vals)
            rows.append(row)
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    analysis = AttemptAnalysis.from_dict(json.loads(args.analysis.read_text()))
    frames = _main_frames(analysis)
    fps = args.fps or analysis.fps
    ablations: dict[str, PointFn] = {
        "A0_legacy_current": _legacy_pivot,
        "A2_robust_limb_point": lambda kp, limb: limb_point(kp, limb),
        "A7_wrist_only": _hand_mode("wrist_only"),
        "A7_wrist_mcp": _hand_mode("wrist_mcp"),
        "A7_wrist_mcp_tip": _hand_mode("wrist_mcp_tip"),
    }
    metrics = {name: _metrics(frames, fps, fn) for name, fn in ablations.items()}
    rows = _flatten(metrics)
    with (args.out / "ablation_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "ablation_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
