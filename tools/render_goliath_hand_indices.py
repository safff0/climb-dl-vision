"""Render numbered Sapiens Goliath hand indices for selected frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from pipeline.common.schemas import AttemptAnalysis
from pipeline.pose.sapiens_pose import PoseBackend, PoseEstimator, PoseEstimatorConfig


def _read_frame(video: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _bboxes_from_diagnostics(path: Path) -> dict[int, tuple[float, float, float, float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    rows = data.get("tracked_smoothed_bboxes", {})
    return {int(k): tuple(float(v) for v in bb) for k, bb in rows.items()}


def _bboxes_from_analysis(path: Path) -> dict[int, tuple[float, float, float, float]]:
    if not path.exists():
        return {}
    analysis = AttemptAnalysis.from_dict(json.loads(path.read_text()))
    main = next((t for t in analysis.pose_tracks if t.is_main_climber), None)
    if main is None and analysis.pose_tracks:
        main = analysis.pose_tracks[0]
    if main is None:
        return {}
    return {f.frame: tuple(f.bbox.to_list()) for f in main.frames}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, default=None)
    ap.add_argument("--bbox-diagnostics", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--times", type=float, nargs="*", default=None)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--model", type=str, default="facebook/sapiens-pose-1b-torchscript")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frames = list(args.frames or [])
    if args.times:
        frames.extend(int(round(t * args.fps)) for t in args.times)
    if not frames:
        frames = [878, 881, 885, 1173, 1177, 1312, 1314, 1330]
    bboxes: dict[int, tuple[float, float, float, float]] = {}
    if args.bbox_diagnostics is not None:
        bboxes.update(_bboxes_from_diagnostics(args.bbox_diagnostics))
    if not bboxes and args.analysis is not None:
        bboxes.update(_bboxes_from_analysis(args.analysis))
    if not bboxes:
        raise RuntimeError("no bboxes found; provide --bbox-diagnostics or --analysis")

    pose = PoseEstimator(PoseEstimatorConfig(
        backend=PoseBackend.SAPIENS,
        model=args.model,
        device=args.device,
    ))
    for frame_idx in sorted(set(frames)):
        bb = bboxes.get(frame_idx)
        if bb is None:
            nearest = min(bboxes, key=lambda k: abs(k - frame_idx))
            bb = bboxes[nearest]
        frame = _read_frame(args.video, frame_idx)
        if frame is None:
            continue
        out = args.out / f"goliath_hand_indices_f{frame_idx:06d}.jpg"
        pose.render_goliath_hand_index_debug(
            frame,
            bb,
            out,
            frame_idx=frame_idx,
            time_sec=frame_idx / max(1.0, args.fps),
        )


if __name__ == "__main__":
    main()
