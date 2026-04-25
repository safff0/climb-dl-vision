"""Recompute keypoints_smooth for saved AttemptAnalysis pose tracks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.analysis.report import write_json
from pipeline.common.schemas import AttemptAnalysis
from pipeline.pose.smooth import SmoothConfig, smooth_pose_track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    analysis = AttemptAnalysis.from_dict(json.loads(args.analysis.read_text()))
    for track in analysis.pose_tracks:
        for frame in track.frames:
            frame.keypoints_smooth = None
            frame.limb_quality = {}
        smooth_pose_track(track.frames, SmoothConfig())
    write_json(args.out, analysis.to_dict())


if __name__ == "__main__":
    main()
