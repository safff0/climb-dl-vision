"""Compile AttemptAnalysis dumps into a browser-ready demo dataset.

Emits one compact ``client.json`` per attempt plus the static assets the
web UI loads directly: ``route_frame.jpg``, ``pickmap.png``,
``mask_contours.json``, ``overlay_720p.mp4`` and WebP versions of the
existing PNG visualisations. A top-level ``manifest.json`` lists every
attempt with minimal metadata for the landing grid.

Design rules (from plan A0):

* Python does the heavy geometry — the browser never decodes RLEs or
  runs contour extraction.
* Pickmap is a hidden PNG where every hold is filled with a unique RGB
  triple so that click-to-hold is a single ``getImageData`` lookup.
* ``client.json`` stays small: keep hold summaries, contact segments,
  events, metrics; drop the 133-keypoint per-frame pose payload.

Usage:
    python tools/export_demo_assets.py \
        --analysis-dir results/thewayup/yolo \
        --raw-video-dir stage_datasets/the_way_up \
        --full-pipeline-dir results/full_pipeline_v2 \
        --out demo/public/data \
        --transcode-video
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline.common.masks import rle_to_mask  # noqa: E402


LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _idx_to_rgb(idx: int) -> tuple[int, int, int]:
    """Encode a 1-based hold index as RGB. 0,0,0 = background."""
    return (idx & 0xFF, (idx >> 8) & 0xFF, (idx >> 16) & 0xFF)


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ! {' '.join(cmd[:3])}... failed: {res.stderr.strip().splitlines()[-1] if res.stderr else 'rc=' + str(res.returncode)}")


def _png_to_webp(src: Path, dst: Path, quality: int = 82) -> bool:
    if not src.exists():
        return False
    try:
        im = Image.open(src).convert("RGBA" if src.suffix == ".png" else "RGB")
        # WebP doesn't carry alpha for the final overlay; route_overlay has
        # alpha but heatmaps/timeline are RGB. We drop alpha for trajectories
        # and timelines (opaque), keep it for route_overlay.
        if src.name == "route_overlay.png":
            im.save(dst, "WEBP", quality=quality, method=6)
        else:
            if im.mode == "RGBA":
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])
                im = bg
            im.save(dst, "WEBP", quality=quality, method=6)
        return True
    except Exception as e:
        print(f"  ! png->webp {src.name} failed: {e}")
        return False


def _extract_first_frame(video: Path, dst: Path, target_w: int = 720) -> bool:
    if not video.exists():
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-vframes", "1",
        "-vf", f"scale={target_w}:-2",
        "-q:v", "3",
        str(dst),
    ]
    _run(cmd)
    return dst.exists()


def _transcode_720p(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", "scale=-2:720",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    _run(cmd)
    return dst.exists()


# ------------------------------------------------------------------
# core: per-attempt compile
# ------------------------------------------------------------------


@dataclass
class Source:
    attempt_id: str
    participant: str
    route_color: str
    label: str
    analysis: Path
    route_overlay: Path | None
    timeline: Path | None
    trajectories: Path | None
    heatmap_dir: Path | None
    overlay_mp4: Path | None
    raw_video: Path | None


def discover_sources(
    analysis_dir: Path, raw_video_dir: Path, full_pipeline_dir: Path | None
) -> list[Source]:
    out: list[Source] = []
    for sub in sorted(analysis_dir.iterdir()):
        if not sub.is_dir() or not (sub / "analysis.json").exists():
            continue
        clip = sub.name  # e.g. p9_orange
        pnum, _, color = clip.partition("_")
        raw = raw_video_dir / pnum / f"{color}.mp4" if color else None
        overlay = sub / "overlay.mp4"
        out.append(Source(
            attempt_id=clip,
            participant=pnum,
            route_color=color or "unknown",
            label=f"{pnum} · {color} route" if color else clip,
            analysis=sub / "analysis.json",
            route_overlay=(sub / "route_overlay.png") if (sub / "route_overlay.png").exists() else None,
            timeline=(sub / "contact_timeline.png") if (sub / "contact_timeline.png").exists() else None,
            trajectories=(sub / "trajectories.png") if (sub / "trajectories.png").exists() else None,
            heatmap_dir=(sub / "heatmaps") if (sub / "heatmaps").exists() else None,
            overlay_mp4=overlay if overlay.exists() else None,
            raw_video=raw if raw and raw.exists() else None,
        ))
    # Optional: attach a full-pipeline variant so the demo can showcase
    # real Mask2Former masks.
    if full_pipeline_dir and full_pipeline_dir.exists():
        for sub in sorted(full_pipeline_dir.iterdir()):
            if not sub.is_dir() or not (sub / "analysis.json").exists():
                continue
            clip = sub.name
            pnum, _, color = clip.partition("_")
            raw = raw_video_dir / pnum / f"{color}.mp4" if color else None
            ov = sub / "overlay.mp4"
            out.append(Source(
                attempt_id=f"full_{clip}",
                participant=pnum,
                route_color=color or "unknown",
                label=f"{pnum} · {color} route · full pipeline",
                analysis=sub / "analysis.json",
                route_overlay=(sub / "route_overlay.png") if (sub / "route_overlay.png").exists() else None,
                timeline=(sub / "contact_timeline.png") if (sub / "contact_timeline.png").exists() else None,
                trajectories=(sub / "trajectories.png") if (sub / "trajectories.png").exists() else None,
                heatmap_dir=(sub / "heatmaps") if (sub / "heatmaps").exists() else None,
                overlay_mp4=ov if ov.exists() else None,
                raw_video=raw if raw and raw.exists() else None,
            ))
    return out


def _mask_from_hold(hold: dict, wh: tuple[int, int]) -> np.ndarray | None:
    """Return a boolean mask for a hold. Uses mask_rle if present, else a
    filled bbox. Falls back to None if geometry is invalid.
    """
    W, H = wh
    rle = hold.get("mask_rle")
    if rle is not None and rle.get("counts") and rle.get("size"):
        try:
            m = rle_to_mask(rle)
            # cocomask returns (H, W) boolean; match the route_frame
            # orientation later if sizes disagree.
            if m.shape[0] != H or m.shape[1] != W:
                # try swap (pipeline stores [H, W] but we may have probed (W, H))
                if m.shape == (W, H):
                    m = m.T
            return m.astype(bool)
        except Exception:
            pass
    b = hold.get("bbox")
    if not b or len(b) < 4:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _probe_image_size(src: Source) -> tuple[int, int]:
    """Return (W, H). Prefer the route overlay if available."""
    if src.route_overlay and src.route_overlay.exists():
        try:
            with Image.open(src.route_overlay) as im:
                return im.size  # (W, H)
        except Exception:
            pass
    if src.raw_video and src.raw_video.exists():
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                 str(src.raw_video)],
                capture_output=True, text=True,
            )
            s = probe.stdout.strip()
            if "x" in s:
                w, h = s.split("x")
                return int(w), int(h)
        except Exception:
            pass
    # sensible default
    return 1280, 720


def _build_pickmap_and_contours(
    holds: list[dict], wh: tuple[int, int]
) -> tuple[Image.Image, dict[str, list[list[list[int]]]], dict[str, list[int]]]:
    """Rasterise holds into a pickmap PNG. Returns (image, contours dict,
    pick_rgb dict). Index 0 = background; holds get indices 1..N mapping
    to unique RGB triples.
    """
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore

    W, H = wh
    pick = np.zeros((H, W, 3), dtype=np.uint8)
    contours: dict[str, list[list[list[int]]]] = {}
    pick_rgb: dict[str, list[int]] = {}

    for i, h in enumerate(holds, start=1):
        mask = _mask_from_hold(h, wh)
        if mask is None or mask.shape != (H, W):
            continue
        rgb = _idx_to_rgb(i)
        pick[mask] = rgb
        hid = h["physical_track_id"]
        pick_rgb[hid] = list(rgb)

        # Polygons
        if cv2 is not None:
            u8 = mask.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys: list[list[list[int]]] = []
            for c in cnts:
                eps = 1.5
                approx = cv2.approxPolyDP(c, eps, True)
                if len(approx) < 3:
                    continue
                polys.append([[int(p[0][0]), int(p[0][1])] for p in approx])
            if polys:
                contours[hid] = polys
        if hid not in contours:
            # fallback: bbox corners
            b = h.get("bbox")
            if b and len(b) >= 4:
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                contours[hid] = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

    return Image.fromarray(pick, mode="RGB"), contours, pick_rgb


def _slim_client_json(
    src: Source,
    analysis: dict,
    wh: tuple[int, int],
    pick_rgb: dict[str, list[int]],
    assets: dict[str, str],
) -> dict:
    W, H = wh
    fps = float(analysis["fps"])
    frame_count = int(analysis["frame_count"])
    duration_sec = float(analysis["duration_sec"])

    holds_in = analysis["route"]["holds"]
    holds_out = []
    for h in holds_in:
        hid = h["physical_track_id"]
        holds_out.append({
            "id": hid,
            "bbox": [round(float(x), 2) for x in h["bbox"]],
            "center": [round(float(h["center"][0]), 2), round(float(h["center"][1]), 2)],
            "area": round(float(h.get("area", 0.0)), 2),
            "seg_class": h.get("seg_class", "hold"),
            "route_state": h.get("route_state", "unknown"),
            "route_score": round(float(h.get("route_score", 0.0)), 4),
            "score_components": {
                "color": round(float(h.get("color_score", 0.0)), 4),
                "graph": round(float(h.get("graph_score", 0.0)), 4),
                "track": round(float(h.get("track_score", 0.0)), 4),
                "det":   round(float(h.get("det_score", 0.0)), 4),
            },
            "usage_score": round(float(h.get("usage_score", 0.0)), 4),
            "usage_by_limb": h.get("usage_by_limb"),
            "type_probs_temporal": {k: round(float(v), 4) for k, v in (h.get("type_probs_temporal") or {}).items()},
            "schema_version": int(h.get("schema_version", 1)),
            "color": {
                "label_raw": h.get("color_label_raw"),
                "conf_raw": round(float(h.get("color_conf_raw", 0.0)), 4),
                "probs_raw": {k: round(float(v), 4) for k, v in (h.get("color_probs_raw") or {}).items()},
                "label_temporal": h.get("color_label_temporal"),
                "conf_temporal": round(float(h.get("color_conf_temporal", 0.0)), 4),
                "probs_temporal": {k: round(float(v), 4) for k, v in (h.get("color_probs_temporal") or {}).items()},
                "entropy": round(float(h.get("color_entropy", 0.0)), 4),
            },
            "type": {
                "label": h.get("type_label", "unknown"),
                "conf": round(float(h.get("type_conf", 0.0)), 4),
            },
            "det_conf_mean": round(float(h.get("det_conf_mean", 0.0)), 4),
            "det_conf_max": round(float(h.get("det_conf_max", 0.0)), 4),
            "frames_seen": h.get("frames_seen", []),
            "pick_rgb": pick_rgb.get(hid),
        })

    segments = {}
    for limb in LIMBS:
        segs = analysis["contacts"].get(limb, [])
        segments[limb] = [
            {
                "hold_id": s.get("hold_id"),
                "start_frame": int(s["start_frame"]),
                "end_frame": int(s["end_frame"]),
                "state": s["state"],
                "confidence": round(float(s.get("confidence", 0.0)), 4),
            }
            for s in segs
        ]

    events = {
        "moves": [
            {
                "id": m.get("event_id"),
                "limb": m["limb"],
                "from": m.get("from_hold"),
                "to": m.get("to_hold"),
                "start_frame": int(m["start_frame"]),
                "end_frame": int(m["end_frame"]),
                "duration_sec": round(float(m.get("duration_sec", 0.0)), 4),
                "path_length": round(float(m.get("path_length", 0.0)), 2),
                "max_speed": round(float(m.get("max_speed", 0.0)), 2),
                "mean_jerk": round(float(m.get("mean_jerk", 0.0)), 2),
                "confidence": round(float(m.get("confidence", 0.0)), 4),
            }
            for m in analysis.get("move_events", [])
        ],
        "readjustments": [
            {
                "id": r.get("readjustment_id"),
                "limb": r["limb"],
                "hold_id": r.get("hold_id"),
                "start_frame": int(r["start_frame"]),
                "end_frame": int(r["end_frame"]),
                "total_amplitude": round(float(r.get("total_amplitude", 0.0)), 2),
                "segment_count": int(r.get("segment_count", 0)),
            }
            for r in analysis.get("readjustments", [])
        ],
        "hesitations": [
            {
                "id": h.get("hesitation_id"),
                "kind": h.get("kind"),
                "limb": h.get("limb"),
                "start_frame": int(h["start_frame"]),
                "end_frame": int(h["end_frame"]),
                "duration_sec": round(float(h.get("duration_sec", 0.0)), 4),
            }
            for h in analysis.get("hesitations", [])
        ],
    }

    # Pick main climber body trajectory at keyframes (every ~4 frames)
    # so the client can draw a small body path without loading the
    # 1 300-frame pose dump.
    body_traj: list[list[float]] = []
    tracks = analysis.get("pose_tracks", [])
    main = next((t for t in tracks if t.get("is_main_climber")), tracks[0] if tracks else None)
    if main is not None:
        frames = main.get("frames", [])
        step = max(1, len(frames) // 200)
        for pf in frames[::step]:
            kps = pf.get("keypoints_smooth") or pf.get("keypoints_raw") or {}
            lh = kps.get("left_hip")
            rh = kps.get("right_hip")
            if lh and rh and lh[2] > 0 and rh[2] > 0:
                body_traj.append([
                    int(pf["frame"]),
                    round(0.5 * (lh[0] + rh[0]), 1),
                    round(0.5 * (lh[1] + rh[1]), 1),
                ])

    metrics = analysis.get("metrics", {})

    return {
        "id": src.attempt_id,
        "label": src.label,
        "participant": src.participant,
        "route_color": src.route_color,
        "width": W,
        "height": H,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "assets": assets,
        "route": {
            "target_color": analysis["route"].get("target_color", src.route_color),
            "counts": {
                "core": sum(1 for h in holds_in if h.get("route_state") == "core"),
                "possible": sum(1 for h in holds_in if h.get("route_state") == "possible"),
                "rejected": sum(1 for h in holds_in if h.get("route_state") == "rejected"),
                "unknown": sum(1 for h in holds_in if h.get("route_state") == "unknown"),
            },
        },
        "holds": holds_out,
        "contacts": {"segments": segments},
        "events": events,
        "metrics": metrics,
        "body_trajectory": body_traj,
    }


def compile_attempt(src: Source, out_dir: Path, transcode_video: bool) -> dict | None:
    dst = out_dir / "attempts" / src.attempt_id
    dst.mkdir(parents=True, exist_ok=True)

    with src.analysis.open() as f:
        analysis = json.load(f)

    wh = _probe_image_size(src)

    # ---- pickmap + contours
    holds = analysis["route"]["holds"]
    pick_img, contours, pick_rgb = _build_pickmap_and_contours(holds, wh)
    pick_img.save(dst / "pickmap.png", optimize=True)
    with (dst / "mask_contours.json").open("w") as f:
        json.dump(contours, f, separators=(",", ":"))

    # ---- route_frame.jpg (first frame of raw video when possible)
    rf_ok = False
    if src.raw_video and src.raw_video.exists():
        rf_ok = _extract_first_frame(src.raw_video, dst / "route_frame.jpg", target_w=min(wh[0], 1080))
    if not rf_ok and src.route_overlay:
        try:
            im = Image.open(src.route_overlay).convert("RGB")
            im.save(dst / "route_frame.jpg", "JPEG", quality=88)
            rf_ok = True
        except Exception:
            pass

    # ---- visualisations → webp
    if src.route_overlay:
        _png_to_webp(src.route_overlay, dst / "route_overlay.webp", quality=85)
    if src.timeline:
        _png_to_webp(src.timeline, dst / "timeline.webp", quality=82)
    if src.trajectories:
        _png_to_webp(src.trajectories, dst / "trajectories.webp", quality=82)

    hm_out = dst / "heatmaps"
    hm_out.mkdir(exist_ok=True)
    if src.heatmap_dir:
        for limb in list(LIMBS) + ["body"]:
            p = src.heatmap_dir / f"heatmap_{limb}.png"
            _png_to_webp(p, hm_out / f"{limb}.webp", quality=70)

    # ---- overlay_720p.mp4
    video_asset: str | None = None
    if transcode_video:
        ok = False
        if src.overlay_mp4 and src.overlay_mp4.exists():
            ok = _transcode_720p(src.overlay_mp4, dst / "overlay_720p.mp4")
        if not ok and src.raw_video and src.raw_video.exists():
            ok = _transcode_720p(src.raw_video, dst / "overlay_720p.mp4")
        if ok:
            video_asset = "overlay_720p.mp4"

    assets = {
        "route_frame": "route_frame.jpg" if rf_ok else None,
        "route_overlay": "route_overlay.webp" if src.route_overlay else None,
        "pickmap": "pickmap.png",
        "mask_contours": "mask_contours.json",
        "timeline": "timeline.webp" if src.timeline else None,
        "trajectories": "trajectories.webp" if src.trajectories else None,
        "video": video_asset,
        "heatmaps": {
            limb: (f"heatmaps/{limb}.webp" if (hm_out / f"{limb}.webp").exists() else None)
            for limb in list(LIMBS) + ["body"]
        },
    }

    client = _slim_client_json(src, analysis, wh, pick_rgb, assets)
    with (dst / "client.json").open("w") as f:
        json.dump(client, f, separators=(",", ":"))

    # manifest entry
    manifest = {
        "id": src.attempt_id,
        "label": src.label,
        "participant": src.participant,
        "route_color": src.route_color,
        "width": wh[0],
        "height": wh[1],
        "fps": client["fps"],
        "frame_count": client["frame_count"],
        "duration_sec": client["duration_sec"],
        "hold_count": len(client["holds"]),
        "core_holds": client["route"]["counts"]["core"],
        "total_moves": len(client["events"]["moves"]),
        "has_overlay_video": video_asset is not None,
        "has_true_masks": any(h.get("mask_rle") for h in holds),
    }
    return manifest


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, required=True,
                    help="Directory containing per-clip analysis subdirs (e.g. results/thewayup/yolo)")
    ap.add_argument("--raw-video-dir", type=Path, default=_REPO / "stage_datasets" / "the_way_up")
    ap.add_argument("--full-pipeline-dir", type=Path, default=_REPO / "results" / "full_pipeline_v2")
    ap.add_argument("--out", type=Path, required=True, help="e.g. demo/public/data")
    ap.add_argument("--transcode-video", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(
        args.analysis_dir, args.raw_video_dir,
        args.full_pipeline_dir if args.full_pipeline_dir.exists() else None,
    )
    if args.limit is not None:
        sources = sources[: args.limit]

    entries = []
    for i, src in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] {src.attempt_id}")
        try:
            m = compile_attempt(src, args.out, transcode_video=args.transcode_video)
            if m is not None:
                entries.append(m)
        except Exception as e:
            print(f"  ! failed: {type(e).__name__}: {e}")

    manifest = {
        "generated_at": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
        ).stdout.strip(),
        "attempts": entries,
    }
    with (args.out / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(entries)} attempts to {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
