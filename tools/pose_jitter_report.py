"""Pose jitter diagnostics for an AttemptAnalysis JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline.analysis.report import write_json
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


def _main_frames(analysis: AttemptAnalysis):
    main = next((t for t in analysis.pose_tracks if t.is_main_climber), None)
    if main is None and analysis.pose_tracks:
        main = analysis.pose_tracks[0]
    return [] if main is None else sorted(main.frames, key=lambda f: f.frame)


REPORT_WINDOWS: dict[str, tuple[int, int]] = {
    "crop_sensitive_125_137": (125, 137),
    "first_occlusion_149_157": (149, 157),
    "false_switch_158_188": (158, 188),
    "zero_reappear_189_193": (189, 193),
    "long_false_contact_194_230": (194, 230),
    "occlusion_exit_231_250": (231, 250),
    "later_regression_1310_1331": (1310, 1331),
}


RAW_PIVOT_CFG = LimbPointConfig(min_conf=0.0, use_precomputed_pivot=False)


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _inside_torso(frame, x: float, y: float) -> bool:
    kp = frame.keypoints_smooth or frame.keypoints_raw
    names = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
    pts: list[tuple[float, float]] = []
    for name in names:
        p = kp.get(name)
        if p is None or p.conf < 0.25:
            return False
        pts.append((float(p.x), float(p.y)))
    return _point_in_polygon(x, y, pts)


def _forearm_baseline(frames, limb: str) -> float | None:
    if limb not in (Limb.LEFT_HAND.value, Limb.RIGHT_HAND.value):
        return None
    side = "left" if limb == Limb.LEFT_HAND.value else "right"
    vals: list[float] = []
    for f in frames:
        rx, ry, rc = limb_point(f.keypoints_raw, limb, RAW_PIVOT_CFG)
        if rc < 0.30 or rx <= 0 or ry <= 0:
            continue
        kp = f.keypoints_smooth or f.keypoints_raw
        elbow = kp.get(f"{side}_elbow")
        if elbow is None or elbow.conf < 0.30:
            continue
        d = float(np.hypot(rx - elbow.x, ry - elbow.y))
        if 5.0 <= d <= 200.0:
            vals.append(d)
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _jerk(points: np.ndarray, fps: float) -> np.ndarray:
    if len(points) < 4:
        return np.zeros(0, dtype=np.float64)
    vel = np.diff(points, axis=0) * fps
    acc = np.diff(vel, axis=0) * fps
    jerk = np.diff(acc, axis=0) * fps
    return np.hypot(jerk[:, 0], jerk[:, 1])


def _max_low_gap(confs: np.ndarray, threshold: float) -> int:
    max_run = 0
    cur = 0
    for c in confs:
        if c < threshold:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _limb_series(frames, fps: float) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    frame_ids = [f.frame for f in frames]
    frame_arr = np.asarray(frame_ids, dtype=np.int64)

    smooth_by_limb: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for limb in LIMBS:
        xs: list[float] = []
        ys: list[float] = []
        cs: list[float] = []
        for f in frames:
            kp = f.keypoints_smooth or f.keypoints_raw
            x, y, c = limb_point(kp, limb)
            xs.append(x)
            ys.append(y)
            cs.append(c)
        smooth_by_limb[limb] = (
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(cs, dtype=np.float64),
        )

    hand_hand_dist = np.full(len(frames), np.nan, dtype=np.float64)
    if frames:
        lx, ly, lc = smooth_by_limb[Limb.LEFT_HAND.value]
        rx, ry, rc = smooth_by_limb[Limb.RIGHT_HAND.value]
        both = (lc > 0) & (rc > 0)
        hand_hand_dist = np.where(both, np.hypot(lx - rx, ly - ry), np.nan)

    for limb in LIMBS:
        raw_xs: list[float] = []
        raw_ys: list[float] = []
        raw_cs: list[float] = []
        for f in frames:
            x, y, c = limb_point(f.keypoints_raw, limb, RAW_PIVOT_CFG)
            raw_xs.append(x)
            raw_ys.append(y)
            raw_cs.append(c)
        raw_x = np.asarray(raw_xs, dtype=np.float64)
        raw_y = np.asarray(raw_ys, dtype=np.float64)
        raw_c = np.asarray(raw_cs, dtype=np.float64)
        xarr, yarr, carr = smooth_by_limb[limb]
        raw_valid = raw_c > 0
        valid = carr > 0

        raw_step = np.full(len(frames), np.nan, dtype=np.float64)
        smooth_step = np.full(len(frames), np.nan, dtype=np.float64)
        residual = np.full(len(frames), np.nan, dtype=np.float64)
        reappearance = np.full(len(frames), np.nan, dtype=np.float64)
        if len(frames) >= 2:
            contiguous = np.diff(frame_arr) == 1
            raw_prev_valid = raw_valid[:-1] & raw_valid[1:] & contiguous
            smooth_prev_valid = valid[:-1] & valid[1:] & contiguous
            raw_step[1:] = np.where(raw_prev_valid, np.hypot(np.diff(raw_x), np.diff(raw_y)), np.nan)
            smooth_step[1:] = np.where(smooth_prev_valid, np.hypot(np.diff(xarr), np.diff(yarr)), np.nan)
        both_valid = raw_valid & valid
        residual = np.where(both_valid, np.hypot(raw_x - xarr, raw_y - yarr), np.nan)

        last_before_gap: tuple[float, float] | None = None
        in_gap = False
        for i in range(len(frames)):
            if valid[i]:
                if in_gap and last_before_gap is not None:
                    reappearance[i] = float(np.hypot(xarr[i] - last_before_gap[0], yarr[i] - last_before_gap[1]))
                last_before_gap = (float(xarr[i]), float(yarr[i]))
                in_gap = False
            elif last_before_gap is not None:
                in_gap = True

        pts_runs: list[np.ndarray] = []
        start = 0
        while start < len(frames):
            while start < len(frames) and not valid[start]:
                start += 1
            end = start
            while end < len(frames) and valid[end] and (end == start or frame_arr[end] - frame_arr[end - 1] == 1):
                end += 1
            if end - start > 0:
                pts_runs.append(np.stack([xarr[start:end], yarr[start:end]], axis=1))
            start = max(end, start + 1)
        finite_steps = smooth_step[np.isfinite(smooth_step)]
        finite_raw_steps = raw_step[np.isfinite(raw_step)]
        jerk = np.concatenate([_jerk(run, fps) for run in pts_runs if len(run) >= 4]) if pts_runs else np.zeros(0)
        path_len = float(finite_steps.sum()) if finite_steps.size else 0.0
        forearm_base = _forearm_baseline(frames, limb)
        observed_raw_jump = 0
        pose_unreliable = 0

        for i, f in enumerate(frames):
            q = getattr(f, "limb_quality", {}).get(limb)
            pose_state = getattr(q, "state", "observed" if carr[i] > 0 else "unknown")
            reliability = float(getattr(q, "reliability", carr[i]))
            reason = str(getattr(q, "reason", ""))
            gated = int(pose_state != "observed" and raw_c[i] > 0)
            pose_unreliable += int(pose_state != "observed" or carr[i] < 0.3)
            if pose_state == "observed" and np.isfinite(raw_step[i]) and raw_step[i] > 35:
                observed_raw_jump += 1

            forearm_len = np.nan
            forearm_ratio = np.nan
            inside = 0
            if limb in (Limb.LEFT_HAND.value, Limb.RIGHT_HAND.value):
                side = "left" if limb == Limb.LEFT_HAND.value else "right"
                kp = f.keypoints_smooth or f.keypoints_raw
                elbow = kp.get(f"{side}_elbow")
                if elbow is not None and elbow.conf >= 0.30 and raw_c[i] > 0:
                    forearm_len = float(np.hypot(raw_x[i] - elbow.x, raw_y[i] - elbow.y))
                    if forearm_base:
                        forearm_ratio = forearm_len / max(1.0, forearm_base)
                if raw_c[i] > 0:
                    inside = int(_inside_torso(f, raw_x[i], raw_y[i]))

            rows.append({
                "frame": frame_ids[i],
                "time_sec": f.time_sec,
                "limb": limb,
                "x": xarr[i],
                "y": yarr[i],
                "conf": carr[i],
                "step_px": "" if not np.isfinite(smooth_step[i]) else float(smooth_step[i]),
                "raw_x": raw_x[i],
                "raw_y": raw_y[i],
                "raw_conf": raw_c[i],
                "smooth_x": xarr[i],
                "smooth_y": yarr[i],
                "smooth_conf": carr[i],
                "raw_step_px": "" if not np.isfinite(raw_step[i]) else float(raw_step[i]),
                "smooth_step_px": "" if not np.isfinite(smooth_step[i]) else float(smooth_step[i]),
                "raw_vs_smooth_residual_px": "" if not np.isfinite(residual[i]) else float(residual[i]),
                "pose_state": pose_state,
                "pose_reliability": reliability,
                "gated_measurement": gated,
                "gate_reason": reason,
                "forearm_len_px": "" if not np.isfinite(forearm_len) else float(forearm_len),
                "forearm_len_ratio": "" if not np.isfinite(forearm_ratio) else float(forearm_ratio),
                "hand_hand_dist_px": "" if not np.isfinite(hand_hand_dist[i]) else float(hand_hand_dist[i]),
                "inside_torso_soft": inside,
                "reappearance_jump_px": "" if not np.isfinite(reappearance[i]) else float(reappearance[i]),
            })

        summary[limb] = {
            "coverage": float(valid.mean()) if len(valid) else 0.0,
            "mean_confidence": float(carr.mean()) if carr.size else 0.0,
            "conf_lt_0_2": float((carr < 0.2).mean()) if carr.size else 0.0,
            "conf_lt_0_3": float((carr < 0.3).mean()) if carr.size else 0.0,
            "conf_lt_0_5": float((carr < 0.5).mean()) if carr.size else 0.0,
            "path_length": path_len,
            "p95_step": float(np.percentile(finite_steps, 95)) if finite_steps.size else 0.0,
            "p99_step": float(np.percentile(finite_steps, 99)) if finite_steps.size else 0.0,
            "max_step": float(finite_steps.max()) if finite_steps.size else 0.0,
            "count_step_gt_20": int((finite_steps > 20).sum()),
            "count_step_gt_40": int((finite_steps > 40).sum()),
            "raw_p95_step": float(np.percentile(finite_raw_steps, 95)) if finite_raw_steps.size else 0.0,
            "raw_count_step_gt_35": int((finite_raw_steps > 35).sum()),
            "accepted_raw_jump_gt35": int(observed_raw_jump),
            "pose_unreliable_fraction": float(pose_unreliable / max(1, len(frames))),
            "p95_jerk": float(np.percentile(jerk, 95)) if jerk.size else 0.0,
            "p99_jerk": float(np.percentile(jerk, 99)) if jerk.size else 0.0,
            "max_consecutive_low_conf_gap": _max_low_gap(carr, 0.3),
        }
    return rows, summary


def _keypoint_stats(frames) -> dict[str, Any]:
    names = sorted({n for f in frames for n in f.keypoints_raw})
    out: dict[str, Any] = {}
    for name in names:
        raw_cs = []
        smooth_cs = []
        residuals = []
        jumps = []
        prev: Keypoint | None = None
        for f in frames:
            raw = f.keypoints_raw.get(name, Keypoint(0.0, 0.0, 0.0))
            sm = (f.keypoints_smooth or {}).get(name)
            raw_cs.append(float(raw.conf))
            if sm is not None:
                smooth_cs.append(float(sm.conf))
                if raw.conf > 0 and sm.conf > 0:
                    residuals.append(float(np.hypot(raw.x - sm.x, raw.y - sm.y)))
            if prev is not None and prev.conf > 0 and raw.conf > 0:
                jumps.append(float(np.hypot(raw.x - prev.x, raw.y - prev.y)))
            prev = raw
        r = np.asarray(raw_cs, dtype=np.float64)
        s = np.asarray(smooth_cs, dtype=np.float64)
        res = np.asarray(residuals, dtype=np.float64)
        j = np.asarray(jumps, dtype=np.float64)
        out[name] = {
            "raw_mean_conf": float(r.mean()) if r.size else 0.0,
            "raw_conf_lt_0_3": float((r < 0.3).mean()) if r.size else 0.0,
            "smooth_mean_conf": float(s.mean()) if s.size else 0.0,
            "raw_vs_smooth_residual_p95": float(np.percentile(res, 95)) if res.size else 0.0,
            "top_jump": float(j.max()) if j.size else 0.0,
        }
    return out


def _finger_distances(frames) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    specs = {
        "left_hand": HAND_MAIN_IDX_LEFT,
        "right_hand": HAND_MAIN_IDX_RIGHT,
    }
    for limb, mapping in specs.items():
        wrist_name = mapping["wrist"]
        for role in ("mcp_index", "mcp_middle", "tip_index"):
            name = mapping[role]
            vals = []
            rejected = 0
            for f in frames:
                kp = f.keypoints_smooth or f.keypoints_raw
                wrist = kp.get(wrist_name)
                finger = kp.get(name)
                if wrist is None or finger is None or wrist.conf <= 0 or finger.conf <= 0:
                    continue
                dist = float(np.hypot(finger.x - wrist.x, finger.y - wrist.y))
                vals.append(dist)
                if finger.conf >= 0.3 and dist > 60.0:
                    rejected += 1
                rows.append({
                    "frame": f.frame,
                    "time_sec": f.time_sec,
                    "limb": limb,
                    "finger": name,
                    "role": role,
                    "distance_px": dist,
                    "wrist_conf": wrist.conf,
                    "finger_conf": finger.conf,
                    "would_reject_distance_gate": int(finger.conf >= 0.3 and dist > 60.0),
                })
            arr = np.asarray(vals, dtype=np.float64)
            summary[f"{limb}:{name}"] = {
                "count": int(arr.size),
                "p95_distance_px": float(np.percentile(arr, 95)) if arr.size else 0.0,
                "p99_distance_px": float(np.percentile(arr, 99)) if arr.size else 0.0,
                "max_distance_px": float(arr.max()) if arr.size else 0.0,
                "distance_gate_reject_count": int(rejected),
            }
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _top_jumps(rows: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    candidates = [r for r in rows if r["step_px"] != ""]
    candidates.sort(key=lambda r: float(r["step_px"]), reverse=True)
    return candidates[:limit]


def _plot_limb_conf(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    for limb in LIMBS:
        rr = [r for r in rows if r["limb"] == limb]
        ax.plot([r["time_sec"] for r in rr], [r["conf"] for r in rr], label=limb)
    for y in (0.2, 0.3, 0.5):
        ax.axhline(y, color="black", linewidth=0.8, alpha=0.25)
    ax.set_xlabel("time, s")
    ax.set_ylabel("confidence")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_limb_step(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    for limb in LIMBS:
        rr = [r for r in rows if r["limb"] == limb and r["step_px"] != ""]
        ax.plot([r["time_sec"] for r in rr], [float(r["step_px"]) for r in rr], label=limb)
    ax.axhline(20, color="black", linewidth=0.8, alpha=0.25)
    ax.axhline(40, color="red", linewidth=0.8, alpha=0.25)
    ax.set_xlabel("time, s")
    ax.set_ylabel("step, px/frame")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_limb_jerk(frames, fps: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    for limb in LIMBS:
        pts = []
        times = []
        frame_ids = []
        for f in frames:
            kp = f.keypoints_smooth or f.keypoints_raw
            x, y, c = limb_point(kp, limb)
            if c <= 0:
                if len(pts) >= 4:
                    arr = np.asarray(pts, dtype=np.float64)
                    jerk = _jerk(arr, fps)
                    ax.plot(times[3:], jerk, label=limb)
                pts = []
                times = []
                frame_ids = []
                continue
            if frame_ids and f.frame - frame_ids[-1] != 1:
                if len(pts) >= 4:
                    arr = np.asarray(pts, dtype=np.float64)
                    jerk = _jerk(arr, fps)
                    ax.plot(times[3:], jerk, label=limb)
                pts = []
                times = []
                frame_ids = []
            pts.append((x, y))
            times.append(f.time_sec)
            frame_ids.append(f.frame)
        if len(pts) >= 4:
            arr = np.asarray(pts, dtype=np.float64)
            jerk = _jerk(arr, fps)
            ax.plot(times[3:], jerk, label=limb)
    ax.set_xlabel("time, s")
    ax.set_ylabel("jerk, px/s^3")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_finger_dist(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    for key in sorted({(r["limb"], r["finger"]) for r in rows}):
        rr = [r for r in rows if (r["limb"], r["finger"]) == key]
        ax.plot([r["time_sec"] for r in rr], [r["distance_px"] for r in rr], label=f"{key[0]}:{key[1]}")
    ax.axhline(60, color="red", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("time, s")
    ax.set_ylabel("finger-wrist distance, px")
    ax.legend(ncol=3, fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _crop_summary(crop_diag: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    if not crop_diag:
        return {}
    frames = np.asarray([d["frame"] for d in crop_diag], dtype=np.float64)
    scales = np.asarray([d["crop_scale"] for d in crop_diag], dtype=np.float64)
    widths = np.asarray([d["crop_width"] for d in crop_diag], dtype=np.float64)
    heights = np.asarray([d["crop_height"] for d in crop_diag], dtype=np.float64)
    edge = np.asarray([d["keypoints_near_edge"] for d in crop_diag], dtype=np.float64)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), dpi=140, sharex=True)
    axes[0].plot(frames, widths, label="crop width")
    axes[0].plot(frames, heights, label="crop height")
    axes[0].legend(fontsize=8)
    axes[1].plot(frames, scales, label="crop scale")
    axes[1].legend(fontsize=8)
    axes[2].plot(frames, edge, label="keypoints near edge")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("frame")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "bbox_crop_diagnostics.png")
    plt.close(fig)
    return {
        "frames": int(len(crop_diag)),
        "crop_scale_p95_step": float(np.percentile(np.abs(np.diff(scales)), 95)) if len(scales) > 1 else 0.0,
        "crop_width_p95": float(np.percentile(widths, 95)),
        "crop_height_p95": float(np.percentile(heights, 95)),
        "keypoints_near_edge_mean": float(edge.mean()),
        "keypoints_near_edge_max": float(edge.max()),
    }


def _contact_metrics(
    analysis: AttemptAnalysis,
    frames,
    limb_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    frame_ids = [f.frame for f in frames]
    if not frame_ids:
        return {}
    row_by_key = {(int(r["frame"]), r["limb"]): r for r in limb_rows}
    out: dict[str, Any] = {}
    all_windows = {"all": (min(frame_ids), max(frame_ids)), **REPORT_WINDOWS}

    for limb in LIMBS:
        limb_out: dict[str, Any] = {}
        segs = analysis.contacts.per_limb.get(limb, [])
        for win_name, (ws, we) in all_windows.items():
            ids = [f for f in frame_ids if ws <= f <= we]
            if not ids:
                continue
            state_by_frame = {f: "none" for f in ids}
            hold_by_frame: dict[int, str | None] = {f: None for f in ids}
            for seg in segs:
                s = max(ws, seg.start_frame)
                e = min(we, seg.end_frame)
                if e < s:
                    continue
                for fr in ids:
                    if s <= fr <= e:
                        state_by_frame[fr] = seg.state
                        hold_by_frame[fr] = seg.hold_id if seg.state == "contact" else None

            real_segments = [
                seg for seg in segs
                if seg.state == "contact"
                and seg.hold_id is not None
                and not (seg.end_frame < ws or seg.start_frame > we)
            ]
            switches = 0
            switches_unreliable = 0
            prev_hold = None
            for fr in ids:
                hid = hold_by_frame[fr]
                if hid is not None and prev_hold is not None and hid != prev_hold:
                    switches += 1
                    row = row_by_key.get((fr, limb), {})
                    unreliable = (
                        row.get("pose_state") != "observed"
                        or float(row.get("smooth_conf", 0.0) or 0.0) < 0.3
                    )
                    switches_unreliable += int(unreliable)
                if hid is not None:
                    prev_hold = hid

            low_pose_segments = 0
            for seg in real_segments:
                seg_ids = [fr for fr in ids if seg.start_frame <= fr <= seg.end_frame]
                if not seg_ids:
                    continue
                low = 0
                for fr in seg_ids:
                    row = row_by_key.get((fr, limb), {})
                    low += int(
                        row.get("pose_state") != "observed"
                        or float(row.get("smooth_conf", 0.0) or 0.0) < 0.3
                    )
                if low / max(1, len(seg_ids)) > 0.5:
                    low_pose_segments += 1

            delays: list[int] = []
            occluded_run_start: int | None = None
            for fr in ids:
                if state_by_frame[fr] == "occluded":
                    if occluded_run_start is None:
                        occluded_run_start = fr
                    continue
                if occluded_run_start is not None:
                    if state_by_frame[fr] == "contact":
                        delays.append(fr - occluded_run_start)
                    occluded_run_start = None

            flap_count = 0
            contact_segs = [
                seg for seg in real_segments
                if seg.hold_id is not None
            ]
            for a, b, c in zip(contact_segs, contact_segs[1:], contact_segs[2:]):
                if a.hold_id == c.hold_id and a.hold_id != b.hold_id:
                    if b.end_frame - b.start_frame + 1 <= 6:
                        flap_count += 1

            wrong_after_zero = 0
            for prev, cur in zip(ids[:-1], ids[1:]):
                if hold_by_frame[cur] is None:
                    continue
                row = row_by_key.get((prev, limb), {})
                prev_zero = (
                    row.get("pose_state") in {"unknown", "occluded", "rejected"}
                    or float(row.get("smooth_conf", 0.0) or 0.0) <= 0
                )
                wrong_after_zero += int(prev_zero)

            limb_out[win_name] = {
                "real_contact_segments_count": int(len(real_segments)),
                "real_hold_switch_count": int(switches),
                "real_hold_switch_count_when_pose_unreliable": int(switches_unreliable),
                "contact_segments_with_low_pose_frac_gt_0_5": int(low_pose_segments),
                "occluded_fraction": float(
                    sum(1 for fr in ids if state_by_frame[fr] == "occluded") / max(1, len(ids))
                ),
                "new_hold_reacquisition_delay_frames_mean": float(np.mean(delays)) if delays else 0.0,
                "new_hold_reacquisition_delay_frames_max": int(max(delays)) if delays else 0,
                "aba_flap_count": int(flap_count),
                "wrong_real_contact_after_zero_count": int(wrong_after_zero),
            }
        out[limb] = limb_out
    return out


def _top_jump_sheet(video: Path, top: list[dict[str, Any]], out: Path) -> None:
    if not video.exists() or not top:
        return
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return
    thumbs = []
    try:
        for row in top[:20]:
            frame_idx = int(row["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            text = f"f={frame_idx} {row['limb']} step={float(row['step_px']):.1f}"
            cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            thumbs.append(thumb)
    finally:
        cap.release()
    if not thumbs:
        return
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y = (i // cols) * 180
        x = (i % cols) * 320
        sheet[y:y + 180, x:x + 320] = thumb
    cv2.imwrite(str(out), sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--crop-diagnostics", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.analysis.read_text())
    analysis = AttemptAnalysis.from_dict(data)
    frames = _main_frames(analysis)
    fps = args.fps or analysis.fps

    limb_rows, limb_summary = _limb_series(frames, fps)
    keypoint_summary = _keypoint_stats(frames)
    finger_rows, finger_summary = _finger_distances(frames)
    contact_summary = _contact_metrics(analysis, frames, limb_rows)
    top = _top_jumps(limb_rows)

    _write_csv(args.out / "limb_timeseries.csv", limb_rows)
    _write_csv(args.out / "top_100_limb_jumps.csv", top)
    _write_csv(args.out / "finger_to_wrist_distance.csv", finger_rows)
    _plot_limb_conf(limb_rows, args.out / "limb_confidence_over_time.png")
    _plot_limb_step(limb_rows, args.out / "limb_displacement_over_time.png")
    _plot_limb_jerk(frames, fps, args.out / "limb_jerk_over_time.png")
    if finger_rows:
        _plot_finger_dist(finger_rows, args.out / "finger_to_wrist_distance.png")

    crop_diag = []
    crop_path = args.crop_diagnostics
    if crop_path is None:
        candidate = args.analysis.parent / "diagnostics" / "crop_diagnostics.json"
        if candidate.exists():
            crop_path = candidate
    if crop_path is not None and crop_path.exists():
        crop_diag = json.loads(crop_path.read_text())
    crop = _crop_summary(crop_diag, args.out)

    video = args.video or Path(analysis.video_path)
    _top_jump_sheet(video, top, args.out / "top_jumps_contact_sheet.jpg")

    summary = {
        "analysis": str(args.analysis),
        "fps": fps,
        "frames": len(frames),
        "limbs": limb_summary,
        "keypoints": keypoint_summary,
        "hand_finger_distances": finger_summary,
        "contacts": contact_summary,
        "crop": crop,
    }
    write_json(args.out / "pose_jitter_summary.json", summary)


if __name__ == "__main__":
    main()
