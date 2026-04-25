from __future__ import annotations

import argparse
import io
import json
import logging
import mimetypes
import os
import posixpath
import re
import socketserver
import sys
import threading
import urllib.parse
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as cocomask

STATIC_DIR = Path(__file__).parent / "viz_static"

class Attempt:

    def __init__(self, aid: str, analysis_path: Path, video_path: Path) -> None:
        self.id = aid
        self.label = aid
        self.analysis_path = analysis_path
        self.video_path = video_path
        self.analysis: dict = {}
        self.holds_viz: list = []
        self.pose_frames: list = []
        self.contacts: dict = {}
        self.events: dict = {}
        self.summary: dict = {}
        self.route_summary: dict = {}
                                                                             
        self.beta_video_path: Path | None = None
        self.beta_summary: dict | None = None
        self.keyframe_cache: "OrderedDict[int, bytes]" = OrderedDict()
        self.keyframe_cache_max = 60
        self.lock = threading.Lock()

class PhotoEntry:

    def __init__(self, pid: str, label: str, image_path: Path,
                 width: int, height: int, holds: list[dict]) -> None:
        self.id = pid
        self.label = label
        self.image_path = image_path
        self.width = width
        self.height = height
        self.holds = holds

STATE: dict = {
    "attempts": OrderedDict(),                   
    "photos":   OrderedDict(),                                                
    "default":  None,
}

def rle_to_polygons(rle_dict) -> list[list[list[int]]]:
    if not rle_dict:
        return []
    rle = dict(rle_dict)
    if isinstance(rle.get("counts"), str):
        rle = {**rle, "counts": rle["counts"].encode("ascii")}
    m = cocomask.decode(rle).astype(np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[list[int]]] = []
    for c in contours:
        if c.shape[0] < 3:
            continue
                                                                               
        eps = max(1.0, 0.005 * cv2.arcLength(c, True))
        c2 = cv2.approxPolyDP(c, eps, True)
        pts = c2.squeeze(axis=1).astype(int).tolist()
        polys.append(pts)
    return polys

def preprocess_photo_holds(raw_holds: list[dict]) -> list[dict]:
    out: list[dict] = []
    for h in raw_holds:
        polys = rle_to_polygons(h.get("mask_rle"))
        out.append({
            "id": h.get("id") or f"h_{len(out):03d}",
            "bbox": [float(x) for x in h.get("bbox", [0, 0, 0, 0])],
            "polygons": polys,
            "color": h.get("color", "UNKNOWN"),
            "color_conf": float(h.get("color_conf", 0.0)),
            "type": h.get("type", "unknown"),
            "type_conf": float(h.get("type_conf", 0.0)),
            "seg_class": h.get("class_name", "hold"),
            "det_conf": float(h.get("det_conf", 0.0)),
            "sam_iou": float(h.get("sam_iou", 0.0)),
            "fill_ratio": float(h.get("fill_ratio", 0.0)),
        })
    return out

def load_photo_set(json_path: Path) -> list[PhotoEntry]:
    payload = json.loads(json_path.read_text())
    raw_photos = payload.get("photos", [])
    images_dir = json_path.parent / "images"
    out: list[PhotoEntry] = []
    for p in raw_photos:
        img_file = p.get("image_file")
        if not img_file:
            continue
        ipath = images_dir / img_file
        if not ipath.exists():
            print(f"[viz] photo image missing, skipping: {ipath}", file=sys.stderr)
            continue
        out.append(PhotoEntry(
            pid=str(p.get("id") or ipath.stem),
            label=str(p.get("label") or p.get("id") or ipath.stem),
            image_path=ipath,
            width=int(p.get("width", 0)),
            height=int(p.get("height", 0)),
            holds=preprocess_photo_holds(p.get("holds", [])),
        ))
    return out

def preprocess_holds(analysis: dict) -> list[dict]:
    holds = analysis.get("route", {}).get("holds", [])
    out: list[dict] = []
    for h in holds:
        polys = rle_to_polygons(h.get("mask_rle"))
        out.append({
            "id": h["physical_track_id"],
            "bbox": [float(x) for x in (h["bbox"] if isinstance(h["bbox"], list) else
                [h["bbox"]["x1"], h["bbox"]["y1"], h["bbox"]["x2"], h["bbox"]["y2"]])],
            "center": list(h.get("center", [0, 0])),
            "color": h.get("color_label_temporal") or h.get("color_label_raw") or "UNKNOWN",
            "color_conf": float(h.get("color_conf_temporal") or h.get("color_conf_raw") or 0.0),
            "type": h.get("type_label") or "unknown",
            "type_conf": float(h.get("type_conf") or 0.0),
            "route_state": h.get("route_state", "unknown"),
            "route_score": float(h.get("route_score", 0.0)),
            "seg_class": h.get("seg_class", "hold"),
            "polygons": polys,
            "frames_seen": h.get("frames_seen", []),
            "det_conf_max": float(h.get("det_conf_max", 0.0)),
            "color_entropy": float(h.get("color_entropy", 0.0)) if h.get("color_entropy") is not None else 0.0,
        })
    return out

def preprocess_pose(analysis: dict) -> list[dict]:
    tracks = analysis.get("pose_tracks", [])
    if not tracks:
        return []
    main = next((t for t in tracks if t.get("is_main_climber")), tracks[0])
    out: list[dict] = []
    for f in main.get("frames", []):
        kp_source = f.get("keypoints_smooth") or f.get("keypoints_raw") or {}
        kps: dict[str, list[float]] = {}
        for name, v in kp_source.items():
            if isinstance(v, list) and len(v) >= 3 and float(v[2]) > 0.15:
                kps[name] = [float(v[0]), float(v[1]), float(v[2])]
        out.append({
            "frame": int(f["frame"]),
            "time_sec": float(f["time_sec"]),
            "bbox": [float(x) for x in (f["bbox"] if isinstance(f["bbox"], list) else
                [f["bbox"]["x1"], f["bbox"]["y1"], f["bbox"]["x2"], f["bbox"]["y2"]])] if f.get("bbox") else None,
            "keypoints": kps,
        })
    return out

def _bbox_as_list(b):
    if isinstance(b, list):
        return [float(x) for x in b]
    if isinstance(b, dict):
        return [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])]
    return None

def build_summary(analysis: dict) -> dict:
    route = analysis.get("route", {})
    holds = route.get("holds", [])
    states = {"core": 0, "possible": 0, "rejected": 0, "unknown": 0}
    for h in holds:
        states[h.get("route_state", "unknown")] = states.get(h.get("route_state", "unknown"), 0) + 1
    return {
        "video_id": analysis.get("video_id"),
        "video_path": analysis.get("video_path"),
        "target_color": analysis.get("target_color"),
        "fps": float(analysis.get("fps", 25.0)),
        "duration_sec": float(analysis.get("duration_sec", 0.0)),
        "frame_count": int(analysis.get("frame_count", 0)),
        "holds_total": len(holds),
        "holds_by_state": states,
        "metrics": analysis.get("metrics", {}),
    }

def extract_keyframe(video_path: Path, frame_idx: int, quality: int = 82) -> bytes:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"frame {frame_idx} not readable")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()

def keyframe_bytes(att: Attempt, frame_idx: int) -> bytes:
    with att.lock:
        if frame_idx in att.keyframe_cache:
            att.keyframe_cache.move_to_end(frame_idx)
            return att.keyframe_cache[frame_idx]
    data = extract_keyframe(att.video_path, frame_idx)
    with att.lock:
        att.keyframe_cache[frame_idx] = data
        att.keyframe_cache.move_to_end(frame_idx)
        while len(att.keyframe_cache) > att.keyframe_cache_max:
            att.keyframe_cache.popitem(last=False)
    return data

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

class Handler(BaseHTTPRequestHandler):
                                        
    def log_message(self, fmt: str, *args) -> None:
        logging.getLogger("viz").info("%s - - %s", self.client_address[0], fmt % args)

    def _send_json(self, obj, status: int = 200) -> None:
        payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200,
                    cache: str = "public, max-age=3600") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, data: str, content_type: str, status: int = 200) -> None:
        b = data.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(b)

    def _send_error(self, status: int, msg: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        body = msg.encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_static(self, rel_path: str) -> None:
                                 
        rel = posixpath.normpath("/" + rel_path).lstrip("/")
        fpath = (STATIC_DIR / rel).resolve()
        if not str(fpath).startswith(str(STATIC_DIR.resolve())) or not fpath.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, f"not found: {rel_path}")
            return
        ctype, _ = mimetypes.guess_type(str(fpath))
        if ctype is None:
            ctype = "application/octet-stream"
        data = fpath.read_bytes()
        self._send_bytes(data, ctype, cache="no-cache")

    def _serve_video(self, att: "Attempt") -> None:
        self._serve_video_path(att.video_path)

    def _serve_video_path(self, vpath: Path | None) -> None:
        if not vpath or not vpath.exists():
            self._send_error(HTTPStatus.NOT_FOUND, "video not available")
            return
        size = vpath.stat().st_size
        rng = self.headers.get("Range")
        ctype = "video/mp4"
        if not rng:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(vpath, "rb") as f:
                self._stream_copy(f, 0, size)
            return
        m = _RANGE_RE.match(rng.strip())
        if not m:
            self._send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "bad Range")
            return
        start_s, end_s = m.group(1), m.group(2)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(vpath, "rb") as f:
            f.seek(start)
            self._stream_copy(f, 0, length)

    def _stream_copy(self, f, offset: int, length: int) -> None:
                                                                           
        remaining = length
        chunk = 64 * 1024
        try:
            while remaining > 0:
                buf = f.read(min(chunk, remaining))
                if not buf:
                    break
                self.wfile.write(buf)
                remaining -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_HEAD(self) -> None:
                                                           
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        att, sub = _match_attempt_path(path)
        if att is not None and sub == "/video":
            size = att.video_path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return
        try:
            self.do_GET()
        except Exception:
            pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
                return
            if path in ("/style.css", "/app.js", "/favicon.ico"):
                self._serve_static(path.lstrip("/"))
                return
            if path == "/api/attempts":
                self._send_json([
                    {"id": a.id, "label": a.label,
                     "video_id": a.summary.get("video_id"),
                     "target_color": a.summary.get("target_color"),
                     "holds_total": a.summary.get("holds_total", 0),
                     "holds_by_state": a.summary.get("holds_by_state", {}),
                     "has_beta": a.beta_video_path is not None}
                    for a in STATE["attempts"].values()
                ])
                return

            if path == "/api/photos":
                self._send_json([
                    {"id": p.id, "label": p.label,
                     "width": p.width, "height": p.height,
                     "n_holds": len(p.holds),
                     "by_class": _photo_class_counts(p.holds)}
                    for p in STATE["photos"].values()
                ])
                return
            if path.startswith("/api/photo/"):
                                                
                rest = path[len("/api/photo/"):]
                if "/" not in rest:
                    self._send_error(HTTPStatus.NOT_FOUND, "expected /api/photo/<id>/{image,holds}")
                    return
                pid, sub = rest.split("/", 1)
                ph = STATE["photos"].get(pid)
                if ph is None:
                    self._send_error(HTTPStatus.NOT_FOUND, f"unknown photo id: {pid}")
                    return
                if sub == "image":
                    ctype, _ = mimetypes.guess_type(str(ph.image_path))
                    self._send_bytes(ph.image_path.read_bytes(),
                                     ctype or "image/jpeg",
                                     cache="public, max-age=86400")
                    return
                if sub == "holds":
                    self._send_json({
                        "id": ph.id, "label": ph.label,
                        "width": ph.width, "height": ph.height,
                        "holds": ph.holds,
                    })
                    return
                self._send_error(HTTPStatus.NOT_FOUND, f"unknown sub-path: {sub}")
                return

            att, sub = _match_attempt_path(path)
            if att is not None:
                if sub == "/summary":   self._send_json(att.summary); return
                if sub == "/holds":     self._send_json(att.holds_viz); return
                if sub == "/pose":      self._send_json(att.pose_frames); return
                if sub == "/contacts":  self._send_json(att.contacts); return
                if sub == "/events":    self._send_json(att.events); return
                if sub == "/route":     self._send_json(att.route_summary); return
                if sub.startswith("/keyframe/"):
                    try:
                        fi = int(sub.rsplit("/", 1)[1])
                    except ValueError:
                        self._send_error(HTTPStatus.BAD_REQUEST, "bad frame index")
                        return
                    try:
                        data = keyframe_bytes(att, fi)
                    except Exception as e:
                        self._send_error(HTTPStatus.NOT_FOUND, str(e))
                        return
                    self._send_bytes(data, "image/jpeg", cache="public, max-age=86400")
                    return
                if sub == "/video":
                    self._serve_video(att)
                    return
                if sub == "/beta_video":
                    if att.beta_video_path is None or not att.beta_video_path.exists():
                        self._send_error(HTTPStatus.NOT_FOUND,
                                         f"no synthetic_beta.mp4 for {att.id}")
                        return
                    self._serve_video_path(att.beta_video_path)
                    return
                if sub == "/beta_summary":
                    self._send_json(att.beta_summary or {})
                    return
                self._send_error(HTTPStatus.NOT_FOUND, f"unknown sub-path: {sub}")
                return

            if path.startswith("/"):
                self._serve_static(path.lstrip("/"))
                return
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as e:                                
            logging.getLogger("viz").exception("handler failed: %s", e)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

def _photo_class_counts(holds: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for h in holds:
        c = h.get("seg_class", "hold")
        out[c] = out.get(c, 0) + 1
    return out

def _match_attempt_path(path: str):
    if not path.startswith("/api/"):
        return None, None
    rest = path[len("/api/"):]
    if not rest:
        return None, None
                                                                             
    slash = rest.find("/")
    aid = rest if slash < 0 else rest[:slash]
    sub = "" if slash < 0 else rest[slash:]
    att = STATE["attempts"].get(aid)
    if att is None:
        return None, None
    return att, sub

class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

def load_analysis(analysis_path: Path) -> dict:
    return json.loads(analysis_path.read_text())

_AID_SAFE_RE = re.compile(r"[^a-zA-Z0-9_\-]")

def _slugify(s: str) -> str:
    s = _AID_SAFE_RE.sub("_", s)
    return s.strip("_") or "attempt"

def _derive_label(path: Path) -> str:
    parent = path.parent.name or path.stem
    return parent

def _parse_attempt_spec(spec: str) -> tuple[str | None, Path]:
    for sep in ("=", ":"):
        if sep in spec and not spec.startswith(("/", ".", "~")):
            head, tail = spec.split(sep, 1)
            if "/" in head or "\\" in head:
                continue
            return head, Path(tail)
    return None, Path(spec)

def _resolve_video(analysis_path: Path, json_video: str | None,
                   override: Path | None) -> Path | None:
    if override and override.exists():
        return override
    canon = analysis_path.parent / "video.mp4"
    if canon.exists():
        return canon
    if json_video:
        bn = analysis_path.parent / Path(json_video).name
        if bn.exists():
            return bn
        jp = Path(json_video)
        if jp.exists():
            return jp
    return None

def load_attempt(aid: str, analysis_path: Path, video_override: Path | None = None) -> Attempt:
    if not analysis_path.exists():
        raise FileNotFoundError(str(analysis_path))
    analysis = load_analysis(analysis_path)
    vpath = _resolve_video(analysis_path, analysis.get("video_path"), video_override)
    if vpath is None:
        raise FileNotFoundError(
            f"video for {aid} not found.\n"
            f"  Drop the source video as {analysis_path.parent / 'video.mp4'} or "
            f"keep the path '{analysis.get('video_path')}' resolvable from cwd."
        )
    att = Attempt(aid=aid, analysis_path=analysis_path, video_path=vpath)
    att.label = aid
                                                                     
    bv = analysis_path.parent / "synthetic_beta.mp4"
    if bv.is_file():
        att.beta_video_path = bv
    bs = analysis_path.parent / "synthesis_summary.json"
    if bs.is_file():
        try:
            att.beta_summary = json.loads(bs.read_text())
        except Exception:
            pass
    att.analysis = analysis
    att.holds_viz = preprocess_holds(analysis)
    att.pose_frames = preprocess_pose(analysis)
    att.contacts = analysis.get("contacts", {})
    att.events = {
        "move_events": analysis.get("move_events", []),
        "readjustments": analysis.get("readjustments", []),
        "hesitations": analysis.get("hesitations", []),
    }
    att.summary = build_summary(analysis)
    route = analysis.get("route", {})
    att.route_summary = {
        "target_color": route.get("target_color"),
        "total_keyframes": int(route.get("total_keyframes", 0) or 0),
        "holds_by_state": att.summary["holds_by_state"],
    }
    return att

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attempt", action="append", default=[],
                    help="Attempt spec: either PATH or LABEL:PATH. Repeat for multiple attempts.")
    ap.add_argument("--analysis", type=Path, default=None,
                    help="Back-compat alias for a single --attempt PATH.")
    ap.add_argument("--video", type=Path, default=None,
                    help="Optional override for the single-attempt case.")
    ap.add_argument("--photos", type=Path, default=None,
                    help="Path to a photos.json produced by tools/infer_photo_holds.py. "
                         "Photos populate the Hold-level tab; if omitted, Hold-level "
                         "falls back to per-attempt video keyframes.")
    ap.add_argument("--attempts-root", type=Path, default=Path("attempts"),
                    help="Auto-discover attempts: every subdir containing "
                         "analysis.json becomes an attempt (label = dir name). "
                         "Used only when no explicit --attempt/--analysis is given.")
    ap.add_argument("--photos-root", type=Path, default=Path("photos"),
                    help="Auto-discover photo set: <root>/photos.json is loaded "
                         "if it exists. Used only when --photos is not given.")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[viz] %(message)s")

    raw: list[tuple[str | None, Path, Path | None]] = []
    for spec in args.attempt:
        label, p = _parse_attempt_spec(spec)
        raw.append((label, p, None))
    if args.analysis is not None:
        raw.append((None, args.analysis, args.video))

    if not raw and args.attempts_root.is_dir():
        discovered: list[tuple[str, Path]] = []
        for sub in sorted(args.attempts_root.iterdir()):
            cand = sub / "analysis.json"
            if cand.is_file():
                discovered.append((sub.name, cand))
        if discovered:
            print(f"[viz] auto-discovered {len(discovered)} attempt(s) under "
                  f"{args.attempts_root}/")
            for label, p in discovered:
                raw.append((label, p, None))

    if not raw:
        print(f"error: no attempts found.\n"
              f"  Either pass --attempt PATH (or --analysis PATH),\n"
              f"  or place per-attempt subdirs (each with analysis.json) under "
              f"{args.attempts_root}/", file=sys.stderr)
        sys.exit(1)

    used_ids: set[str] = set()
    for label, analysis_path, video_override in raw:
        base = _slugify(label) if label else _slugify(_derive_label(analysis_path))
        aid = base
        i = 2
        while aid in used_ids:
            aid = f"{base}-{i}"; i += 1
        used_ids.add(aid)
        print(f"[viz] loading attempt '{aid}' ← {analysis_path}")
        try:
            att = load_attempt(aid, analysis_path, video_override=video_override)
        except Exception as e:
            print(f"[viz] error loading {analysis_path}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[viz]   video: {att.video_path}")
        print(f"[viz]   {len(att.holds_viz)} holds · {len(att.pose_frames)} pose frames")
        STATE["attempts"][aid] = att

    STATE["default"] = next(iter(STATE["attempts"].keys()))

    photos_json: Path | None = args.photos
    if photos_json is None:
        candidate = args.photos_root / "photos.json"
        if candidate.is_file():
            photos_json = candidate
            print(f"[viz] auto-discovered photo set at {candidate}")
    if photos_json is not None:
        if not photos_json.exists():
            print(f"error: photos json not found: {photos_json}", file=sys.stderr)
            sys.exit(1)
        try:
            entries = load_photo_set(photos_json)
        except Exception as e:
            print(f"error loading photo set: {e}", file=sys.stderr)
            sys.exit(1)
        for ph in entries:
            STATE["photos"][ph.id] = ph
        print(f"[viz] photo-set: {len(entries)} photo(s): {', '.join(STATE['photos'].keys())}")

    print(f"[viz] ready: {len(STATE['attempts'])} attempt(s): "
          f"{', '.join(STATE['attempts'].keys())}")
    print(f"[viz] open: http://{args.host}:{args.port}")

    with ThreadingServer((args.host, args.port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[viz] bye")

if __name__ == "__main__":
    main()
