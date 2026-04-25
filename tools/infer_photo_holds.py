from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.inference.pipeline import ClimbPipeline
from pipeline.common.masks import mask_to_rle

_AID_RE_BAD = "[^a-zA-Z0-9_\\-]"

def _slug(s: str) -> str:
    import re
    out = re.sub(_AID_RE_BAD, "_", s).strip("_")
    return out or "photo"

def _read_rgb(p: Path) -> np.ndarray:
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot decode {p}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _process_one(pipeline: ClimbPipeline, img: np.ndarray, keep_masks: bool = True) -> list[dict]:
    H, W = img.shape[:2]
    if max(H, W) > pipeline.cfg.tile_if_over:
        instances = pipeline._detect_tiled(img)
    elif pipeline.cfg.use_tta:
        instances = pipeline._detect_with_tta(img)
    else:
        instances = pipeline._detect(img)
    instances = pipeline._refine_masks(img, instances)
    instances = pipeline._classify_colors(img, instances)
    instances = pipeline._classify_types(img, instances)

    out: list[dict] = []
    for i, inst in enumerate(instances):
        m = inst.get("mask")
        rle = mask_to_rle(m) if (keep_masks and m is not None) else None
        out.append({
            "id": f"h_{i:03d}",
            "bbox": [float(x) for x in inst["bbox"]],
            "class": int(inst["class"]),
            "class_name": inst["class_name"],
            "det_conf": float(inst["score"]),
            "color": inst.get("color", "UNKNOWN"),
            "color_conf": float(inst.get("color_conf", 0.0)),
            "color_probs": list(inst.get("color_probs", [])),
            "type": inst.get("type", "unknown"),
            "type_conf": float(inst.get("type_conf", 0.0)),
            "type_probs": list(inst.get("type_probs", [])),
            "sam_iou": float(inst.get("sam_iou", 0.0)),
            "fill_ratio": float(inst.get("fill_ratio", 0.0)),
            "mask_rle": rle,
        })
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", default=[], required=True,
                    help="Path to an input image. Repeat for multiple images.")
    ap.add_argument("--label", action="append", default=[],
                    help="Optional label for each image (zip-aligned with --image). "
                         "Defaults to the filename stem.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output directory (will be created).")
    ap.add_argument("--maskformer-dir", type=Path,
                    default=Path("runs/maskformer_stage2/best_ema"))
    ap.add_argument("--color-weights", type=Path,
                    default=Path("runs/color_wayup/best.safetensors"))
    ap.add_argument("--type-weights", type=Path,
                    default=Path("runs/eva02_type/best_ema.safetensors"))
    ap.add_argument("--score-thr", type=float, default=0.20,
                    help="Recall vs precision tradeoff at the M2F output.")
    ap.add_argument("--no-tta", action="store_true",
                    help="Skip multi-scale + hflip TTA (3-6× faster, slight recall hit).")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--type-tta", action="store_true",
                    help="Enable rot+hflip TTA on the type classifier (slower, marginal accuracy gain).")
    ap.add_argument("--max-detections", type=int, default=500,
                    help="Cap on instances per photo (raise for very dense competition walls).")
    ap.add_argument("--max-quality", action="store_true",
                    help="Best-effort settings: TTA on, SAM on, type-TTA on, score-thr=0.10, "
                         "max-detections=700. Slower but recall-oriented.")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.max_quality:
        args.no_tta = False
        args.no_sam = False
        args.type_tta = True
        args.score_thr = min(args.score_thr, 0.10)
        args.max_detections = max(args.max_detections, 700)

    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    images_dir.mkdir(exist_ok=True)

    image_paths = [Path(p) for p in args.image]
    labels = list(args.label) if args.label else []
    while len(labels) < len(image_paths):
        labels.append(image_paths[len(labels)].stem)

    color_weights = args.color_weights
    if not color_weights.exists():
        legacy = Path("runs/color/best_ema.safetensors")
        if legacy.exists():
            color_weights = legacy
            print(f"[infer-photo] color fine-tune missing → using {legacy}")

    print(f"[infer-photo] loading ClimbPipeline (TTA={not args.no_tta}, SAM={not args.no_sam}, "
          f"type_TTA={args.type_tta}, score_thr={args.score_thr}, max_det={args.max_detections})")
    pipeline_kwargs = dict(
        maskformer_dir=str(args.maskformer_dir),
        color_weights=str(color_weights),
        color_temperature=0.649 if args.color_weights.name == "best.safetensors" else 1.0,
        use_sam_refine=not args.no_sam,
        use_tta=not args.no_tta,
        score_thr=args.score_thr,
        max_detections=args.max_detections,
        type_tta=args.type_tta,
        device=args.device,
    )
    if args.type_weights and args.type_weights.exists():
        pipeline_kwargs["type_weights"] = str(args.type_weights)
    pipeline = ClimbPipeline(**pipeline_kwargs)

    photo_records: list[dict] = []
    used_ids: set[str] = set()
    for src_path, label in zip(image_paths, labels):
        if not src_path.exists():
            print(f"[infer-photo] skip (missing): {src_path}", file=sys.stderr)
            continue
        base = _slug(label)
        aid = base
        i = 2
        while aid in used_ids:
            aid = f"{base}-{i}"; i += 1
        used_ids.add(aid)

        ext = src_path.suffix.lower() or ".jpg"
        dst_img = images_dir / f"{aid}{ext}"
        if not dst_img.exists() or dst_img.stat().st_size != src_path.stat().st_size:
            shutil.copyfile(src_path, dst_img)

        img = _read_rgb(src_path)
        H, W = img.shape[:2]
        print(f"[infer-photo] {aid}  {W}x{H}  {src_path}")
        t0 = time.time()
        holds = _process_one(pipeline, img, keep_masks=True)
        dt = time.time() - t0
        n_hold = sum(1 for h in holds if h["class_name"] == "hold")
        n_vol  = sum(1 for h in holds if h["class_name"] == "volume")
        print(f"[infer-photo]   {len(holds)} instances ({n_hold} hold + {n_vol} volume) in {dt:.1f}s")

        photo_records.append({
            "id": aid,
            "label": label,
            "image_file": dst_img.name,
            "source_path": str(src_path),
            "width": W, "height": H,
            "holds": holds,
            "infer_seconds": round(dt, 2),
        })

    out_json = args.output / "photos.json"
    out_json.write_text(json.dumps({"photos": photo_records}, ensure_ascii=False))
    print(f"[infer-photo] wrote {out_json} ({out_json.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
