import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from pipelines.climb.inference import ClimbPipeline


def _crop_hold(frame: np.ndarray, bbox, pad: float = 0.20) -> np.ndarray:
    H, W = frame.shape[:2]
    if hasattr(bbox, "x1"):
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    else:
        x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    side = max(w, h) * (1.0 + pad)
    nx1 = int(max(0, round(cx - side / 2)))
    ny1 = int(max(0, round(cy - side / 2)))
    nx2 = int(min(W, round(cx + side / 2)))
    ny2 = int(min(H, round(cy + side / 2)))
    return frame[ny1:ny2, nx1:nx2]


def _letterbox(img: np.ndarray, S: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((S, S, 3), dtype=np.uint8)
    scale = S / max(h, w)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((S, S, 3), dtype=np.uint8)
    py = (S - nh) // 2
    px = (S - nw) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


@torch.inference_mode()
def color_probs_batch(pipeline: ClimbPipeline, crops: list[np.ndarray]) -> np.ndarray:
    from common.tta import classifier_tta_rot_flip

    if not crops:
        return np.zeros((0, len(pipeline.color_class_names)), dtype=np.float32)
    S = pipeline.color_image_size
    letter = np.stack([_letterbox(c, S) for c in crops], axis=0)
    batch = torch.from_numpy(letter).to(pipeline.device)
    batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
    batch = (batch - pipeline._mean) / pipeline._std
    chunks: list[torch.Tensor] = []
    cs = pipeline.color_chunk_size
    with torch.amp.autocast(pipeline.device.type, dtype=torch.float16):
        for s in range(0, batch.shape[0], cs):
            sub = batch[s:s + cs]
            if pipeline.color_tta:
                chunks.append(classifier_tta_rot_flip(
                    sub, pipeline.color_model, temperature=pipeline.color_temperature,
                ))
            else:
                scaled = pipeline.color_model(sub) / max(1e-3, pipeline.color_temperature)
                chunks.append(torch.softmax(scaled, dim=1))
    return torch.cat(chunks, dim=0).float().cpu().numpy()


def fit_temperature(
    pairs: list[tuple[str, np.ndarray]],
    class_names: list[str],
) -> float:
    if not pairs:
        return 1.0
    targets: list[int] = []
    logits: list[np.ndarray] = []
    for tc, p in pairs:
        if tc not in class_names:
            continue
        targets.append(class_names.index(tc))
        logits.append(np.log(np.clip(p, 1e-9, 1.0)))
    if not targets:
        return 1.0
    logits_t = torch.tensor(np.stack(logits, axis=0), dtype=torch.float32)
    tgt_t = torch.tensor(targets, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones([]))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        Tp = T.abs() + 1e-3
        nll = torch.nn.functional.cross_entropy(logits_t / Tp, tgt_t)
        nll.backward()
        return nll

    opt.step(closure)
    return float(T.abs().item() + 1e-3)


def confusion_matrix(
    pairs: list[tuple[str, np.ndarray]],
    class_names: list[str],
) -> dict:
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(class_names), dtype=np.float64))
    counts: dict[str, int] = defaultdict(int)
    for tc, p in pairs:
        if tc not in class_names:
            continue
        sums[tc] += p
        counts[tc] += 1
    matrix: dict[str, dict[str, float]] = {}
    for tc, s in sums.items():
        mean = s / max(1, counts[tc])
        matrix[tc] = {c: float(mean[i]) for i, c in enumerate(class_names)}
    return {"confusion_matrix": matrix, "per_target_counts": dict(counts)}


def collect_pairs_from_folders(
    pipeline: ClimbPipeline,
    crops_root: Path,
) -> list[tuple[str, np.ndarray]]:
    pairs: list[tuple[str, np.ndarray]] = []
    for cls_dir in sorted(p for p in Path(crops_root).iterdir() if p.is_dir()):
        target = cls_dir.name
        if target not in pipeline.color_class_names:
            continue
        crops: list[np.ndarray] = []
        for img_path in sorted(cls_dir.glob("*.jpg")):
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            crops.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        probs = color_probs_batch(pipeline, crops)
        for p in probs:
            pairs.append((target, p))
    return pairs


def run_calibration(
    maskformer_dir: str,
    color_weights: str,
    crops_root: Path,
    out_path: Path,
    color_model_config: str = "eva02_color",
    use_tta: bool = True,
) -> dict:
    from common.config import cfg

    ccfg = cfg.model_cfg(color_model_config)
    class_names = list(ccfg["class_names"])
    pipeline = ClimbPipeline(
        maskformer_dir=str(maskformer_dir),
        color_weights=str(color_weights),
        color_class_names=class_names,
        color_model_name=ccfg.get("backbone", "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"),
        color_image_size=int(ccfg.get("image_size", 448)),
        color_temperature=float(ccfg.get("color_temperature", 1.0)),
        use_sam_refine=False,
        use_tta=False,
        color_tta=use_tta,
    )

    pairs = collect_pairs_from_folders(pipeline, Path(crops_root))
    report = confusion_matrix(pairs, class_names)
    report["temperature"] = fit_temperature(pairs, class_names)

    counts_correct: Counter[str] = Counter()
    counts_total: Counter[str] = Counter()
    for tc, p in pairs:
        if tc not in class_names:
            continue
        pred = class_names[int(np.argmax(p))]
        counts_total[tc] += 1
        if pred == tc:
            counts_correct[tc] += 1
    report["pre_cal_accuracy_per_target"] = {
        c: counts_correct[c] / max(1, counts_total[c]) for c in counts_total
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
