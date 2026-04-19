import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common.config import cfg
from common.tiling import iter_tiles, merge_instances_by_mask_iou
from common.tta import detector_tta_hflip_scales
from common.types import PipelineMode, SegClass
from pipelines import register_pipeline

logger = logging.getLogger(__name__)

CLASS_NAMES = [SegClass.HOLD.value, SegClass.VOLUME.value]


def _read_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(path).convert("RGB"))


def _mask_bbox(mask: np.ndarray) -> list[float] | None:
    rows = mask.any(axis=1)
    if not rows.any():
        return None
    cols = mask.any(axis=0)
    y0 = int(rows.argmax())
    y1 = int(len(rows) - rows[::-1].argmax())
    x0 = int(cols.argmax())
    x1 = int(len(cols) - cols[::-1].argmax())
    return [float(x0), float(y0), float(x1), float(y1)]


class Mask2FormerDetector:
    def __init__(
        self,
        weights_dir: str | Path,
        device: str | torch.device = "cuda",
        detector_image_size: int = 1536,
        score_thr: float = 0.3,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        max_detections: int = 500,
        tile_if_over: int = 2048,
        tile_batch: int = 4,
        use_tta: bool = False,
    ) -> None:
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self.device = torch.device(device)
        self.processor = Mask2FormerImageProcessor.from_pretrained(weights_dir)
        self.detector = Mask2FormerForUniversalSegmentation.from_pretrained(weights_dir)
        self.detector.to(self.device).eval()
        self.detector_image_size = detector_image_size
        self.score_thr = score_thr
        self.mask_threshold = mask_threshold
        self.overlap_mask_area_threshold = overlap_mask_area_threshold
        self.max_detections = max_detections
        self.tile_if_over = tile_if_over
        self.tile_batch = tile_batch
        self.use_tta = use_tta

    @torch.inference_mode()
    def detect_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        if not images:
            return []

        target = self.detector_image_size
        resized_batch = []
        target_sizes: list[tuple[int, int]] = []
        crop_metas: list[tuple[int, int, int, int]] = []
        original_sizes: list[tuple[int, int]] = []
        for img in images:
            H, W = img.shape[:2]
            scale = target / max(H, W)
            nW, nH = int(round(W * scale)), int(round(H * scale))
            resized = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((target, target, 3), dtype=resized.dtype)
            py = (target - nH) // 2
            px = (target - nW) // 2
            canvas[py:py + nH, px:px + nW] = resized
            resized_batch.append(canvas)
            target_sizes.append((target, target))
            crop_metas.append((px, py, nW, nH))
            original_sizes.append((H, W))

        inputs = self.processor(images=resized_batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        with torch.amp.autocast(self.device.type, dtype=torch.float16):
            outputs = self.detector(pixel_values=pixel_values)

        per_image_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_thr,
            mask_threshold=self.mask_threshold,
            overlap_mask_area_threshold=self.overlap_mask_area_threshold,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        out_batch: list[list[dict]] = []
        for results, (px, py, nW, nH), (origH, origW) in zip(per_image_results, crop_metas, original_sizes):
            segmentation = results["segmentation"]
            segments = results["segments_info"]
            seg_cpu = segmentation.cpu().numpy() if isinstance(segmentation, torch.Tensor) else np.array(segmentation)

            instances: list[dict] = []
            for seg in segments:
                sid = seg["id"]
                label_id = int(seg["label_id"])
                score = float(seg["score"])
                if seg_cpu.ndim == 3:
                    mask = seg_cpu[sid].astype(bool)
                else:
                    mask = (seg_cpu == sid)
                cropped = mask[py:py + nH, px:px + nW]
                mask = cv2.resize(
                    cropped.astype(np.uint8), (origW, origH), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                bbox = _mask_bbox(mask)
                if bbox is None:
                    continue
                instances.append({
                    "mask": mask,
                    "bbox": bbox,
                    "class": label_id,
                    "class_name": CLASS_NAMES[label_id] if label_id < len(CLASS_NAMES) else str(label_id),
                    "score": score,
                })
            instances.sort(key=lambda r: r["score"], reverse=True)
            out_batch.append(instances[: self.max_detections])
        return out_batch

    def detect(self, image: np.ndarray) -> list[dict]:
        H, W = image.shape[:2]
        if max(H, W) > self.tile_if_over:
            return self._detect_tiled(image)
        if self.use_tta:
            preds = detector_tta_hflip_scales(image, self.detect_batch, scales=(0.85, 1.0, 1.15), hflip=True)
            return merge_instances_by_mask_iou(preds, iou_thr=0.5, containment_thr=0.7, union=False)
        return self.detect_batch([image])[0]

    def _detect_tiled(self, image: np.ndarray) -> list[dict]:
        H, W = image.shape[:2]
        tiles = list(iter_tiles(image, size=self.detector_image_size, overlap=0.25))
        if not tiles:
            return []

        all_instances: list[dict] = []
        TB = max(1, self.tile_batch)
        for start in range(0, len(tiles), TB):
            chunk = tiles[start: start + TB]
            chunk_results = self.detect_batch([t.image for t in chunk])
            for tile, tile_inst in zip(chunk, chunk_results):
                for inst in tile_inst:
                    full_mask = np.zeros((H, W), dtype=bool)
                    tm = inst["mask"]
                    full_mask[tile.y0: tile.y0 + tm.shape[0], tile.x0: tile.x0 + tm.shape[1]] = tm
                    x0, y0, x1, y1 = inst["bbox"]
                    all_instances.append({
                        **inst,
                        "mask": full_mask,
                        "bbox": [x0 + tile.x0, y0 + tile.y0, x1 + tile.x0, y1 + tile.y0],
                    })
        return merge_instances_by_mask_iou(all_instances, iou_thr=0.7)


@register_pipeline("mask2former", PipelineMode.INFERENCE)
def run_inference(model_name: str, weights: str, output: str, image_dir: str, preview: bool = False):
    device = torch.device(cfg.torch.device)
    mcfg = cfg.model_cfg(model_name)

    detector = Mask2FormerDetector(
        weights_dir=weights,
        device=device,
        detector_image_size=mcfg.get("detector_image_size", 1536),
        score_thr=mcfg.get("score_thr", 0.3),
        mask_threshold=mcfg.get("mask_threshold", 0.5),
        overlap_mask_area_threshold=mcfg.get("overlap_mask_area_threshold", 0.8),
        max_detections=mcfg.get("max_detections", 500),
        tile_if_over=mcfg.get("tile_if_over", 2048),
        tile_batch=mcfg.get("tile_batch", 4),
        use_tta=mcfg.get("use_tta", False),
    )

    test_dir = Path(image_dir)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in extensions)

    all_results: dict[str, list[dict]] = {}
    for img_path in tqdm(image_paths, desc="Inference"):
        img = _read_rgb(img_path)
        instances = detector.detect(img)
        per_image = [
            {
                "bbox": [float(x) for x in inst["bbox"]],
                "class": inst["class"],
                "class_name": inst["class_name"],
                "score": float(inst["score"]),
            }
            for inst in instances
        ]
        all_results[img_path.name] = per_image

        if preview:
            overlay = img.copy()
            for inst in instances:
                m = inst["mask"]
                overlay[m] = (overlay[m] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                x0, y0, x1, y1 = [int(v) for v in inst["bbox"]]
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
            Image.fromarray(overlay).save(out_dir / img_path.name)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved %d results to %s", len(image_paths), out_dir)
