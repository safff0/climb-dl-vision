import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as cocomask
from tqdm import tqdm

from common.config import cfg
from common.tiling import iter_tiles, merge_instances_by_mask_iou
from common.tta import classifier_tta_rot_flip, detector_tta_hflip_scales
from pipelines.mask2former.inference import Mask2FormerDetector

logger = logging.getLogger(__name__)

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "Black":  (40,  40,  40),
    "Blue":   (220, 80,  0),
    "Gray":   (160, 160, 160),
    "Green":  (0,   200, 0),
    "Orange": (0,   140, 255),
    "Pink":   (180, 105, 255),
    "Purple": (200, 0,   200),
    "Red":    (0,   0,   220),
    "White":  (230, 230, 230),
    "Yellow": (0,   230, 230),
}


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


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off: y_off + new_h, x_off: x_off + new_w] = resized
    return canvas


def _mask_to_rle(m: np.ndarray) -> dict:
    rle = cocomask.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


class ClimbPipeline:
    def __init__(
        self,
        maskformer_dir: str | Path,
        color_weights: str | Path | None = None,
        color_model_name: str = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        color_class_names: list[str] | None = None,
        color_image_size: int = 448,
        type_weights: str | Path | None = None,
        type_model_name: str = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        type_class_names: list[str] | None = None,
        type_image_size: int = 448,
        use_sam_refine: bool = False,
        sam_model: str = "facebook/sam2.1-hiera-large",
        use_tta: bool = False,
        score_thr: float = 0.3,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        detector_image_size: int = 1536,
        tile_if_over: int = 2048,
        tile_batch: int = 4,
        max_detections: int = 500,
        color_tta: bool = True,
        color_masked: bool = True,
        color_pad: float = 0.15,
        color_chunk_size: int = 32,
        type_chunk_size: int = 32,
        sam_min_iou: float = 0.70,
        sam_min_stability: float = 0.60,
        min_fill_ratio: float = 0.15,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.use_tta = use_tta
        self.use_sam_refine = use_sam_refine
        self.score_thr = score_thr
        self.color_image_size = color_image_size
        self.color_tta = color_tta
        self.color_masked = color_masked
        self.color_pad = color_pad
        self.color_chunk_size = color_chunk_size
        self.type_image_size = type_image_size
        self.type_chunk_size = type_chunk_size
        self.sam_min_iou = sam_min_iou
        self.sam_min_stability = sam_min_stability
        self.min_fill_ratio = min_fill_ratio
        self.detector_image_size = detector_image_size
        self.tile_if_over = tile_if_over
        self.tile_batch = tile_batch

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        self.detector = Mask2FormerDetector(
            weights_dir=maskformer_dir,
            device=self.device,
            detector_image_size=detector_image_size,
            score_thr=score_thr,
            mask_threshold=mask_threshold,
            overlap_mask_area_threshold=overlap_mask_area_threshold,
            max_detections=max_detections,
            tile_if_over=tile_if_over,
            tile_batch=tile_batch,
            use_tta=False,
        )

        self.color_model = None
        self.color_class_names: list[str] = []
        if color_weights is not None:
            import timm
            from safetensors.torch import load_file

            if color_class_names is None:
                raise ValueError("color_class_names required when color_weights is provided")
            self.color_class_names = list(color_class_names)
            self.color_model = timm.create_model(
                color_model_name, pretrained=False, num_classes=len(self.color_class_names)
            )
            state = load_file(str(color_weights))
            self.color_model.load_state_dict(state, strict=False)
            self.color_model.to(self.device).eval()
            self._color_mean = torch.tensor(IMNET_MEAN, device=self.device).view(1, 3, 1, 1)
            self._color_std = torch.tensor(IMNET_STD, device=self.device).view(1, 3, 1, 1)

        self.type_model = None
        self.type_class_names: list[str] = []
        if type_weights is not None:
            import timm
            from safetensors.torch import load_file

            if type_class_names is None:
                raise ValueError("type_class_names required when type_weights is provided")
            self.type_class_names = list(type_class_names)
            self.type_model = timm.create_model(
                type_model_name, pretrained=False, num_classes=len(self.type_class_names)
            )
            state = load_file(str(type_weights))
            self.type_model.load_state_dict(state, strict=False)
            self.type_model.to(self.device).eval()
            self._type_mean = torch.tensor(IMNET_MEAN, device=self.device).view(1, 3, 1, 1)
            self._type_std = torch.tensor(IMNET_STD, device=self.device).view(1, 3, 1, 1)

        self.sam = None
        if self.use_sam_refine:
            from common.sam_refiner import SAMRefiner

            self.sam = SAMRefiner(sam_model, device=self.device)

    def _detect(self, image: np.ndarray) -> list[dict]:
        H, W = image.shape[:2]
        if max(H, W) > self.tile_if_over:
            return self._detect_tiled(image)
        if self.use_tta:
            preds = detector_tta_hflip_scales(
                image, self.detector.detect_batch, scales=(0.85, 1.0, 1.15), hflip=True
            )
            return merge_instances_by_mask_iou(preds, iou_thr=0.5, containment_thr=0.7, union=False)
        return self.detector.detect_batch([image])[0]

    def _detect_tiled(self, image: np.ndarray) -> list[dict]:
        H, W = image.shape[:2]
        tiles = list(iter_tiles(image, size=self.detector_image_size, overlap=0.25))
        if not tiles:
            return []
        all_instances: list[dict] = []
        TB = max(1, self.tile_batch)
        for start in range(0, len(tiles), TB):
            chunk = tiles[start: start + TB]
            chunk_results = self.detector.detect_batch([t.image for t in chunk])
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

    def _refine_masks(self, image: np.ndarray, instances: list[dict]) -> list[dict]:
        if not instances or self.sam is None:
            return instances
        self.sam.set_image(image)
        try:
            boxes = [inst["bbox"] for inst in instances]
            coarse = [inst["mask"] for inst in instances]
            refined = self.sam.refine(boxes, coarse_masks=coarse, multimask=True)
            out: list[dict] = []
            for inst, r in zip(instances, refined):
                sam_iou = float(r["sam_iou"])
                sam_stab = float(r["sam_stability"])
                trust_sam = sam_iou >= self.sam_min_iou and sam_stab >= self.sam_min_stability
                new_mask = r["mask"] if trust_sam else inst["mask"]
                bbox = _mask_bbox(new_mask)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                ba = max(1, (x1 - x0) * (y1 - y0))
                fill = float(new_mask.sum()) / ba
                if fill < self.min_fill_ratio:
                    continue
                out.append({
                    **inst,
                    "mask": new_mask,
                    "bbox": bbox,
                    "sam_iou": sam_iou,
                    "sam_stability": sam_stab,
                    "sam_trusted": trust_sam,
                    "fill_ratio": fill,
                    "score": sam_iou,
                })
            return merge_instances_by_mask_iou(out, iou_thr=0.5, containment_thr=0.7, union=False)
        finally:
            self.sam.reset()

    @torch.inference_mode()
    def _classify_colors(self, image: np.ndarray, instances: list[dict]) -> list[dict]:
        if not instances or self.color_model is None:
            return [{**inst, "color": None, "color_conf": 0.0} for inst in instances]

        H_im, W_im = image.shape[:2]
        S = self.color_image_size
        pad = self.color_pad

        crops: list[np.ndarray] = []
        valid_idxs: list[int] = []
        for i, inst in enumerate(instances):
            x0, y0, x1, y1 = inst["bbox"]
            side = max(x1 - x0, y1 - y0) * (1 + pad)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            nx0 = max(0, int(cx - side / 2))
            ny0 = max(0, int(cy - side / 2))
            nx1 = min(W_im, int(cx + side / 2))
            ny1 = min(H_im, int(cy + side / 2))
            if nx1 - nx0 < 8 or ny1 - ny0 < 8:
                continue
            crop = image[ny0:ny1, nx0:nx1].copy()
            if self.color_masked:
                m = inst["mask"][ny0:ny1, nx0:nx1]
                crop[~m] = 0
            crops.append(_letterbox(crop, S))
            valid_idxs.append(i)

        out: list[dict] = [
            {**inst, "color": "UNKNOWN", "color_conf": 0.0} for inst in instances
        ]
        if not valid_idxs:
            return out

        arr_u8 = np.stack(crops, axis=0)
        batch = torch.from_numpy(arr_u8).to(self.device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        batch = (batch - self._color_mean) / self._color_std

        chunks: list[torch.Tensor] = []
        with torch.amp.autocast(self.device.type, dtype=torch.float16):
            for s in range(0, batch.shape[0], self.color_chunk_size):
                sub = batch[s: s + self.color_chunk_size]
                if self.color_tta:
                    chunks.append(classifier_tta_rot_flip(sub, self.color_model))
                else:
                    chunks.append(torch.softmax(self.color_model(sub), dim=1))
        probs = torch.cat(chunks, dim=0).float().cpu().numpy()
        top_idx = probs.argmax(axis=1)
        top_conf = probs[np.arange(len(top_idx)), top_idx]

        for k, vi in enumerate(valid_idxs):
            idx = int(top_idx[k])
            out[vi] = {
                **instances[vi],
                "color": self.color_class_names[idx],
                "color_conf": float(top_conf[k]),
                "color_probs": probs[k].tolist(),
            }
        return out

    @torch.inference_mode()
    def _classify_types(self, image: np.ndarray, instances: list[dict]) -> list[dict]:
        if not instances or self.type_model is None:
            return [{**inst, "hold_type": None, "type_conf": 0.0} for inst in instances]

        H_im, W_im = image.shape[:2]
        S = self.type_image_size

        crops: list[np.ndarray] = []
        valid_idxs: list[int] = []
        for i, inst in enumerate(instances):
            x0, y0, x1, y1 = inst["bbox"]
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = image[int(y0):int(y1), int(x0):int(x1)].copy()
            crops.append(_letterbox(crop, S))
            valid_idxs.append(i)

        out: list[dict] = [
            {**inst, "hold_type": "UNKNOWN", "type_conf": 0.0} for inst in instances
        ]
        if not valid_idxs:
            return out

        arr_u8 = np.stack(crops, axis=0)
        batch = torch.from_numpy(arr_u8).to(self.device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        batch = (batch - self._type_mean) / self._type_std

        chunks: list[torch.Tensor] = []
        with torch.amp.autocast(self.device.type, dtype=torch.float16):
            for s in range(0, batch.shape[0], self.type_chunk_size):
                sub = batch[s: s + self.type_chunk_size]
                if self.use_tta:
                    chunks.append(classifier_tta_rot_flip(sub, self.type_model))
                else:
                    chunks.append(torch.softmax(self.type_model(sub), dim=1))
        probs = torch.cat(chunks, dim=0).float().cpu().numpy()
        top_idx = probs.argmax(axis=1)
        top_conf = probs[np.arange(len(top_idx)), top_idx]

        for k, vi in enumerate(valid_idxs):
            idx = int(top_idx[k])
            out[vi] = {
                **instances[vi],
                "hold_type": self.type_class_names[idx],
                "type_conf": float(top_conf[k]),
            }
        return out

    def process(self, image: str | Path | np.ndarray) -> list[dict]:
        if isinstance(image, (str, Path)):
            img = _read_rgb(Path(image))
        else:
            img = image

        instances = self._detect(img)
        instances = self._refine_masks(img, instances)
        instances = self._classify_colors(img, instances)
        instances = self._classify_types(img, instances)

        for inst in instances:
            inst["mask_rle"] = _mask_to_rle(inst["mask"])
            inst["bbox"] = [float(x) for x in inst["bbox"]]
            inst["det_conf"] = float(inst.pop("score"))
            inst.pop("mask", None)
            inst.pop("color_probs", None)
        return instances


def run_climb_inference(
    maskformer_dir: str,
    image_dir: str,
    output: str,
    color_weights: str | None = None,
    color_model_config: str = "eva02_color",
    type_weights: str | None = None,
    type_model_config: str = "eva02_type",
    use_sam_refine: bool = False,
    sam_model: str = "facebook/sam2.1-hiera-large",
    use_tta: bool = False,
    preview: bool = False,
):
    device = cfg.torch.device
    _default_backbone = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"

    color_class_names: list[str] | None = None
    color_model_name = _default_backbone
    color_image_size = 448
    if color_weights is not None:
        ccfg = cfg.model_cfg(color_model_config)
        color_class_names = list(ccfg["class_names"])
        color_model_name = ccfg.get("backbone", color_model_name)
        color_image_size = ccfg.get("image_size", color_image_size)

    type_class_names: list[str] | None = None
    type_model_name = _default_backbone
    type_image_size = 448
    if type_weights is not None:
        from data.crop_dataset import get_dataset_info
        tcfg = cfg.model_cfg(type_model_config)
        type_model_name = tcfg.get("backbone", type_model_name)
        type_image_size = tcfg.get("image_size", type_image_size)
        info = get_dataset_info(type_model_config)
        type_class_names = info.class_names

    pipe = ClimbPipeline(
        maskformer_dir=maskformer_dir,
        color_weights=color_weights,
        color_model_name=color_model_name,
        color_class_names=color_class_names,
        color_image_size=color_image_size,
        type_weights=type_weights,
        type_model_name=type_model_name,
        type_class_names=type_class_names,
        type_image_size=type_image_size,
        use_sam_refine=use_sam_refine,
        sam_model=sam_model,
        use_tta=use_tta,
        device=device,
    )

    test_dir = Path(image_dir)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in extensions)

    all_results: dict[str, list[dict]] = {}
    for img_path in tqdm(image_paths, desc="climb inference"):
        img = _read_rgb(img_path)
        results = pipe.process(img)
        all_results[img_path.name] = [
            {k: v for k, v in r.items() if k != "mask"}
            for r in results
        ]

        if preview:
            overlay = img.copy()
            for r in results:
                rle = dict(r["mask_rle"])
                rle["counts"] = rle["counts"].encode("ascii")
                m = cocomask.decode(rle).astype(bool)
                overlay[m] = (overlay[m] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
                x0, y0, x1, y1 = [int(v) for v in r["bbox"]]

                color_name = r.get("color")
                box_bgr = _COLOR_BGR.get(color_name, (0, 255, 0))
                cv2.rectangle(overlay, (x0, y0), (x1, y1), box_bgr, 3)

                hold_type = r.get("hold_type")
                seg_label = r.get("class_name", "")
                type_label = hold_type if (hold_type and hold_type not in ("UNKNOWN", None)) else seg_label
                label_parts = [p for p in [type_label, color_name] if p and p != "UNKNOWN"]
                label = " | ".join(label_parts)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                pos = (x0, max(22, y0 - 6))
                cv2.putText(overlay, label, pos, font, font_scale, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(overlay, label, pos, font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
            Image.fromarray(overlay).save(out_dir / img_path.name)

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved %d results to %s", len(image_paths), out_dir)
