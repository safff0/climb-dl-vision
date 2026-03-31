import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.ops import box_iou
from tqdm import tqdm

from common.config import cfg
from common.types import CropMeta, CropRecord, Split
from models import create_model

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.3
SCORE_THRESHOLD = 0.1
JITTER_RANGE = 0.15


def _jitter_box(box: list[float], iw: int, ih: int) -> list[int]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx = random.uniform(-JITTER_RANGE, JITTER_RANGE) * w
    dy = random.uniform(-JITTER_RANGE, JITTER_RANGE) * h
    dw = random.uniform(-JITTER_RANGE * 0.5, JITTER_RANGE * 0.5) * w
    dh = random.uniform(-JITTER_RANGE * 0.5, JITTER_RANGE * 0.5) * h
    return [
        max(0, int(x1 + dx - dw)),
        max(0, int(y1 + dy - dh)),
        min(iw, int(x2 + dx + dw)),
        min(ih, int(y2 + dy + dh)),
    ]


def _save_crop_and_mask(img, mask_np, x1, y1, x2, y2, crop_size, crop_idx, out_split):
    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize((crop_size, crop_size), Image.BILINEAR)

    crop_name = f"crop_{crop_idx:06d}.jpg"
    crop.save(out_split / crop_name)

    mask_name = None
    if mask_np is not None:
        mask_crop = mask_np[y1:y2, x1:x2]
        mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((crop_size, crop_size), Image.BILINEAR)
        mask_name = f"mask_{crop_idx:06d}.png"
        mask_pil.save(out_split / mask_name)

    return crop_name, mask_name


def prepare_segmentor_crops(classifier_model_name: str):
    device = torch.device(cfg.torch.device)
    mcfg = cfg.model_cfg(classifier_model_name)
    crop_size = mcfg["crop_size"]
    padding = mcfg["crop_padding"]
    segmentor_model = mcfg["segmentor_model"]
    segmentor_weights = mcfg["segmentor_weights"]
    dataset_root = mcfg["dataset"]

    segmentor = create_model(segmentor_model).to(device)
    segmentor.load_state_dict(torch.load(segmentor_weights, map_location=device, weights_only=True))
    segmentor.eval()

    to_tensor = T.ToTensor()
    output_root = Path(dataset_root) / "segmentor_crops"
    random.seed(cfg.torch.seed)

    seg_dataset_root = cfg.model_cfg(segmentor_model)["dataset"]
    seg_ann_path = Path(seg_dataset_root) / "train" / "_annotations.coco.json"
    with open(seg_ann_path) as f:
        seg_cats = json.load(f)["categories"]
    volume_label_id = next((c["id"] for c in seg_cats if c["name"].lower() == "volume"), -1)

    for split in [Split.TRAIN, Split.VALID]:
        split_dir = Path(dataset_root) / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            logger.warning("Skipping %s — no annotations", split)
            continue

        coco = COCO(str(ann_path))
        categories = coco.loadCats(coco.getCatIds())
        cat_id_to_idx = {}
        class_names = []
        for idx, cat in enumerate(categories):
            cat_id_to_idx[cat["id"]] = idx
            class_names.append(cat["name"])

        out_split = output_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        crop_records: list[CropRecord] = []
        crop_idx = 0
        matched_count = 0
        fallback_count = 0
        total_gt = 0

        for img_id in tqdm(coco.getImgIds(), desc=f"[{split}] Segmentor crops"):
            img_info = coco.loadImgs(img_id)[0]
            img = Image.open(split_dir / img_info["file_name"]).convert("RGB")
            img_tensor = to_tensor(img).to(device)
            iw, ih = img.size

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            gt_boxes = []
            gt_labels = []
            gt_masks = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w < 1 or h < 1:
                    continue
                cat_idx = cat_id_to_idx.get(ann["category_id"])
                if cat_idx is None:
                    continue
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(cat_idx)
                seg = ann.get("segmentation", [])
                if seg and not (isinstance(seg, list) and len(seg) == 0):
                    gt_masks.append(coco.annToMask(ann))
                else:
                    gt_masks.append(np.ones((ih, iw), dtype=np.uint8))

            if not gt_boxes:
                continue

            total_gt += len(gt_boxes)
            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)

            with torch.no_grad():
                pred = segmentor([img_tensor])[0]

            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_masks = pred["masks"].cpu()
            pred_labels = pred["labels"].cpu()
            keep = (pred_scores > SCORE_THRESHOLD) & (pred_labels != volume_label_id)
            pred_boxes = pred_boxes[keep]
            pred_masks = pred_masks[keep]

            matched_gt = set()
            if len(pred_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes_t)

                for pred_i in range(len(pred_boxes)):
                    best_gt = ious[pred_i].argmax().item()
                    best_iou = ious[pred_i, best_gt].item()
                    if best_iou < IOU_THRESHOLD or best_gt in matched_gt:
                        continue
                    matched_gt.add(best_gt)

                    box = pred_boxes[pred_i]
                    label = gt_labels[best_gt]
                    x1, y1, x2, y2 = box.int().tolist()
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(iw, x2)
                    y2 = min(ih, y2)

                    mask_np = (pred_masks[pred_i, 0] > 0.5).numpy().astype(np.uint8)

                    crop_name, mask_name = _save_crop_and_mask(
                        img, mask_np, x1, y1, x2, y2, crop_size, crop_idx, out_split,
                    )
                    crop_records.append(CropRecord(
                        file=crop_name,
                        label=label,
                        source_image=img_info["file_name"],
                        pred_box=[x1, y1, x2, y2],
                        mask_file=mask_name,
                    ))
                    crop_idx += 1
                    matched_count += 1

            for gt_i in range(len(gt_boxes)):
                if gt_i in matched_gt:
                    continue

                gt_box = gt_boxes[gt_i]
                label = gt_labels[gt_i]
                gt_mask = gt_masks[gt_i]

                jittered = _jitter_box(gt_box, iw, ih)
                x1 = max(0, jittered[0])
                y1 = max(0, jittered[1])
                x2 = min(iw, jittered[2])
                y2 = min(ih, jittered[3])

                crop_name, mask_name = _save_crop_and_mask(
                    img, gt_mask, x1, y1, x2, y2, crop_size, crop_idx, out_split,
                )
                crop_records.append(CropRecord(
                    file=crop_name,
                    label=label,
                    source_image=img_info["file_name"],
                    pred_box=[x1, y1, x2, y2],
                    mask_file=mask_name,
                ))
                crop_idx += 1
                fallback_count += 1

        meta = CropMeta(
            class_names=class_names,
            num_classes=len(class_names),
            crops=crop_records,
        )
        with open(out_split / "labels.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        logger.info(
            "[%s] Saved %d crops (%d matched, %d fallback GT+jitter, %d total GT)",
            split, len(crop_records), matched_count, fallback_count, total_gt,
        )
