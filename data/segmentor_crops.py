import json
import logging
from pathlib import Path

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

        for img_id in tqdm(coco.getImgIds(), desc=f"[{split}] Segmentor crops"):
            img_info = coco.loadImgs(img_id)[0]
            img = Image.open(split_dir / img_info["file_name"]).convert("RGB")
            img_tensor = to_tensor(img).to(device)
            iw, ih = img.size

            with torch.no_grad():
                pred = segmentor([img_tensor])[0]

            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            keep = pred_scores > 0.3
            pred_boxes = pred_boxes[keep]

            if len(pred_boxes) == 0:
                continue

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            gt_boxes = []
            gt_labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w < 1 or h < 1:
                    continue
                cat_idx = cat_id_to_idx.get(ann["category_id"])
                if cat_idx is None:
                    continue
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(cat_idx)

            if not gt_boxes:
                continue

            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
            ious = box_iou(pred_boxes, gt_boxes_t)

            matched_gt = set()
            for pred_i in range(len(pred_boxes)):
                best_gt = ious[pred_i].argmax().item()
                best_iou = ious[pred_i, best_gt].item()
                if best_iou < IOU_THRESHOLD:
                    continue
                if best_gt in matched_gt:
                    continue
                matched_gt.add(best_gt)

                box = pred_boxes[pred_i]
                label = gt_labels[best_gt]
                x1, y1, x2, y2 = box.int().tolist()
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(iw, x2 + padding)
                y2 = min(ih, y2 + padding)

                crop = img.crop((x1, y1, x2, y2))
                crop = crop.resize((crop_size, crop_size), Image.BILINEAR)

                crop_name = f"crop_{crop_idx:06d}.jpg"
                crop.save(out_split / crop_name)
                crop_records.append(CropRecord(
                    file=crop_name,
                    label=label,
                    source_image=img_info["file_name"],
                    pred_box=[x1, y1, x2, y2],
                ))
                crop_idx += 1

        meta = CropMeta(
            class_names=class_names,
            num_classes=len(class_names),
            crops=crop_records,
        )
        with open(out_split / "labels.json", "w") as f:
            json.dump(meta.to_dict(), f, indent=2)

        logger.info("[%s] Saved %d crops to %s", split, len(crop_records), out_split)
