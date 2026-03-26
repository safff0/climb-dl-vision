import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode
from data.crop_dataset import get_dataset_info
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)
SCORE_THRESHOLD = 0.7


def _load_model(model_name: str, weights_path: str, device):
    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _crop_and_prepare(img_tensor, box, crop_size, padding):
    _, h, w = img_tensor.shape
    x1, y1, x2, y2 = box.int().tolist()
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = img_tensor[:, y1:y2, x1:x2]
    crop = T.Resize((crop_size, crop_size))(crop)

    return crop.unsqueeze(0)


def _class_aware_nms(boxes, scores, labels, masks, iou_threshold=0.5):
    keep_indices = []
    for cls_id in labels.unique():
        cls_mask = labels == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = cls_mask.nonzero(as_tuple=True)[0]
        nms_keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.append(cls_indices[nms_keep])

    if not keep_indices:
        return boxes[:0], scores[:0], labels[:0], masks[:0]

    keep = torch.cat(keep_indices)
    return boxes[keep], scores[keep], labels[keep], masks[keep]


def run_full_inference(
    segmentor_weights: str,
    image_dir: str,
    output: str,
    color_weights: str = None,
    type_weights: str = None,
    preview: bool = False,
):
    device = torch.device(cfg.torch.device)
    seg_model_name = "mask_rcnn_hold"
    seg_mcfg = cfg.model_cfg(seg_model_name)

    segmentor = _load_model(seg_model_name, segmentor_weights, device)

    color_classifier = None
    color_names = None
    color_crop_size = 224
    color_padding = 16
    if color_weights:
        color_model_name = "hold_color_classifier"
        color_mcfg = cfg.model_cfg(color_model_name)
        color_crop_size = color_mcfg["crop_size"]
        color_padding = color_mcfg["crop_padding"]
        color_classifier = _load_model(color_model_name, color_weights, device)
        color_names = get_dataset_info(color_model_name).class_names

    type_classifier = None
    type_names = None
    type_crop_size = 224
    type_padding = 16
    if type_weights:
        type_model_name = "hold_type_classifier"
        type_mcfg = cfg.model_cfg(type_model_name)
        type_crop_size = type_mcfg["crop_size"]
        type_padding = type_mcfg["crop_padding"]
        type_classifier = _load_model(type_model_name, type_weights, device)
        type_names = get_dataset_info(type_model_name).class_names

    seg_dataset_root = seg_mcfg["dataset"]
    seg_ann_path = Path(seg_dataset_root) / "train" / "_annotations.coco.json"
    with open(seg_ann_path) as f:
        seg_cats = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    hold_label_ids = {cid for cid, name in seg_cats.items() if name.lower() == "hold"}

    test_dir = Path(image_dir)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in extensions)

    to_tensor = T.ToTensor()

    with torch.no_grad():
        for idx, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Inference"):
            img = Image.open(img_path).convert("RGB")
            img_tensor = to_tensor(img).to(device)

            seg_pred = segmentor([img_tensor])[0]
            keep = seg_pred["scores"] > SCORE_THRESHOLD
            boxes = seg_pred["boxes"][keep]
            scores_t = seg_pred["scores"][keep]
            masks = seg_pred["masks"][keep] > 0.5
            labels = seg_pred["labels"][keep]

            boxes, scores_t, labels, masks = _class_aware_nms(
                boxes, scores_t, labels, masks, iou_threshold=0.5,
            )

            annotations = []
            for i in range(len(boxes)):
                label_id = labels[i].item()
                is_hold = label_id in hold_label_ids

                ann = {
                    "box": boxes[i],
                    "mask": masks[i].squeeze(0),
                    "seg_label": seg_cats.get(label_id, str(label_id)),
                    "score": scores_t[i].item(),
                    "color": None,
                    "type": None,
                }

                if is_hold:
                    if color_classifier is not None:
                        crop = _crop_and_prepare(img_tensor, boxes[i], color_crop_size, color_padding)
                        pred = color_classifier(crop.to(device)).argmax(dim=1).item()
                        ann["color"] = color_names[pred]

                    if type_classifier is not None:
                        crop = _crop_and_prepare(img_tensor, boxes[i], type_crop_size, type_padding)
                        pred = type_classifier(crop.to(device)).argmax(dim=1).item()
                        ann["type"] = type_names[pred]

                annotations.append(ann)

            img_uint8 = (img_tensor * 255).to(torch.uint8).cpu()

            if annotations:
                all_masks = torch.stack([a["mask"].cpu() for a in annotations])
                img_uint8 = draw_segmentation_masks(img_uint8, all_masks, alpha=0.4)

                box_tensor = torch.stack([a["box"].cpu() for a in annotations])
                label_strs = []
                for a in annotations:
                    parts = [a["seg_label"]]
                    if a["color"]:
                        parts.append(a["color"])
                    if a["type"]:
                        parts.append(a["type"])
                    label_strs.append(" | ".join(parts))
                img_uint8 = draw_bounding_boxes(img_uint8, box_tensor, labels=label_strs, width=2)

            result_img = T.ToPILImage()(img_uint8)
            result_img.save(out_dir / img_path.name)

            if preview and idx == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.imshow(img)
                ax1.set_title("Original")
                ax1.axis("off")
                ax2.imshow(np.array(result_img))
                ax2.set_title(f"Detections ({len(annotations)} objects)")
                ax2.axis("off")
                plt.tight_layout()
                preview_path = out_dir / "preview.png"
                plt.savefig(preview_path, dpi=150)
                plt.close()
                logger.info("Preview saved to %s", preview_path)

    logger.info("Saved %d results to %s", len(image_paths), out_dir)


@register_pipeline("hold_classifier", PipelineMode.INFERENCE)
def run_inference(model_name: str, weights: str, output: str, image_dir: str, preview: bool = False):
    mcfg = cfg.model_cfg(model_name)
    augment_mode = mcfg.get("augment_mode", "type")

    color_weights = weights if augment_mode == "color" else None
    type_weights = weights if augment_mode == "type" else None

    run_full_inference(
        segmentor_weights=mcfg["segmentor_weights"],
        image_dir=image_dir,
        output=output,
        color_weights=color_weights,
        type_weights=type_weights,
        preview=preview,
    )
