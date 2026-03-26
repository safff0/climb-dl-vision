import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode
from data.crop_dataset import get_dataset_info
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)
SCORE_THRESHOLD = 0.5


def _load_segmentor(segmentor_model: str, weights_path: str, device):
    model = create_model(segmentor_model).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _crop_and_prepare(img_tensor, box, mask, crop_size, padding, use_mask):
    _, h, w = img_tensor.shape
    x1, y1, x2, y2 = box.int().tolist()
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = img_tensor[:, y1:y2, x1:x2]
    crop = T.Resize((crop_size, crop_size))(crop)

    if use_mask:
        mask_crop = mask[y1:y2, x1:x2].unsqueeze(0).float()
        mask_crop = T.Resize((crop_size, crop_size))(mask_crop)
        crop = torch.cat([crop, mask_crop], dim=0)

    return crop.unsqueeze(0)


@register_pipeline("hold_classifier", PipelineMode.INFERENCE)
def run_inference(model_name: str, weights: str, output: str, image_dir: str, preview: bool = False):
    device = torch.device(cfg.torch.device)
    mcfg = cfg.model_cfg(model_name)
    crop_size = mcfg["crop_size"]
    padding = mcfg["crop_padding"]
    use_mask = mcfg["use_mask_channel"]
    segmentor_model = mcfg["segmentor_model"]
    segmentor_weights = mcfg["segmentor_weights"]

    segmentor = _load_segmentor(segmentor_model, segmentor_weights, device)

    classifier = create_model(model_name).to(device)
    classifier.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    classifier.eval()

    info = get_dataset_info(model_name)

    seg_dataset_root = cfg.model_cfg(segmentor_model)["dataset"]
    seg_ann_path = Path(seg_dataset_root) / "train" / "_annotations.coco.json"
    with open(seg_ann_path) as f:
        seg_cats = {c["id"]: c["name"] for c in json.load(f)["categories"]}

    hold_label_ids = [cid for cid, name in seg_cats.items() if name.lower() == "hold"]

    test_dir = Path(image_dir)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in extensions)

    to_tensor = T.ToTensor()

    with torch.no_grad():
        for idx, img_path in tqdm(enumerate(image_paths), desc="Processing images"):
            img = Image.open(img_path).convert("RGB")
            img_tensor = to_tensor(img).to(device)

            seg_pred = segmentor([img_tensor])[0]
            keep = seg_pred["scores"] > SCORE_THRESHOLD
            boxes = seg_pred["boxes"][keep]
            masks = seg_pred["masks"][keep] > 0.5
            labels = seg_pred["labels"][keep]
            scores = seg_pred["scores"][keep]

            annotations = []
            for i in range(len(boxes)):
                label_id = labels[i].item()
                is_hold = label_id in hold_label_ids

                ann = {
                    "box": boxes[i],
                    "mask": masks[i].squeeze(0),
                    "seg_label": seg_cats.get(label_id, str(label_id)),
                    "score": scores[i].item(),
                    "attribute": None,
                }

                if is_hold:
                    crop = _crop_and_prepare(
                        img_tensor, boxes[i], masks[i].squeeze(0),
                        crop_size, padding, use_mask,
                    )
                    cls_out = classifier(crop.to(device))
                    cls_pred = cls_out.argmax(dim=1).item()
                    ann["attribute"] = info.class_names[cls_pred]

                annotations.append(ann)

            img_uint8 = (img_tensor * 255).to(torch.uint8).cpu()
            all_masks = torch.stack([a["mask"].cpu() for a in annotations]) if annotations else torch.zeros(0, img_tensor.shape[1], img_tensor.shape[2], dtype=torch.bool)

            if all_masks.shape[0] > 0:
                img_uint8 = draw_segmentation_masks(img_uint8, all_masks, alpha=0.4)

            if annotations:
                box_tensor = torch.stack([a["box"].cpu() for a in annotations])
                label_strs = []
                for a in annotations:
                    text = a["seg_label"]
                    if a["attribute"]:
                        text += f": {a['attribute']}"
                    label_strs.append(text)
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
