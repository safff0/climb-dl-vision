import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from common.config import cfg
from common.types import PipelineMode
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)
SCORE_THRESHOLD = 0.5


def _load_category_names(dataset_root: str) -> dict[int, str]:
    ann_path = Path(dataset_root) / "train" / "_annotations.coco.json"
    with open(ann_path) as f:
        data = json.load(f)
    return {cat["id"]: cat["name"] for cat in data["categories"]}


def _tiled_predict(model, img_tensor, tile_size, tile_overlap, device):
    _, h, w = img_tensor.shape
    stride = tile_size - tile_overlap

    all_boxes = []
    all_scores = []
    all_labels = []
    all_masks = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)

            tile = img_tensor[:, y1:y2, x1:x2].to(device)
            pred = model([tile])[0]

            if len(pred["boxes"]) == 0:
                continue

            boxes = pred["boxes"].cpu()
            boxes[:, [0, 2]] += x1
            boxes[:, [1, 3]] += y1

            all_boxes.append(boxes)
            all_scores.append(pred["scores"].cpu())
            all_labels.append(pred["labels"].cpu())

            masks_full = torch.zeros(len(boxes), 1, h, w)
            tile_masks = pred["masks"].cpu()
            masks_full[:, :, y1:y2, x1:x2] = tile_masks
            all_masks.append(masks_full)

    if not all_boxes:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
            "masks": torch.zeros((0, 1, h, w)),
        }

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)

    keep = nms(boxes, scores, iou_threshold=0.5)
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
        "masks": masks[keep],
    }


def _visualize(img_tensor, boxes, masks, labels, category_names):
    img_uint8 = (img_tensor * 255).to(torch.uint8).cpu()

    if masks.shape[0] > 0:
        img_uint8 = draw_segmentation_masks(img_uint8, masks.squeeze(1).cpu(), alpha=0.4)

    if boxes.shape[0] > 0:
        label_strs = [category_names.get(l.item(), str(l.item())) for l in labels]
        img_uint8 = draw_bounding_boxes(img_uint8, boxes.cpu(), labels=label_strs, width=2)

    return img_uint8


@register_pipeline("mask_rcnn", PipelineMode.INFERENCE)
def run_inference(model_name: str, weights: str, output: str, preview: bool = False):
    device = torch.device(cfg.torch.device)

    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    mcfg = cfg.model_cfg(model_name)
    dataset_root = mcfg["dataset"]
    use_tiles = mcfg.get("tile_inference", False)
    tile_size = mcfg.get("tile_size", 1024)
    tile_overlap = mcfg.get("tile_overlap", 128)

    category_names = _load_category_names(dataset_root)
    test_dir = Path(dataset_root) / "test"
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in extensions)

    to_tensor = T.ToTensor()

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            img_tensor = to_tensor(img)

            if use_tiles:
                prediction = _tiled_predict(model, img_tensor, tile_size, tile_overlap, device)
            else:
                prediction = model([img_tensor.to(device)])[0]
                prediction = {k: v.cpu() for k, v in prediction.items()}

            keep = prediction["scores"] > SCORE_THRESHOLD
            boxes = prediction["boxes"][keep]
            masks = prediction["masks"][keep] > 0.5
            labels = prediction["labels"][keep]

            result = _visualize(img_tensor, boxes, masks, labels, category_names)
            result_img = T.ToPILImage()(result)
            result_img.save(out_dir / img_path.name)

            if preview and idx == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.imshow(img)
                ax1.set_title("Original")
                ax1.axis("off")
                ax2.imshow(result.permute(1, 2, 0).numpy())
                ax2.set_title(f"Detections ({len(boxes)} objects)")
                ax2.axis("off")
                plt.tight_layout()
                preview_path = out_dir / "preview.png"
                plt.savefig(preview_path, dpi=150)
                plt.close()
                logger.info("Preview saved to %s", preview_path)

    logger.info("Saved %d results to %s", len(image_paths), out_dir)
