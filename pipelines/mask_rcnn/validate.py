import json
import tempfile

import numpy as np
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.config import cfg
from data.coco_dataset import get_coco_dataloader
from models import create_model
from pipelines import register_pipeline


def evaluate_coco(model, model_name: str, device: torch.device) -> dict:
    model.eval()
    loader = get_coco_dataloader(model_name, "valid")

    dataset_root = cfg.models.get(model_name, {}).get("dataset", "")
    coco_gt = COCO(f"{dataset_root}/valid/_annotations.coco.json")

    bbox_results = []
    segm_results = []

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for target, pred in zip(targets, predictions):
                image_id = target["image_id"].item()
                boxes = pred["boxes"].cpu()
                scores = pred["scores"].cpu()
                labels = pred["labels"].cpu()
                masks = pred["masks"].cpu()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    score = scores[i].item()
                    cat_id = labels[i].item()

                    bbox_results.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    })

                    binary_mask = (masks[i, 0] > 0.5).numpy().astype(np.uint8)
                    rle = mask_util.encode(np.asfortranarray(binary_mask))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    segm_results.append({
                        "image_id": image_id,
                        "category_id": cat_id,
                        "segmentation": rle,
                        "score": score,
                    })

    metrics = {"bbox_ap": 0.0, "segm_ap": 0.0}

    if not bbox_results:
        return metrics

    for iou_type, results in [("bbox", bbox_results), ("segm", segm_results)]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(results, f)
            results_file = f.name

        coco_dt = coco_gt.loadRes(results_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        metrics[f"{iou_type}_ap"] = coco_eval.stats[0]

    return metrics


@register_pipeline("mask_rcnn", "validate")
def run_validate(model_name: str, weights: str):
    device = torch.device(cfg.torch.device)

    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))

    metrics = evaluate_coco(model, model_name, device)
    print(f"bbox AP: {metrics['bbox_ap']:.4f}")
    print(f"segm AP: {metrics['segm_ap']:.4f}")
