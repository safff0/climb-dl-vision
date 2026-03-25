import json
import tempfile

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.config import cfg
from data.coco_dataset import get_coco_dataloader
from models import create_model
from pipelines import register_pipeline


@register_pipeline("mask_rcnn", "validate")
def run_validate(model_name: str, weights: str):
    device = torch.device(cfg.torch.device)

    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    loader = get_coco_dataloader(model_name, "valid")

    dataset_root = cfg.models.get(model_name, {}).get("dataset", "")
    coco_gt = COCO(f"{dataset_root}/valid/_annotations.coco.json")

    results = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            predictions = model(images)

            for target, pred in zip(targets, predictions):
                image_id = target["image_id"].item()
                boxes = pred["boxes"].cpu()
                scores = pred["scores"].cpu()
                labels = pred["labels"].cpu()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].tolist()
                    result = {
                        "image_id": image_id,
                        "category_id": labels[i].item(),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": scores[i].item(),
                    }
                    results.append(result)

    if not results:
        print("No detections produced")
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        results_file = f.name

    coco_dt = coco_gt.loadRes(results_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
