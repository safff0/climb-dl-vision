import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from common.config import cfg
from models import create_model
from pipelines import register_pipeline

SCORE_THRESHOLD = 0.5


def _load_category_names(dataset_root: str) -> dict[int, str]:
    ann_path = Path(dataset_root) / "train" / "_annotations.coco.json"
    with open(ann_path) as f:
        data = json.load(f)
    return {cat["id"]: cat["name"] for cat in data["categories"]}


def _visualize(img_tensor, boxes, masks, labels, category_names):
    img_uint8 = (img_tensor * 255).to(torch.uint8).cpu()

    if masks.shape[0] > 0:
        img_uint8 = draw_segmentation_masks(img_uint8, masks.squeeze(1).cpu(), alpha=0.4)

    if boxes.shape[0] > 0:
        label_strs = [category_names.get(l.item(), str(l.item())) for l in labels]
        img_uint8 = draw_bounding_boxes(img_uint8, boxes.cpu(), labels=label_strs, width=2)

    return img_uint8


@register_pipeline("mask_rcnn", "inference")
def run_inference(model_name: str, weights: str, output: str, preview: bool = False):
    device = torch.device(cfg.torch.device)

    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    dataset_root = cfg.models.get(model_name, {}).get("dataset", "")
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
            img_tensor = to_tensor(img).to(device)
            prediction = model([img_tensor])[0]

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
                print(f"Preview saved to {preview_path}")

    print(f"Saved {len(image_paths)} results to {out_dir}")
