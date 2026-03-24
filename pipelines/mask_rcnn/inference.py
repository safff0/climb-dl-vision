from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from common.config import cfg
from models import create_model
from pipelines import register_pipeline

SCORE_THRESHOLD = 0.5


@register_pipeline("mask_rcnn", "inference")
def run_inference(model_name: str, weights: str, output: str):
    device = torch.device(cfg.torch.device)

    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    dataset_root = cfg.models.get(model_name, {}).get("dataset", "")
    test_dir = Path(dataset_root) / "test"
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in extensions)

    to_tensor = T.ToTensor()

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = to_tensor(img).to(device)
            prediction = model([img_tensor])[0]

            keep = prediction["scores"] > SCORE_THRESHOLD
            boxes = prediction["boxes"][keep]
            masks = prediction["masks"][keep] > 0.5
            labels = prediction["labels"][keep]

            img_uint8 = (img_tensor * 255).to(torch.uint8).cpu()

            if masks.shape[0] > 0:
                img_uint8 = draw_segmentation_masks(img_uint8, masks.squeeze(1).cpu(), alpha=0.4)

            if boxes.shape[0] > 0:
                label_strs = [str(l.item()) for l in labels]
                img_uint8 = draw_bounding_boxes(img_uint8, boxes.cpu(), labels=label_strs, width=2)

            result_img = T.ToPILImage()(img_uint8)
            result_img.save(out_dir / img_path.name)

    print(f"Saved {len(image_paths)} results to {out_dir}")
