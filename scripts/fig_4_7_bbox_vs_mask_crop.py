import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import cfg
from models import create_model

SCORE_THRESHOLD = 0.5
PADDING = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fig_4_7.pdf")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(cfg.torch.device)

    model = create_model("mask_rcnn_hold").to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)
    ih, iw = img_np.shape[:2]
    img_tensor = T.ToTensor()(img).to(device)

    with torch.no_grad():
        pred = model([img_tensor])[0]

    keep = pred["scores"] > SCORE_THRESHOLD
    boxes = pred["boxes"][keep].cpu()
    masks = (pred["masks"][keep] > 0.5).cpu()

    indices = list(range(len(boxes)))
    random.shuffle(indices)
    indices = indices[:args.n]

    fig, axes = plt.subplots(len(indices), 2, figsize=(6, 3 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        box = boxes[idx].int().tolist()
        x1 = max(0, box[0] - PADDING)
        y1 = max(0, box[1] - PADDING)
        x2 = min(iw, box[2] + PADDING)
        y2 = min(ih, box[3] + PADDING)

        bbox_crop = img_np[y1:y2, x1:x2]

        mask = masks[idx, 0].numpy().astype(np.uint8)
        mask_crop = mask[y1:y2, x1:x2]
        masked_crop = bbox_crop.copy()
        masked_crop[mask_crop == 0] = 0

        axes[i, 0].imshow(bbox_crop)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Bounding box crop")

        axes[i, 1].imshow(masked_crop)
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("Mask-guided crop")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
