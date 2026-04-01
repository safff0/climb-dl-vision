import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import cfg
from models import create_model

SCORE_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fig_4_6.pdf")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(cfg.torch.device)

    model = create_model("mask_rcnn_hold").to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    split_dir = Path(args.dataset) / args.split
    coco = COCO(str(split_dir / "_annotations.coco.json"))
    to_tensor = T.ToTensor()

    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)

    pairs = []
    for img_id in img_ids:
        if len(pairs) >= args.n:
            break

        img_info = coco.loadImgs(img_id)[0]
        img = Image.open(split_dir / img_info["file_name"]).convert("RGB")
        img_tensor = to_tensor(img).to(device)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        gt_boxes = []
        gt_masks = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 20 or h < 20:
                continue
            gt_boxes.append([x, y, x + w, y + h])
            gt_masks.append(coco.annToMask(ann))

        if not gt_boxes:
            continue

        with torch.no_grad():
            pred = model([img_tensor])[0]

        pred_boxes = pred["boxes"].cpu()
        pred_masks = pred["masks"].cpu()
        pred_scores = pred["scores"].cpu()
        keep = pred_scores > SCORE_THRESHOLD
        pred_boxes = pred_boxes[keep]
        pred_masks = pred_masks[keep]

        if len(pred_boxes) == 0:
            continue

        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
        ious = box_iou(pred_boxes, gt_boxes_t)

        for gt_i in range(len(gt_boxes)):
            if len(pairs) >= args.n:
                break
            best_pred = ious[:, gt_i].argmax().item()
            if ious[best_pred, gt_i] < 0.3:
                continue

            box = gt_boxes[gt_i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            gt_crop = gt_masks[gt_i][y1:y2, x1:x2]
            pred_crop = (pred_masks[best_pred, 0] > 0.5).numpy().astype(np.uint8)[y1:y2, x1:x2]
            img_crop = np.array(img)[y1:y2, x1:x2]

            pairs.append((img_crop, gt_crop, pred_crop))

    cols = 3
    rows = len(pairs)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, (img_crop, gt_mask, pred_mask) in enumerate(pairs):
        axes[i, 0].imshow(img_crop)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Crop")

        axes[i, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title("GT mask")

        axes[i, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title("Predicted mask")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
