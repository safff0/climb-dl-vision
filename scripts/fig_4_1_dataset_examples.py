import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fig_4_1.pdf")
    args = parser.parse_args()

    random.seed(args.seed)
    split_dir = Path(args.dataset) / args.split
    coco = COCO(str(split_dir / "_annotations.coco.json"))

    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)
    img_ids = img_ids[:args.n]

    cols = 3
    rows = (len(img_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if len(img_ids) == 1 else axes.flatten()

    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    for ax_idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img = Image.open(split_dir / img_info["file_name"]).convert("RGB")
        img_tensor = T.ToTensor()(img)
        img_uint8 = (img_tensor * 255).to(torch.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        masks = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            masks.append(coco.annToMask(ann))
            labels.append(categories.get(ann["category_id"], "?"))

        if masks:
            mask_tensor = torch.as_tensor(np.array(masks), dtype=torch.bool)
            img_uint8 = draw_segmentation_masks(img_uint8, mask_tensor, alpha=0.4)
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
            img_uint8 = draw_bounding_boxes(img_uint8, box_tensor, labels=labels, width=2, font_size=12)

        axes[ax_idx].imshow(img_uint8.permute(1, 2, 0).numpy())
        axes[ax_idx].axis("off")

    for ax_idx in range(len(img_ids), len(axes)):
        axes[ax_idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
