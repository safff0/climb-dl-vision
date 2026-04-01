import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="fig_4_2.pdf")
    args = parser.parse_args()

    random.seed(args.seed)
    split_dir = Path(args.dataset) / args.split
    coco = COCO(str(split_dir / "_annotations.coco.json"))
    categories = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    ann_ids = list(coco.anns.keys())
    random.shuffle(ann_ids)

    selected = []
    for ann_id in ann_ids:
        ann = coco.anns[ann_id]
        x, y, w, h = ann["bbox"]
        if w > 20 and h > 20:
            selected.append(ann)
        if len(selected) >= args.n:
            break

    fig, axes = plt.subplots(args.n, 2, figsize=(8, 3 * args.n))

    for i, ann in enumerate(selected):
        img_info = coco.loadImgs(ann["image_id"])[0]
        img = Image.open(split_dir / img_info["file_name"]).convert("RGB")

        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        img_with_box = img.copy()
        draw = ImageDraw.Draw(img_with_box)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)

        crop = img.crop((x1, y1, x2, y2))
        label = categories.get(ann["category_id"], "unknown")

        axes[i, 0].imshow(np.array(img_with_box))
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Wall image")

        axes[i, 1].imshow(np.array(crop))
        axes[i, 1].set_title(f"Color: {label}", fontsize=16, fontweight="bold")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
