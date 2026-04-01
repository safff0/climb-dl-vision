import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.color_normalization import apply_color_normalization
from common.config import cfg
from data.handcrafted_features import extract_color_features
from models import create_model
from models.color_handcrafted import HandcraftedColorClassifier

SCORE_THRESHOLD = 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--segmentor-weights", required=True)
    parser.add_argument("--color-weights", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--output", default="fig_4_8.pdf")
    args = parser.parse_args()

    device = torch.device(cfg.torch.device)

    segmentor = create_model("mask_rcnn_hold").to(device)
    segmentor.load_state_dict(torch.load(args.segmentor_weights, map_location=device, weights_only=True))
    segmentor.eval()

    hc_model = HandcraftedColorClassifier.load(args.color_weights)
    hc_config = cfg.model_cfg("hold_color_catboost")
    hc_color_norm = hc_config.get("color_normalization", "none")

    seg_mcfg = cfg.model_cfg("mask_rcnn_hold")
    seg_ann = Path(seg_mcfg["dataset"]) / "train" / "_annotations.coco.json"
    with open(seg_ann) as f:
        seg_cats = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    hold_ids = {cid for cid, n in seg_cats.items() if n.lower() == "hold"}

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)
    ih, iw = img_np.shape[:2]

    hc_img_np = img_np
    if hc_color_norm != "none":
        hc_img_np = apply_color_normalization(img_np, hc_color_norm)

    img_tensor = T.ToTensor()(img).to(device)

    with torch.no_grad():
        pred = segmentor([img_tensor])[0]

    keep = pred["scores"] > SCORE_THRESHOLD
    boxes = pred["boxes"][keep].cpu()
    masks = (pred["masks"][keep] > 0.5).cpu()
    labels = pred["labels"][keep].cpu()

    hold_indices = [i for i in range(len(labels)) if labels[i].item() in hold_ids][:args.n]

    cols = min(len(hold_indices), 3)
    rows = (len(hold_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for ax_i, idx in enumerate(hold_indices):
        box = boxes[idx].int().tolist()
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(iw, box[2]), min(ih, box[3])

        mask = masks[idx, 0].numpy().astype(np.uint8)
        crop = hc_img_np[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        masked = crop.copy()
        masked[mask_crop == 0] = 0

        feats = extract_color_features(
            crop, mask_crop,
            hc_config.get("hue_bins", 8),
            hc_config.get("dominant_colors", 3),
            hc_config.get("erode_pixels", 3),
        )
        proba = hc_model.predict_proba(feats.reshape(1, -1))[0]
        pred_idx = int(proba.argmax())
        pred_color = hc_model.class_names[pred_idx]
        conf = proba[pred_idx]

        r, c = ax_i // cols, ax_i % cols
        axes[r, c].imshow(masked)
        axes[r, c].set_title(f"{pred_color} ({conf:.2f})", fontsize=16, fontweight="bold")
        axes[r, c].axis("off")

    for ax_i in range(len(hold_indices), rows * cols):
        r, c = ax_i // cols, ax_i % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
