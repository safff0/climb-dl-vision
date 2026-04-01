import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.color_normalization import apply_color_normalization
from common.config import cfg
from common.preprocessing import crop_and_normalize
from data.handcrafted_features import extract_color_features
from models import create_model
from models.color_handcrafted import HandcraftedColorClassifier
from data.crop_dataset import get_dataset_info

SCORE_THRESHOLD = 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--segmentor-weights", required=True)
    parser.add_argument("--cnn-weights", required=True)
    parser.add_argument("--catboost-weights", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--output", default="fig_4_9.pdf")
    args = parser.parse_args()

    device = torch.device(cfg.torch.device)

    segmentor = create_model("mask_rcnn_hold").to(device)
    segmentor.load_state_dict(torch.load(args.segmentor_weights, map_location=device, weights_only=True))
    segmentor.eval()

    cnn_model_name = "hold_color_classifier"
    cnn_mcfg = cfg.model_cfg(cnn_model_name)
    cnn_crop_size = cnn_mcfg["crop_size"]
    cnn_use_mask = cnn_mcfg.get("use_mask_channel", False)
    cnn_color_norm = cnn_mcfg.get("color_normalization", "none")
    cnn = create_model(cnn_model_name).to(device)
    cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device, weights_only=True))
    cnn.eval()
    cnn_names = get_dataset_info(cnn_model_name).class_names

    hc_model = HandcraftedColorClassifier.load(args.catboost_weights)
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
    img_tensor = T.ToTensor()(img).to(device)

    cnn_img_tensor = img_tensor
    if cnn_color_norm != "none":
        cnn_img_tensor = T.ToTensor()(
            Image.fromarray(apply_color_normalization(img_np, cnn_color_norm))
        ).to(device)

    hc_img_np = img_np
    if hc_color_norm != "none":
        hc_img_np = apply_color_normalization(img_np, hc_color_norm)

    with torch.no_grad():
        pred = segmentor([img_tensor])[0]

    keep = pred["scores"] > SCORE_THRESHOLD
    boxes = pred["boxes"][keep].cpu()
    masks = (pred["masks"][keep] > 0.5).cpu()
    labels = pred["labels"][keep].cpu()

    hold_indices = [i for i in range(len(labels)) if labels[i].item() in hold_ids][:args.n]

    rows = len(hold_indices)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(hold_indices):
        box = boxes[idx]
        det_mask = masks[idx, 0]
        bx = box.int().tolist()
        x1, y1, x2, y2 = max(0, bx[0]), max(0, bx[1]), min(iw, bx[2]), min(ih, bx[3])

        mask_np = det_mask.numpy().astype(np.uint8)
        vis_crop = img_np[y1:y2, x1:x2].copy()
        vis_crop[mask_np[y1:y2, x1:x2] == 0] = 0

        with torch.no_grad():
            cnn_mask = det_mask if cnn_use_mask else None
            cnn_crop = crop_and_normalize(cnn_img_tensor.cpu(), box, cnn_crop_size, padding=0, mask=cnn_mask)
            cnn_logits = cnn(cnn_crop.unsqueeze(0).to(device))
            cnn_probs = F.softmax(cnn_logits, dim=1).squeeze(0).cpu()
            cnn_pred = cnn_names[cnn_probs.argmax().item()]
            cnn_conf = cnn_probs.max().item()

        hc_crop = hc_img_np[y1:y2, x1:x2]
        hc_mask_crop = mask_np[y1:y2, x1:x2]
        hc_feats = extract_color_features(
            hc_crop, hc_mask_crop,
            hc_config.get("hue_bins", 8),
            hc_config.get("dominant_colors", 3),
            hc_config.get("erode_pixels", 3),
        )
        hc_proba = hc_model.predict_proba(hc_feats.reshape(1, -1))[0]
        hc_pred_idx = int(hc_proba.argmax())
        hc_pred = hc_model.class_names[hc_pred_idx]
        hc_conf = hc_proba[hc_pred_idx]

        axes[row, 0].imshow(vis_crop)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Masked crop", fontsize=14, fontweight="bold")

        axes[row, 1].imshow(vis_crop)
        axes[row, 1].set_title(f"CNN: {cnn_pred} ({cnn_conf:.2f})", fontsize=14, fontweight="bold",
                               color="green" if cnn_pred == hc_pred else "black")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(vis_crop)
        axes[row, 2].set_title(f"CatBoost: {hc_pred} ({hc_conf:.2f})", fontsize=14, fontweight="bold",
                               color="green" if cnn_pred == hc_pred else "black")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
