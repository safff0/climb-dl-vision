import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import cfg
from models import create_model

SCORE_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="fig_4_3.pdf")
    args = parser.parse_args()

    device = torch.device(cfg.torch.device)
    model = create_model("mask_rcnn_hold").to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    seg_mcfg = cfg.model_cfg("mask_rcnn_hold")
    seg_ann = Path(seg_mcfg["dataset"]) / "train" / "_annotations.coco.json"
    import json
    with open(seg_ann) as f:
        seg_cats = {c["id"]: c["name"] for c in json.load(f)["categories"]}

    hold_ids = {cid for cid, n in seg_cats.items() if n.lower() == "hold"}
    volume_ids = {cid for cid, n in seg_cats.items() if n.lower() == "volume"}

    img = Image.open(args.image).convert("RGB")
    img_tensor = T.ToTensor()(img).to(device)

    with torch.no_grad():
        pred = model([img_tensor])[0]

    keep = pred["scores"] > SCORE_THRESHOLD
    boxes = pred["boxes"][keep].cpu()
    masks = (pred["masks"][keep] > 0.5).squeeze(1).cpu()
    labels = pred["labels"][keep].cpu()
    scores = pred["scores"][keep].cpu()

    img_uint8 = (img_tensor.cpu() * 255).to(torch.uint8)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(18, 6))

    vis_all = draw_segmentation_masks(img_uint8.clone(), masks, alpha=0.4)
    label_strs = [f"{seg_cats.get(l.item(), '?')} {s:.2f}" for l, s in zip(labels, scores)]
    vis_all = draw_bounding_boxes(vis_all, boxes, labels=label_strs, width=4, font_size=18, colors="lime")
    ax_a.imshow(vis_all.permute(1, 2, 0).numpy())
    ax_a.set_title("(a) All detections")
    ax_a.axis("off")

    hold_mask = torch.tensor([l.item() in hold_ids for l in labels])
    if hold_mask.any():
        vis_hold = draw_segmentation_masks(img_uint8.clone(), masks[hold_mask], alpha=0.5)
        vis_hold = draw_bounding_boxes(vis_hold, boxes[hold_mask], width=4, colors="lime")
    else:
        vis_hold = img_uint8.clone()
    ax_b.imshow(vis_hold.permute(1, 2, 0).numpy())
    ax_b.set_title("(b) Holds only")
    ax_b.axis("off")

    vol_mask = torch.tensor([l.item() in volume_ids for l in labels])
    if vol_mask.any():
        vis_vol = draw_segmentation_masks(img_uint8.clone(), masks[vol_mask], alpha=0.5)
        vis_vol = draw_bounding_boxes(vis_vol, boxes[vol_mask], width=4, colors="cyan")
    else:
        vis_vol = img_uint8.clone()
    ax_c.imshow(vis_vol.permute(1, 2, 0).numpy())
    ax_c.set_title("(c) Volumes only")
    ax_c.axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
