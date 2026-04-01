import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.ops import box_iou
from tqdm import tqdm

from common.color_normalization import apply_color_normalization
from common.config import cfg
from common.preprocessing import crop_and_normalize
from common.types import Split
from data.gnn_dataset import build_graph
from data.handcrafted_features import extract_color_features
from models import create_model
from models.color_handcrafted import HandcraftedColorClassifier

logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3


def _crop_and_classify(classifier, img_tensor, box, crop_size, device, mask=None):
    crop = crop_and_normalize(img_tensor, box, crop_size, padding=0, mask=mask)
    logits = classifier(crop.unsqueeze(0).to(device))
    return logits.squeeze(0).cpu()


def prepare_gnn_data(gnn_model_name: str):
    device = torch.device(cfg.torch.device)
    mcfg = cfg.model_cfg(gnn_model_name)
    k = mcfg.get("k_neighbors", 6)
    segmentor_model = mcfg["segmentor_model"]
    segmentor_weights = mcfg["segmentor_weights"]
    color_model = mcfg["color_model"]
    color_weights = mcfg["color_weights"]
    dataset_root = mcfg["dataset"]

    color_model_type = mcfg.get("color_model_type", "cnn")
    color_mcfg = cfg.model_cfg(color_model)

    segmentor = create_model(segmentor_model).to(device)
    segmentor.load_state_dict(torch.load(segmentor_weights, map_location=device, weights_only=True))
    segmentor.eval()

    cnn_classifier = None
    hc_classifier = None
    crop_size = 224
    use_mask = False
    hc_color_norm = "none"
    hc_config = {}

    if color_model_type == "catboost":
        hc_classifier = HandcraftedColorClassifier.load(color_weights)
        hc_config = color_mcfg
        hc_color_norm = color_mcfg.get("color_normalization", "none")
    else:
        crop_size = color_mcfg["crop_size"]
        use_mask = color_mcfg.get("use_mask_channel", False)
        cnn_classifier = create_model(color_model).to(device)
        cnn_classifier.load_state_dict(torch.load(color_weights, map_location=device, weights_only=True))
        cnn_classifier.eval()

    gt_label_map = None
    if hc_classifier is not None:
        gt_label_map = hc_classifier.label_map

    to_tensor = T.ToTensor()
    out_root = Path(dataset_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in [Split.TRAIN, Split.VALID]:
        color_dataset_root = color_mcfg["dataset"]
        split_dir = Path(color_dataset_root) / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            logger.warning("Skipping %s — no annotations", split)
            continue

        coco = COCO(str(ann_path))
        categories = coco.loadCats(coco.getCatIds())
        cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}

        graphs = []

        for img_id in tqdm(coco.getImgIds(), desc=f"[{split}] Building graphs"):
            img_info = coco.loadImgs(img_id)[0]
            img = Image.open(split_dir / img_info["file_name"]).convert("RGB")
            img_tensor = to_tensor(img).to(device)
            iw, ih = img.size

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            gt_boxes = []
            gt_labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w < 1 or h < 1:
                    continue
                cat_idx = cat_id_to_idx.get(ann["category_id"])
                if cat_idx is None:
                    continue
                gt_boxes.append([x, y, x + w, y + h])
                gt_labels.append(cat_idx)

            if len(gt_boxes) < 2:
                continue

            gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
            gt_labels_t = torch.tensor(gt_labels, dtype=torch.long)

            with torch.no_grad():
                pred = segmentor([img_tensor])[0]

            pred_boxes = pred["boxes"].cpu()
            pred_scores = pred["scores"].cpu()
            pred_masks = pred["masks"].cpu()
            keep = pred_scores > SCORE_THRESHOLD
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_masks = pred_masks[keep]

            final_boxes = []
            final_scores = []
            final_labels = []
            final_masks = []
            matched_gt = set()

            if len(pred_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes_t)
                for pred_i in range(len(pred_boxes)):
                    best_gt = ious[pred_i].argmax().item()
                    best_iou = ious[pred_i, best_gt].item()
                    if best_iou < IOU_THRESHOLD or best_gt in matched_gt:
                        continue
                    matched_gt.add(best_gt)
                    final_boxes.append(pred_boxes[pred_i])
                    final_scores.append(pred_scores[pred_i].item())
                    final_labels.append(gt_labels[best_gt])
                    mask = (pred_masks[pred_i, 0] > 0.5)
                    final_masks.append(mask)

            for gt_i in range(len(gt_boxes)):
                if gt_i in matched_gt:
                    continue
                final_boxes.append(gt_boxes_t[gt_i])
                final_scores.append(0.5)
                final_labels.append(gt_labels[gt_i])
                gt_box = gt_boxes_t[gt_i].int()
                bh = max(1, (gt_box[3] - gt_box[1]).item())
                bw = max(1, (gt_box[2] - gt_box[0]).item())
                final_masks.append(torch.ones(ih, iw))

            if len(final_boxes) < 2:
                continue

            boxes_t = torch.stack(final_boxes)
            scores_t = torch.tensor(final_scores)

            if gt_label_map is not None:
                mapped = [gt_label_map.get(l, -1) for l in final_labels]
                valid = [i for i, m in enumerate(mapped) if m >= 0]
                if len(valid) < 2:
                    continue
                boxes_t = boxes_t[valid]
                scores_t = scores_t[valid]
                final_labels = [mapped[i] for i in valid]
                final_masks = [final_masks[i] for i in valid]

            labels_t = torch.tensor(final_labels, dtype=torch.long)

            color_logits_list = []
            if hc_classifier is not None:
                img_np = np.array(img)
                if hc_color_norm != "none":
                    img_np = apply_color_normalization(img_np, hc_color_norm)
                for i in range(len(boxes_t)):
                    bx = boxes_t[i].int().tolist()
                    x1c, y1c = max(0, bx[0]), max(0, bx[1])
                    x2c, y2c = min(iw, bx[2]), min(ih, bx[3])
                    hc_crop = img_np[y1c:y2c, x1c:x2c]
                    hc_mask = final_masks[i][y1c:y2c, x1c:x2c].numpy().astype(np.uint8)
                    feats = extract_color_features(
                        hc_crop, hc_mask,
                        hc_config.get("hue_bins", 8),
                        hc_config.get("dominant_colors", 3),
                        hc_config.get("erode_pixels", 3),
                    )
                    proba = hc_classifier.predict_proba(feats.reshape(1, -1))[0]
                    color_logits_list.append(torch.tensor(proba, dtype=torch.float32))
            else:
                with torch.no_grad():
                    for i in range(len(boxes_t)):
                        mask_for_crop = final_masks[i] if use_mask else None
                        logits = _crop_and_classify(
                            cnn_classifier, img_tensor, boxes_t[i],
                            crop_size, device,
                            mask=mask_for_crop,
                        )
                        color_logits_list.append(logits)

            color_logits = torch.stack(color_logits_list)

            graph = build_graph(
                boxes_t, color_logits, scores_t,
                iw, ih, k=k, color_labels=labels_t,
            )
            if graph is not None:
                graphs.append(graph)

        out_path = out_root / f"{split}_graphs.pt"
        torch.save(graphs, out_path)
        logger.info("[%s] Saved %d graphs to %s", split, len(graphs), out_path)
