import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from sklearn.cluster import KMeans

from common.color_normalization import apply_color_normalization
from common.config import cfg
from common.types import Split

logger = logging.getLogger(__name__)


def extract_color_features(
    crop_np: np.ndarray,
    mask_np: np.ndarray = None,
    hue_bins: int = 8,
    dominant_colors: int = 3,
) -> np.ndarray:
    if mask_np is None:
        mask_np = np.ones(crop_np.shape[:2], dtype=np.uint8)

    mask_bool = mask_np > 0
    if mask_bool.sum() < 10:
        mask_bool = np.ones(crop_np.shape[:2], dtype=bool)

    lab = cv2.cvtColor(crop_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV).astype(np.float32)

    masked_lab = lab[mask_bool]
    masked_hsv = hsv[mask_bool]

    l_vals = masked_lab[:, 0]
    a_vals = masked_lab[:, 1]
    b_vals = masked_lab[:, 2]
    features = [
        l_vals.mean(), l_vals.std(), np.median(l_vals),
        a_vals.mean(), a_vals.std(), np.median(a_vals),
        b_vals.mean(), b_vals.std(), np.median(b_vals),
    ]

    hue = masked_hsv[:, 0]
    hist, _ = np.histogram(hue, bins=hue_bins, range=(0, 180), density=True)
    features.extend(hist.tolist())

    sat = masked_hsv[:, 1]
    val = masked_hsv[:, 2]
    features.extend([sat.mean(), sat.std(), val.mean(), val.std()])

    if len(masked_lab) >= dominant_colors:
        km = KMeans(n_clusters=dominant_colors, random_state=42, n_init=3, max_iter=50)
        km.fit(masked_lab)
        centers = km.cluster_centers_
        counts = Counter(km.labels_)
        sorted_clusters = sorted(range(dominant_colors), key=lambda i: -counts.get(i, 0))
        for ci in sorted_clusters:
            features.extend(centers[ci].tolist())
    else:
        features.extend([0.0] * dominant_colors * 3)

    return np.array(features, dtype=np.float32)


def extract_features_from_dataset(
    model_name: str,
    split: Split,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mcfg = cfg.model_cfg(model_name)
    dataset_root = mcfg["dataset"]
    hue_bins = mcfg.get("hue_bins", 8)
    dominant_colors = mcfg.get("dominant_colors", 3)
    color_norm = mcfg.get("color_normalization", "none")

    split_dir = Path(dataset_root) / split
    coco = COCO(str(split_dir / "_annotations.coco.json"))

    categories = coco.loadCats(coco.getCatIds())
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names = [cat["name"] for cat in categories]

    all_features = []
    all_labels = []

    for ann_id, ann in coco.anns.items():
        x, y, w, h = ann["bbox"]
        if w < 1 or h < 1:
            continue
        cat_idx = cat_id_to_idx.get(ann["category_id"])
        if cat_idx is None:
            continue

        img_info = coco.loadImgs(ann["image_id"])[0]
        img = Image.open(split_dir / img_info["file_name"]).convert("RGB")
        img_np = np.array(img)

        if color_norm != "none":
            img_np = apply_color_normalization(img_np, color_norm)

        iw, ih = img.size
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(iw, int(x + w))
        y2 = min(ih, int(y + h))
        crop = img_np[y1:y2, x1:x2]

        seg = ann.get("segmentation", [])
        mask_crop = None
        if seg and not (isinstance(seg, list) and len(seg) == 0):
            full_mask = coco.annToMask(ann)
            mask_crop = full_mask[y1:y2, x1:x2]

        features = extract_color_features(crop, mask_crop, hue_bins, dominant_colors)
        all_features.append(features)
        all_labels.append(cat_idx)

    return np.array(all_features), np.array(all_labels), class_names
