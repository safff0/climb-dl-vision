from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as cocomask
from torch.utils.data import DataLoader, Dataset

from common.config import cfg
from common.seg_augment import apply_transform, build_train_transform, build_val_transform
from common.types import Split


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _read_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _ann_to_mask(a: dict[str, Any], h: int, w: int) -> np.ndarray:
    seg = a.get("segmentation")
    if seg is None:
        return np.zeros((h, w), dtype=bool)
    if isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):
            rle = cocomask.frPyObjects(seg, h, w)
        else:
            rle = seg
        return cocomask.decode(rle).astype(bool)
    if isinstance(seg, list):
        if not seg:
            return np.zeros((h, w), dtype=bool)
        valid = [p for p in seg if len(p) >= 6]
        if not valid:
            return np.zeros((h, w), dtype=bool)
        rle = cocomask.frPyObjects(valid, h, w)
        merged = cocomask.merge(rle) if isinstance(rle, list) else rle
        return cocomask.decode(merged).astype(bool)
    return np.zeros((h, w), dtype=bool)


class CocoInstanceSegDataset(Dataset):
    def __init__(self, ann_path: Path | str, image_root: Path | str | None = None):
        import json

        self.ann_path = Path(ann_path)
        self.image_root = Path(image_root) if image_root else self.ann_path.parent
        with open(self.ann_path, "rb") as f:
            coco = json.loads(f.read())
        self.images = coco["images"]
        self.categories = coco["categories"]
        self.anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for a in coco["annotations"]:
            self.anns_by_image[a["image_id"]].append(a)
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(self.categories, key=lambda x: x["id"]))}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        im = self.images[idx]
        img_path = self.image_root / im["file_name"]
        image = _read_rgb(img_path)
        h, w = im["height"], im["width"]
        anns = self.anns_by_image.get(im["id"], [])

        masks = []
        labels = []
        for a in anns:
            coco_area = a.get("area", 0)
            if coco_area and coco_area < 4:
                continue
            m = _ann_to_mask(a, h, w)
            if m.sum() < 4:
                continue
            masks.append(m.astype(np.uint8))
            labels.append(self.cat_id_to_idx[a["category_id"]])

        return {
            "image": image,
            "masks": masks,
            "class_labels": labels,
            "image_id": im["id"],
            "file_name": im["file_name"],
        }


def worker_preprocess(sample: dict, mask_downscale: int = 4) -> dict[str, Any]:
    img = sample["image"] if isinstance(sample["image"], np.ndarray) else np.array(sample["image"])
    masks = sample["masks"]
    labels = sample["class_labels"]
    h, w = img.shape[:2]

    pixel_values = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    pixel_values = (pixel_values - _IMAGENET_MEAN) / _IMAGENET_STD

    if masks:
        n = len(masks)
        stacked = np.empty((n, h, w), dtype=np.uint8)
        for i, m in enumerate(masks):
            stacked[i] = m
        m_t = torch.from_numpy(stacked).float()
        if mask_downscale > 1:
            h_d = h // mask_downscale
            w_d = w // mask_downscale
            m_t = F.interpolate(m_t.unsqueeze(0), size=(h_d, w_d), mode="nearest").squeeze(0)
    else:
        size = (h // mask_downscale, w // mask_downscale) if mask_downscale > 1 else (h, w)
        m_t = torch.zeros((0, *size), dtype=torch.float32)

    return {
        "pixel_values": pixel_values,
        "mask_labels": m_t,
        "class_labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_mask2former(batch: list[dict]) -> dict[str, Any]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    return {
        "pixel_values": pixel_values,
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }


class _TransformedSeg(Dataset):
    def __init__(self, inner: Dataset, transform):
        self.inner = inner
        self.transform = transform

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        augmented = apply_transform(self.inner[i], self.transform)
        return worker_preprocess(augmented)


def get_instance_seg_dataloader(model_name: str, split: Split) -> DataLoader:
    mcfg = cfg.model_cfg(model_name)
    dataset_root = Path(mcfg["dataset"])
    image_size = mcfg.get("image_size", 1024)
    train_cfg = cfg.train_cfg(model_name)
    val_cfg = cfg.validate_cfg(model_name)
    batch_size = train_cfg.batch_size if split == Split.TRAIN else val_cfg.batch_size

    split_dir = dataset_root / str(split)
    ann_path = split_dir / "_annotations.coco.json"
    inner = CocoInstanceSegDataset(ann_path, image_root=split_dir)

    if split == Split.TRAIN:
        tf = build_train_transform(image_size, hold_color_sensitive=mcfg.get("hold_color_sensitive", False))
    else:
        tf = build_val_transform(image_size)

    dataset = _TransformedSeg(inner, tf)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == Split.TRAIN),
        num_workers=cfg.torch.num_workers,
        pin_memory=True,
        drop_last=(split == Split.TRAIN),
        collate_fn=collate_mask2former,
        persistent_workers=cfg.torch.num_workers > 0,
    )
