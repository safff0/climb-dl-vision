import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from common.config import cfg
from common.types import AugmentMode, CropMeta, DatasetInfo, Split

logger = logging.getLogger(__name__)


def _get_type_augmentations(crop_size):
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.Resize((crop_size, crop_size)),
        T.ToTensor(),
    ])


def _get_color_augmentations(crop_size):
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Resize((crop_size, crop_size)),
        T.ToTensor(),
    ])


def _get_val_transforms(crop_size):
    return T.Compose([
        T.Resize((crop_size, crop_size)),
        T.ToTensor(),
    ])


class CropDataset(Dataset):
    def __init__(self, root: str, split: Split, crop_size: int = 128,
                 padding: int = 16, use_mask: bool = True, transforms=None):
        self.root = Path(root) / split
        self.coco = COCO(str(self.root / "_annotations.coco.json"))
        self.crop_size = crop_size
        self.padding = padding
        self.use_mask = use_mask
        self.transforms = transforms

        categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_idx = {}
        self.class_names = []
        for idx, cat in enumerate(categories):
            self.cat_id_to_idx[cat["id"]] = idx
            self.class_names.append(cat["name"])
        self.num_classes = len(self.class_names)

        self.samples = []
        for ann_id, ann in self.coco.anns.items():
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            cat_idx = self.cat_id_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue
            self.samples.append((ann["image_id"], ann_id, cat_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, ann_id, label = self.samples[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.root / img_info["file_name"]).convert("RGB")
        ann = self.coco.loadAnns(ann_id)[0]

        x, y, w, h = ann["bbox"]
        iw, ih = img.size
        pad = self.padding
        x1 = max(0, int(x) - pad)
        y1 = max(0, int(y) - pad)
        x2 = min(iw, int(x + w) + pad)
        y2 = min(ih, int(y + h) + pad)
        crop = img.crop((x1, y1, x2, y2))

        if self.use_mask:
            mask = self.coco.annToMask(ann)
            mask_crop = mask[y1:y2, x1:x2]
            mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)

        if self.transforms:
            crop = self.transforms(crop)
        else:
            crop = T.Compose([T.Resize((self.crop_size, self.crop_size)), T.ToTensor()])(crop)

        if self.use_mask:
            mask_tensor = T.ToTensor()(mask_pil)
            crop = torch.cat([crop, mask_tensor], dim=0)

        return crop, label

    def get_class_weights(self) -> torch.Tensor:
        counts = Counter(s[2] for s in self.samples)
        total = len(self.samples)
        weights = torch.zeros(self.num_classes)
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / (self.num_classes * count)
        return weights

    def get_sample_weights(self) -> list[float]:
        counts = Counter(s[2] for s in self.samples)
        return [1.0 / counts[s[2]] for s in self.samples]


class SegmentorCropDataset(Dataset):
    def __init__(self, crops_dir: str, transforms=None):
        self.root = Path(crops_dir)
        with open(self.root / "labels.json") as f:
            self.meta = CropMeta.from_dict(json.load(f))
        self.class_names = self.meta.class_names
        self.num_classes = self.meta.num_classes
        self.transforms = transforms

    def __len__(self):
        return len(self.meta.crops)

    def __getitem__(self, idx):
        record = self.meta.crops[idx]
        img = Image.open(self.root / record.file).convert("RGB")
        label = record.label

        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        counts = Counter(c.label for c in self.meta.crops)
        total = len(self.meta.crops)
        weights = torch.zeros(self.num_classes)
        for cls_idx, count in counts.items():
            weights[cls_idx] = total / (self.num_classes * count)
        return weights

    def get_sample_weights(self) -> list[float]:
        counts = Counter(c.label for c in self.meta.crops)
        return [1.0 / counts[c.label] for c in self.meta.crops]


def get_crop_dataloader(model_name: str, split: Split) -> DataLoader:
    mcfg = cfg.model_cfg(model_name)
    dataset_root = mcfg["dataset"]
    crop_size = mcfg["crop_size"]
    padding = mcfg["crop_padding"]
    use_mask = mcfg["use_mask_channel"]
    augment_mode = AugmentMode(mcfg["augment_mode"])
    train_cfg = cfg.train_cfg(model_name)
    val_cfg = cfg.validate_cfg(model_name)
    batch_size = train_cfg.batch_size if split == Split.TRAIN else val_cfg.batch_size

    if split == Split.TRAIN:
        if augment_mode == AugmentMode.COLOR:
            transforms = _get_color_augmentations(crop_size)
        else:
            transforms = _get_type_augmentations(crop_size)
    else:
        transforms = _get_val_transforms(crop_size)

    segmentor_crops_dir = Path(dataset_root) / "segmentor_crops" / split
    if (segmentor_crops_dir / "labels.json").exists():
        logger.info("Using segmentor crops from %s", segmentor_crops_dir)
        dataset = SegmentorCropDataset(str(segmentor_crops_dir), transforms)
    else:
        dataset = CropDataset(dataset_root, split, crop_size, padding, use_mask, transforms)

    sampler = None
    shuffle = False
    if split == Split.TRAIN:
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.torch.num_workers,
    )


def get_dataset_info(model_name: str, split: Split = Split.TRAIN) -> DatasetInfo:
    mcfg = cfg.model_cfg(model_name)
    dataset_root = mcfg["dataset"]
    ann_path = Path(dataset_root) / split / "_annotations.coco.json"
    with open(ann_path) as f:
        data = json.load(f)
    categories = data["categories"]
    return DatasetInfo(
        num_classes=len(categories),
        class_names=[c["name"] for c in categories],
        cat_id_to_name={c["id"]: c["name"] for c in categories},
    )
