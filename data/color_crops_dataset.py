from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from common.config import cfg
from common.types import Split


def _read_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def build_color_transforms(image_size: int):
    train_tf = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, fill=0, position="center"),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(rotate=(-15, 15), scale=(0.85, 1.15), translate_percent=0.05, p=0.5, border_mode=0),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.GaussNoise(std_range=(0.0055, 0.0124), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ],
                p=0.2,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.4),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 24), hole_width_range=(8, 24), fill=0, p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_tf = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, fill=0, position="center"),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return train_tf, val_tf


class ColorCropsDataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: list[str], transform: A.Compose):
        self.root = Path(root) / split
        self.transform = transform
        self.class_names = class_names
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(class_names):
            d = self.root / cls
            if not d.exists():
                continue
            for p in sorted(d.glob("*.jpg")):
                self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = _read_rgb(path)
        out = self.transform(image=img)
        return out["image"], label

    def class_counts(self) -> np.ndarray:
        counter = Counter(label for _, label in self.samples)
        return np.array([counter.get(i, 1) for i in range(len(self.class_names))], dtype=np.float32)


def compute_class_weights(dataset: ColorCropsDataset) -> torch.Tensor:
    counts = dataset.class_counts()
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def get_color_dataloader(model_name: str, split: Split) -> DataLoader:
    mcfg = cfg.model_cfg(model_name)
    dataset_root = Path(mcfg["dataset"])
    image_size = mcfg.get("image_size", 448)
    class_names = list(mcfg["class_names"])
    train_cfg = cfg.train_cfg(model_name)
    val_cfg = cfg.validate_cfg(model_name)
    batch_size = train_cfg.batch_size if split == Split.TRAIN else val_cfg.batch_size

    split_name = {Split.TRAIN: "train", Split.VALID: "val", Split.TEST: "test"}[split]
    train_tf, val_tf = build_color_transforms(image_size)
    tf = train_tf if split == Split.TRAIN else val_tf
    dataset = ColorCropsDataset(dataset_root, split_name, class_names, tf)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == Split.TRAIN),
        num_workers=cfg.torch.num_workers,
        pin_memory=True,
        drop_last=(split == Split.TRAIN),
        persistent_workers=cfg.torch.num_workers > 0,
    )
