from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common.config import cfg
from common.types import Split


class CocoDataset(Dataset):
    def __init__(self, root: str, split: Split, transforms=None):
        self.root = Path(root) / split
        self.coco = COCO(str(self.root / "_annotations.coco.json"))
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(self.root / img_info["file_name"]).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        img_tensor = T.ToTensor()(img)

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = image.flip(-1)
            w = image.shape[-1]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
            target["masks"] = target["masks"].flip(-1)
        return image, target


class ColorJitterDetection:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, image, target):
        image = self.jitter(image)
        return image, target


class RandomCropDetection:
    def __init__(self, min_scale=0.7):
        self.min_scale = min_scale

    def __call__(self, image, target):
        _, h, w = image.shape
        scale = torch.empty(1).uniform_(self.min_scale, 1.0).item()
        new_h, new_w = int(h * scale), int(w * scale)
        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        image = image[:, top:top + new_h, left:left + new_w]

        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = (widths > 2) & (heights > 2)

        target["boxes"] = boxes[keep]
        target["labels"] = target["labels"][keep]
        target["masks"] = target["masks"][keep][:, top:top + new_h, left:left + new_w]
        target["area"] = target["area"][keep]
        target["iscrowd"] = target["iscrowd"][keep]

        return image, target


class ComposeDetection:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_coco_dataloader(model_name: str, split: Split) -> DataLoader:
    mcfg = cfg.model_cfg(model_name)
    dataset_root = mcfg["dataset"]
    val_cfg = cfg.validate_cfg(model_name)
    train_cfg = cfg.train_cfg(model_name)
    batch_size = train_cfg.batch_size if split == Split.TRAIN else val_cfg.batch_size

    if split == Split.TRAIN:
        transforms = ComposeDetection([
            RandomHorizontalFlip(),
            RandomCropDetection(min_scale=0.7),
            ColorJitterDetection(brightness=0.3, contrast=0.3, saturation=0.2),
        ])
    else:
        transforms = None
    dataset = CocoDataset(dataset_root, split, transforms=transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == Split.TRAIN),
        num_workers=cfg.torch.num_workers,
        collate_fn=collate_fn,
    )
