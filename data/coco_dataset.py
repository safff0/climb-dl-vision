from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common.config import cfg


class CocoDataset(Dataset):
    def __init__(self, root: str, split: str, transforms=None):
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


def collate_fn(batch):
    return tuple(zip(*batch))


def get_coco_dataloader(model_name: str, split: str) -> DataLoader:
    model_cfg = cfg.models.get(model_name, {})
    dataset_root = model_cfg.get("dataset", "")
    mode = "train" if split == "train" else "validate"
    mode_cfg = cfg.model_cfg(model_name, mode)
    batch_size = mode_cfg.get("batch_size", 1)

    transforms = RandomHorizontalFlip() if split == "train" else None
    dataset = CocoDataset(dataset_root, split, transforms=transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.torch.num_workers,
        collate_fn=collate_fn,
    )
