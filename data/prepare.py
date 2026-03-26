import json
import logging
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown

from common.config import cfg

logger = logging.getLogger(__name__)


def _download_and_extract(url: str, dest: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "dataset.zip"
        logger.info("Downloading from Google Drive...")
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
        logger.info("Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path / "extracted")
        extracted = tmp_path / "extracted"
        subdirs = list(extracted.iterdir())
        root = subdirs[0] if len(subdirs) == 1 and subdirs[0].is_dir() else extracted
        shutil.copytree(root, dest, dirs_exist_ok=True)
    return dest


def _load_coco_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _merge_splits(raw_dir: Path) -> dict:
    merged = {"images": [], "annotations": [], "categories": None}
    img_id_offset = 0
    ann_id_offset = 0
    all_images_dir = raw_dir / "_all_images"
    all_images_dir.mkdir(exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_dir = raw_dir / split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            logger.debug("Skipping %s — no annotations found", split)
            continue

        data = _load_coco_json(ann_path)

        if merged["categories"] is None:
            merged["categories"] = data["categories"]

        old_to_new_img = {}
        for img in data["images"]:
            old_id = img["id"]
            new_id = old_id + img_id_offset
            old_to_new_img[old_id] = new_id
            img["id"] = new_id
            merged["images"].append(img)

            src = split_dir / img["file_name"]
            if src.exists():
                shutil.copy2(src, all_images_dir / img["file_name"])

        for ann in data["annotations"]:
            ann["id"] = ann["id"] + ann_id_offset
            ann["image_id"] = old_to_new_img[ann["image_id"]]
            merged["annotations"].append(ann)

        if data["images"]:
            img_id_offset = max(img["id"] for img in merged["images"]) + 1
        if data["annotations"]:
            ann_id_offset = max(ann["id"] for ann in merged["annotations"]) + 1

        logger.info("Merged %s: %d images, %d annotations", split, len(data["images"]), len(data["annotations"]))

    return merged


def _split_dataset(merged: dict, val_split: float, seed: int) -> tuple[dict, dict]:
    random.seed(seed)
    images = merged["images"][:]
    random.shuffle(images)

    val_count = max(1, int(len(images) * val_split))
    val_images = images[:val_count]
    train_images = images[val_count:]

    val_img_ids = {img["id"] for img in val_images}
    train_img_ids = {img["id"] for img in train_images}

    train_anns = [a for a in merged["annotations"] if a["image_id"] in train_img_ids]
    val_anns = [a for a in merged["annotations"] if a["image_id"] in val_img_ids]

    train_data = {"images": train_images, "annotations": train_anns, "categories": merged["categories"]}
    val_data = {"images": val_images, "annotations": val_anns, "categories": merged["categories"]}

    return train_data, val_data


def _save_split(data: dict, images_src: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for img in data["images"]:
        src = images_src / img["file_name"]
        if src.exists():
            shutil.copy2(src, dest / img["file_name"])
    with open(dest / "_annotations.coco.json", "w") as f:
        json.dump(data, f)


def create_dataset(dataset_name: str, url: str):
    ds_cfg = cfg.datasets[dataset_name]
    out_dir = Path(ds_cfg.dir)
    val_split = ds_cfg.val_split

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        raw_dir.mkdir()
        _download_and_extract(url, raw_dir)

        logger.info("Merging splits...")
        merged = _merge_splits(raw_dir)
        logger.info("Total: %d images, %d annotations", len(merged["images"]), len(merged["annotations"]))

        train_data, val_data = _split_dataset(merged, val_split, cfg.torch.seed)
        logger.info("Split: %d train, %d valid (%.0f%% val)", len(train_data["images"]), len(val_data["images"]), val_split * 100)

        all_images = raw_dir / "_all_images"

        if out_dir.exists():
            shutil.rmtree(out_dir)

        _save_split(train_data, all_images, out_dir / "train")
        _save_split(val_data, all_images, out_dir / "valid")

    logger.info("Dataset saved to %s", out_dir)
