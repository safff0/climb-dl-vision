import albumentations as A
import numpy as np


def build_train_transform(image_size: int, hold_color_sensitive: bool = False) -> A.Compose:
    h_shift = 5 if hold_color_sensitive else 12
    s_shift = 8 if hold_color_sensitive else 15
    v_shift = 10 if hold_color_sensitive else 18
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
                position="center",
            ),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.7,
            ),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=h_shift,
                sat_shift_limit=s_shift,
                val_shift_limit=v_shift,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                    A.GaussNoise(std_range=(0.0088, 0.0196), p=1.0),
                ],
                p=0.2,
            ),
            A.CLAHE(clip_limit=2.0, p=0.1),
        ]
    )


def build_val_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
                position="center",
            ),
        ]
    )


def apply_transform(sample: dict, transform: A.Compose) -> dict:
    image = sample["image"]
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    masks = sample["masks"]
    labels = sample["class_labels"]
    if not masks:
        res = transform(image=image, masks=[])
        return {
            "image": res["image"],
            "masks": [],
            "class_labels": [],
            "image_id": sample.get("image_id"),
        }

    res = transform(image=image, masks=masks)
    aug_masks = res["masks"]
    aug_labels = []
    keep_masks = []
    for m, l in zip(aug_masks, labels):
        if m.sum() >= 4:
            keep_masks.append(m)
            aug_labels.append(l)
    return {
        "image": res["image"],
        "masks": keep_masks,
        "class_labels": aug_labels,
        "image_id": sample.get("image_id"),
    }
