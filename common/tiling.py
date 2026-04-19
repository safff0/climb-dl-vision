from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    image: np.ndarray


def iter_tiles(image: np.ndarray, size: int = 1536, overlap: float = 0.25) -> Iterator[Tile]:
    h, w = image.shape[:2]
    if max(h, w) <= size:
        yield Tile(0, 0, w, h, image)
        return

    stride = int(round(size * (1 - overlap)))
    ys = list(range(0, max(1, h - size + 1), stride))
    if ys[-1] + size < h:
        ys.append(h - size)
    xs = list(range(0, max(1, w - size + 1), stride))
    if xs[-1] + size < w:
        xs.append(w - size)

    for y0 in ys:
        for x0 in xs:
            y1 = y0 + size
            x1 = x0 + size
            patch = image[y0:y1, x0:x1]
            yield Tile(x0, y0, x1, y1, patch)


def merge_instances_by_mask_iou(
    instances: list[dict],
    iou_thr: float = 0.7,
    containment_thr: float = 0.0,
    union: bool = True,
) -> list[dict]:
    if not instances:
        return []
    instances = sorted(instances, key=lambda x: x["score"], reverse=True)
    n = len(instances)
    boxes = np.array([inst["bbox"] for inst in instances], dtype=np.float32)
    classes = np.array([inst["class"] for inst in instances])
    areas = np.array([int(inst["mask"].sum()) for inst in instances], dtype=np.int64)

    kept_idxs: list[int] = []
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]:
            continue
        base_mask = instances[i]["mask"]
        base_area = int(areas[i])
        b_x0, b_y0, b_x1, b_y1 = float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3])
        for j in range(i + 1, n):
            if used[j] or classes[j] != classes[i]:
                continue
            j_x0, j_y0, j_x1, j_y1 = float(boxes[j, 0]), float(boxes[j, 1]), float(boxes[j, 2]), float(boxes[j, 3])
            ix0 = max(b_x0, j_x0); iy0 = max(b_y0, j_y0)
            ix1 = min(b_x1, j_x1); iy1 = min(b_y1, j_y1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            x0, y0, x1, y1 = int(ix0), int(iy0), int(ix1), int(iy1)
            sub_a = base_mask[y0:y1, x0:x1]
            sub_b = instances[j]["mask"][y0:y1, x0:x1]
            inter = int(np.logical_and(sub_a, sub_b).sum())
            if inter == 0:
                continue
            u = base_area + int(areas[j]) - inter
            if u <= 0:
                continue
            iou = inter / u
            j_area = int(areas[j])
            min_area = min(base_area, j_area)
            cont = inter / max(1, min_area)
            duplicate = iou > iou_thr or (containment_thr > 0 and cont > containment_thr)
            if duplicate:
                if union:
                    base_mask = np.logical_or(base_mask, instances[j]["mask"])
                    base_area = int(base_mask.sum())
                    b_x0 = min(b_x0, j_x0); b_y0 = min(b_y0, j_y0)
                    b_x1 = max(b_x1, j_x1); b_y1 = max(b_y1, j_y1)
                used[j] = True
        instances[i] = {
            **instances[i],
            "mask": base_mask,
            "bbox": [float(b_x0), float(b_y0), float(b_x1), float(b_y1)],
        }
        kept_idxs.append(i)
        used[i] = True
    return [instances[i] for i in kept_idxs]
