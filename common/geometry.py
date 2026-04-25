from typing import Sequence

import numpy as np


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def bbox_center(b: Sequence[float]) -> tuple[float, float]:
    return 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ac = bbox_center(a)
    bc = bbox_center(b)
    return float(np.hypot(ac[0] - bc[0], ac[1] - bc[1]))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def mask_iou_crops(
    a_box: tuple[int, int, int, int],
    a_crop: np.ndarray,
    a_area: int,
    b_box: tuple[int, int, int, int],
    b_crop: np.ndarray,
    b_area: int,
) -> float:
    ax1, ay1, ax2, ay2 = a_box
    bx1, by1, bx2, by2 = b_box
    ix1 = ax1 if ax1 > bx1 else bx1
    iy1 = ay1 if ay1 > by1 else by1
    ix2 = ax2 if ax2 < bx2 else bx2
    iy2 = ay2 if ay2 < by2 else by2
    if ix1 >= ix2 or iy1 >= iy2:
        inter = 0
    else:
        a_sub = a_crop[iy1 - ay1:iy2 - ay1, ix1 - ax1:ix2 - ax1]
        b_sub = b_crop[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1]
        inter = int(np.logical_and(a_sub, b_sub).sum())
    union = int(a_area) + int(b_area) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def mask_containment(small: np.ndarray, large: np.ndarray) -> float:
    if small.shape != large.shape:
        return 0.0
    s = small.astype(bool)
    s_area = s.sum()
    if s_area == 0:
        return 0.0
    return float(np.logical_and(s, large.astype(bool)).sum() / s_area)


def distance_point_to_mask(
    point: tuple[float, float], mask: np.ndarray
) -> tuple[float, bool]:
    import cv2

    px, py = point
    H, W = mask.shape
    if H == 0 or W == 0:
        return float("inf"), False
    ix = int(np.clip(px, 0, W - 1))
    iy = int(np.clip(py, 0, H - 1))
    m = mask.astype(np.uint8)
    inside = bool(m[iy, ix])
    if inside:
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        return -float(dist[iy, ix]), True
    dist_outside = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist_outside[iy, ix]), False


def bbox_of_mask(mask: np.ndarray) -> list[float] | None:
    rows = mask.any(axis=1)
    if not rows.any():
        return None
    cols = mask.any(axis=0)
    y0 = int(rows.argmax())
    y1 = int(len(rows) - rows[::-1].argmax())
    x0 = int(cols.argmax())
    x1 = int(len(cols) - cols[::-1].argmax())
    return [float(x0), float(y0), float(x1), float(y1)]


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def signed_y_up(y: float, H: int) -> float:
    return float(H) - float(y)
