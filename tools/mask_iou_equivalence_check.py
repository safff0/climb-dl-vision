"""Equivalence check: mask_iou_crops against full-frame mask_iou.

For a set of synthetic mask pairs at different overlap configurations,
assert mask_iou_crops(tight_bbox, crop, area, ...) == mask_iou(full, full)
to exact equality (both are integer pixel-count ratios).
"""
from __future__ import annotations

import numpy as np

from pipeline.common.geometry import mask_iou, mask_iou_crops


def _random_mask(
    rng: np.random.Generator,
    H: int,
    W: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill: float,
) -> np.ndarray:
    """Binary mask on a (H,W) canvas with random pixels inside [x1..x2, y1..y2]."""
    m = np.zeros((H, W), dtype=bool)
    sub = rng.random((y2 - y1, x2 - x1)) < fill
    m[y1:y2, x1:x2] = sub
    return m


def _tight(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    crop = np.ascontiguousarray(mask[y1:y2, x1:x2])
    area = int(crop.sum())
    return (x1, y1, x2, y2), crop, area


def _one_case(rng: np.random.Generator, H: int, W: int, a_box, b_box, fill_a: float, fill_b: float) -> None:
    a_full = _random_mask(rng, H, W, *a_box, fill_a)
    b_full = _random_mask(rng, H, W, *b_box, fill_b)

    # Full-frame reference.
    ref = mask_iou(a_full, b_full)

    # Crop representation using detection bboxes (which fully enclose the mask).
    a_b, a_c, a_a = _tight(a_full, *a_box)
    b_b, b_c, b_a = _tight(b_full, *b_box)
    got = mask_iou_crops(a_b, a_c, a_a, b_b, b_c, b_a)

    assert ref == got, f"mismatch: ref={ref!r} got={got!r} a_box={a_box} b_box={b_box}"


def main() -> None:
    rng = np.random.default_rng(42)
    H, W = 720, 1280

    # Non-overlapping bboxes.
    _one_case(rng, H, W, (10, 10, 60, 60), (500, 500, 560, 560), 0.5, 0.5)
    # Touching but not overlapping.
    _one_case(rng, H, W, (100, 100, 200, 200), (200, 100, 300, 200), 0.5, 0.5)
    # Heavy overlap, small sizes.
    _one_case(rng, H, W, (100, 100, 200, 200), (110, 110, 210, 210), 0.6, 0.6)
    # Full containment.
    _one_case(rng, H, W, (200, 200, 400, 400), (250, 250, 350, 350), 0.7, 0.9)
    # Offset same size.
    _one_case(rng, H, W, (0, 0, 50, 50), (20, 20, 70, 70), 1.0, 1.0)
    # Large overlap large sizes.
    _one_case(rng, H, W, (100, 100, 600, 400), (200, 150, 700, 450), 0.3, 0.3)
    # At frame edge.
    _one_case(rng, H, W, (0, 0, 100, 100), (50, 50, 150, 150), 0.5, 0.5)
    _one_case(rng, H, W, (W - 80, H - 80, W, H), (W - 120, H - 120, W - 40, H - 40), 0.5, 0.5)
    # Near-empty mask.
    _one_case(rng, H, W, (100, 100, 200, 200), (100, 100, 200, 200), 0.02, 0.02)
    # Both identical.
    a_full = _random_mask(rng, H, W, 300, 300, 400, 400, 0.4)
    b_full = a_full.copy()
    ref = mask_iou(a_full, b_full)
    a_b, a_c, a_a = _tight(a_full, 300, 300, 400, 400)
    b_b, b_c, b_a = _tight(b_full, 300, 300, 400, 400)
    got = mask_iou_crops(a_b, a_c, a_a, b_b, b_c, b_a)
    assert ref == got == 1.0, (ref, got)

    # Fuzz: 200 random configurations.
    for _ in range(200):
        ax1 = int(rng.integers(0, W - 60))
        ay1 = int(rng.integers(0, H - 60))
        ax2 = ax1 + int(rng.integers(10, 200))
        ay2 = ay1 + int(rng.integers(10, 200))
        ax2 = min(ax2, W)
        ay2 = min(ay2, H)
        bx1 = ax1 + int(rng.integers(-80, 80))
        by1 = ay1 + int(rng.integers(-80, 80))
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = bx1 + int(rng.integers(10, 200))
        by2 = by1 + int(rng.integers(10, 200))
        bx2 = min(bx2, W)
        by2 = min(by2, H)
        if bx2 <= bx1 or by2 <= by1:
            continue
        fa = float(rng.uniform(0.05, 1.0))
        fb = float(rng.uniform(0.05, 1.0))
        _one_case(rng, H, W, (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2), fa, fb)

    print("mask_iou_crops equivalence: OK")


if __name__ == "__main__":
    main()
