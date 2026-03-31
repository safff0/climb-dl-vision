from enum import StrEnum

import numpy as np


class ColorNormMethod(StrEnum):
    NONE = "none"
    GRAY_WORLD = "gray_world"
    SHADES_OF_GRAY = "shades_of_gray"
    WHITE_BALANCE = "white_balance"


def gray_world(img: np.ndarray) -> np.ndarray:
    result = img.astype(np.float32)
    for c in range(3):
        mean = result[:, :, c].mean()
        if mean > 0:
            result[:, :, c] *= 128.0 / mean
    return np.clip(result, 0, 255).astype(np.uint8)


def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    result = img.astype(np.float32)
    for c in range(3):
        norm = np.power(result[:, :, c], p).mean() ** (1.0 / p)
        if norm > 0:
            result[:, :, c] *= 128.0 / norm
    return np.clip(result, 0, 255).astype(np.uint8)


def simple_white_balance(img: np.ndarray, percentile: int = 95) -> np.ndarray:
    result = img.astype(np.float32)
    for c in range(3):
        high = np.percentile(result[:, :, c], percentile)
        if high > 0:
            result[:, :, c] *= 255.0 / high
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_color_normalization(img: np.ndarray, method: str) -> np.ndarray:
    method = ColorNormMethod(method)
    if method == ColorNormMethod.NONE:
        return img
    if method == ColorNormMethod.GRAY_WORLD:
        return gray_world(img)
    if method == ColorNormMethod.SHADES_OF_GRAY:
        return shades_of_gray(img)
    if method == ColorNormMethod.WHITE_BALANCE:
        return simple_white_balance(img)
    return img
