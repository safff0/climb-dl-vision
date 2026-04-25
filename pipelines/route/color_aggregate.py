import math

import numpy as np


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        T = 1.0
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    log_p = log_p / T
    log_p = log_p - log_p.max()
    probs_t = np.exp(log_p)
    return probs_t / probs_t.sum()


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())


def max_entropy(n: int) -> float:
    return math.log(max(n, 2))
