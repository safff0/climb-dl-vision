from typing import Callable

import cv2
import numpy as np
import torch


def detector_tta_hflip_scales(
    image: np.ndarray,
    predict_batch_fn: Callable[[list[np.ndarray]], list[list[dict]]],
    scales: tuple[float, ...] = (0.75, 1.0, 1.25),
    hflip: bool = True,
) -> list[dict]:
    H, W = image.shape[:2]
    all_preds: list[dict] = []

    for s in scales:
        if s == 1.0:
            im = image
        else:
            im = cv2.resize(image, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR)

        batch = [im]
        if hflip:
            batch.append(np.ascontiguousarray(im[:, ::-1, :]))
        per_image = predict_batch_fn(batch)

        for p in per_image[0]:
            m = p["mask"]
            if s != 1.0:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            bx = np.array(p["bbox"], dtype=np.float32) / s
            all_preds.append({**p, "mask": m, "bbox": bx.tolist()})

        if hflip and len(per_image) > 1:
            for p in per_image[1]:
                m = p["mask"][:, ::-1]
                if s != 1.0:
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                bx = np.array(p["bbox"], dtype=np.float32)
                x0, y0, x1, y1 = bx.tolist()
                w_scaled = W * s
                bx = np.array([w_scaled - x1, y0, w_scaled - x0, y1]) / s
                all_preds.append({**p, "mask": m, "bbox": bx.tolist()})

    return all_preds


def classifier_tta_rot_flip(imgs: torch.Tensor, model, temperature: float = 1.0) -> torch.Tensor:
    B = imgs.shape[0]
    stacked = torch.cat(
        [
            imgs,
            torch.flip(imgs, dims=[3]),
            torch.rot90(imgs, 1, dims=(2, 3)),
            torch.rot90(imgs, 2, dims=(2, 3)),
            torch.rot90(imgs, 3, dims=(2, 3)),
        ],
        dim=0,
    )
    logits = model(stacked) / max(1e-3, temperature)
    probs = torch.softmax(logits, dim=1)
    return probs.view(5, B, -1).mean(dim=0)
