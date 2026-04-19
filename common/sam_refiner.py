import cv2
import numpy as np
import torch


class SAMRefiner:
    def __init__(self, model: str = "facebook/sam2.1-hiera-large", device: str | torch.device = "cuda") -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device)
        self.predictor = SAM2ImagePredictor.from_pretrained(model, device=self.device)
        self._image_set = False
        self._current_hw: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)
        self._image_set = True
        self._current_hw = image.shape[:2]

    def reset(self) -> None:
        reset_fn = getattr(self.predictor, "reset_predictor", None)
        if reset_fn is None:
            reset_fn = getattr(self.predictor, "reset_image", None)
        if reset_fn is not None:
            reset_fn()
        self._image_set = False
        self._current_hw = None

    def refine(
        self,
        boxes: list[list[float]],
        coarse_masks: list[np.ndarray] | None = None,
        multimask: bool = True,
        max_boxes_per_call: int = 64,
    ) -> list[dict]:
        if not self._image_set:
            raise RuntimeError("call set_image() before refine()")
        if not boxes:
            return []

        N = len(boxes)
        boxes_np = np.asarray(boxes, dtype=np.float32)

        mask_logits_np: np.ndarray | None = None
        if coarse_masks is not None and any(cm is not None and cm.any() for cm in coarse_masks):
            mask_logits_np = np.zeros((N, 1, 256, 256), dtype=np.float32)
            for i, cm in enumerate(coarse_masks):
                if cm is None or not cm.any():
                    continue
                m_r = cv2.resize(cm.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                mask_logits_np[i, 0] = (m_r.astype(np.float32) - 0.5) * 20.0

        results: list[dict] = []
        for start in range(0, N, max_boxes_per_call):
            end = min(start + max_boxes_per_call, N)
            chunk_boxes_np = boxes_np[start:end]
            chunk_mask_np = mask_logits_np[start:end] if mask_logits_np is not None else None
            with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
                mi, ucoords, lbl, ubox = self.predictor._prep_prompts(
                    None, None, chunk_boxes_np, chunk_mask_np, True
                )
                masks_t, iou_t, low_t = self.predictor._predict(
                    ucoords, lbl, ubox, mi,
                    multimask_output=multimask, return_logits=False,
                )

            if masks_t.ndim == 5:
                masks_t = masks_t.squeeze(0)
                iou_t = iou_t.squeeze(0)
                low_t = low_t.squeeze(0)
            n = masks_t.shape[0]
            best_idx_t = iou_t.argmax(dim=1)
            ar = torch.arange(n, device=masks_t.device)
            best_masks_t = masks_t[ar, best_idx_t].bool()
            best_iou_t = iou_t[ar, best_idx_t]
            best_low_t = low_t[ar, best_idx_t]

            upper = (best_low_t > 1.0).sum(dim=(1, 2)).float()
            lower = (best_low_t > -1.0).sum(dim=(1, 2)).float()
            stab_t = upper / (lower + 1e-6)

            best_masks_np = best_masks_t.cpu().numpy()
            best_iou_np = best_iou_t.float().cpu().numpy()
            stab_np = stab_t.float().cpu().numpy()
            for j in range(n):
                results.append({
                    "mask": best_masks_np[j],
                    "sam_iou": float(best_iou_np[j]),
                    "sam_stability": float(stab_np[j]),
                })
        return results
