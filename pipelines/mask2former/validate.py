import logging
from pathlib import Path

import torch
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode, Split
from data.coco_instance_dataset import get_instance_seg_dataloader
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


@torch.inference_mode()
def compute_val_loss(model, loader, device, amp_enabled: bool, amp_dtype=torch.bfloat16) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for batch in tqdm(loader, desc="val", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        mask_labels = [m.to(device, non_blocking=True) for m in batch["mask_labels"]]
        class_labels = [c.to(device, non_blocking=True) for c in batch["class_labels"]]
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            out = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        total = total + out.loss.detach()
        n += 1
    return float(total) / max(1, n)


@register_pipeline("mask2former", PipelineMode.VALIDATE)
def run_validate(model_name: str, weights: str):
    from transformers import Mask2FormerForUniversalSegmentation

    device = torch.device(cfg.torch.device)
    weights_path = Path(weights)
    if weights_path.is_dir():
        model = Mask2FormerForUniversalSegmentation.from_pretrained(weights_path).to(device)
    else:
        model = create_model(model_name).to(device)
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    mcfg = cfg.model_cfg(model_name)
    amp_dtype_name = mcfg.get("amp_dtype", "bf16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32

    loader = get_instance_seg_dataloader(model_name, Split.VALID)
    val_loss = compute_val_loss(model, loader, device, amp_enabled, amp_dtype)
    logger.info("val_loss: %.4f", val_loss)
