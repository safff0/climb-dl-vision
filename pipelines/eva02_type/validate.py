import logging
from pathlib import Path

import torch
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode, Split
from data.color_crops_dataset import get_type_dataloader
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


@torch.inference_mode()
def evaluate_macro_f1(model, loader, device, amp_enabled, amp_dtype, class_names, ema=None, tta: bool = True):
    from sklearn.metrics import f1_score

    raw = getattr(model, "_orig_mod", model)
    stashed = None
    if ema is not None:
        stashed = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
        ema.apply_to(raw)
    model.eval()

    pred_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            if tta:
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
                probs_all = torch.softmax(model(stacked), dim=1)
                logits_sum = probs_all.view(5, B, -1).sum(dim=0)
            else:
                logits_sum = torch.softmax(model(imgs), dim=1)
        pred_chunks.append(logits_sum.argmax(dim=1))
        label_chunks.append(labels)

    y_pred = torch.cat(pred_chunks).cpu().numpy()
    y_true = torch.cat(label_chunks).cpu().numpy()

    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0)
    per_class = {class_names[i]: float(per[i]) for i in range(len(class_names))}

    if stashed is not None:
        raw.load_state_dict(stashed)
    return macro, per_class


@register_pipeline("eva02_type", PipelineMode.VALIDATE)
def run_validate(model_name: str, weights: str):
    from safetensors.torch import load_file

    device = torch.device(cfg.torch.device)
    mcfg = cfg.model_cfg(model_name)

    amp_dtype_name = mcfg.get("amp_dtype", "bf16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32

    model = create_model(model_name).to(device).to(memory_format=torch.channels_last)
    state = load_file(str(Path(weights)))
    model.load_state_dict(state, strict=False)
    model.eval()

    loader = get_type_dataloader(model_name, Split.VALID)
    class_names = loader.dataset.class_names

    macro, per_class = evaluate_macro_f1(model, loader, device, amp_enabled, amp_dtype, class_names)
    logger.info("macro F1: %.4f", macro)
    for name, score in per_class.items():
        logger.info("  %s: %.4f", name, score)
