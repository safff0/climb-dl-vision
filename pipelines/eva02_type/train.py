import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode, Split
from data.color_crops_dataset import CocoCropAlbDataset, compute_class_weights, get_type_dataloader
from models import create_model
from pipelines import register_pipeline
from pipelines.eva02_type.validate import evaluate_macro_f1

logger = logging.getLogger(__name__)


class _EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply_to(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                v.copy_(self.shadow[k])


def _cosine(step, total, warmup, min_factor):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(1.0, max(0.0, p))
    cos = 0.5 * (1 + math.cos(math.pi * p))
    return min_factor + (1 - min_factor) * cos


def _save_ema_weights(model, ema: _EMA, path: Path):
    from safetensors.torch import save_file

    raw = getattr(model, "_orig_mod", model)
    current = {k: v.detach().cpu().clone().contiguous() for k, v in raw.state_dict().items()}
    try:
        ema.apply_to(raw)
        sd = {k: v.detach().cpu().contiguous() for k, v in raw.state_dict().items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(path))
    finally:
        raw.load_state_dict(current)


@register_pipeline("eva02_type", PipelineMode.TRAIN)
def run_train(model_name: str, output: str):
    device = torch.device(cfg.torch.device)
    torch.manual_seed(cfg.torch.seed)
    np.random.seed(cfg.torch.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    mcfg = cfg.model_cfg(model_name)
    tcfg = cfg.train_cfg(model_name)

    amp_dtype_name = mcfg.get("amp_dtype", "bf16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32
    use_scaler = amp_dtype == torch.float16

    grad_accum = mcfg.get("grad_accum", 1)
    weight_decay = mcfg.get("weight_decay", 0.05)
    warmup_epochs = mcfg.get("warmup_epochs", 3)
    label_smoothing = mcfg.get("label_smoothing", 0.05)
    ema_decay = mcfg.get("ema_decay", 0.9998)

    model = create_model(model_name).to(device).to(memory_format=torch.channels_last)

    train_loader = get_type_dataloader(model_name, Split.TRAIN)
    val_loader = get_type_dataloader(model_name, Split.VALID)

    class_names = train_loader.dataset.class_names
    weights = compute_class_weights(train_loader.dataset).to(device)
    logger.info("class weights: %s", dict(zip(class_names, weights.tolist())))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = _EMA(model, ema_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    total_steps = steps_per_epoch * tcfg.epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(out_dir / "train_log.jsonl", "a")

    best_f1 = 0.0
    step = 0

    for epoch in range(tcfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{tcfg.epochs}", leave=False)
        running = 0.0
        for i, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits = model(imgs)
                loss = loss_fn(logits, labels) / grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss) * grad_accum

            if (i + 1) % grad_accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = tcfg.lr * _cosine(step, total_steps, warmup_steps, 0.01)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                step += 1
                pbar.set_postfix(loss=f"{running/(i+1):.3f}", lr=f"{lr:.2e}")

        macro_f1, per_class = evaluate_macro_f1(
            model, val_loader, device, amp_enabled, amp_dtype, class_names, ema=ema
        )
        entry = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "val_macro_f1_ema": macro_f1,
            "per_class_f1_ema": per_class,
        }
        log_fh.write(json.dumps(entry) + "\n")
        log_fh.flush()
        logger.info(json.dumps(entry))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            _save_ema_weights(model, ema, out_dir / "best_ema.safetensors")

    _save_ema_weights(model, ema, out_dir / "final_ema.safetensors")
    log_fh.close()
    logger.info("Best macro F1 (EMA): %.4f at %s", best_f1, out_dir / "best_ema.safetensors")
