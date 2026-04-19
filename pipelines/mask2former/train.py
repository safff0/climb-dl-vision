import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode, Split
from data.coco_instance_dataset import get_instance_seg_dataloader
from models import create_model
from pipelines import register_pipeline
from pipelines.mask2former.validate import compute_val_loss

logger = logging.getLogger(__name__)


class _EMA:
    def __init__(self, model, decay: float = 0.9998):
        self.decay = decay
        self.keys: list[str] = []
        self.model_refs: list[torch.Tensor] = []
        self.shadow_tensors: list[torch.Tensor] = []
        for name, p in model.named_parameters():
            if p.dtype.is_floating_point:
                self.keys.append(name)
                self.model_refs.append(p.data)
                self.shadow_tensors.append(p.data.detach().clone())
        for name, b in model.named_buffers():
            if b.dtype.is_floating_point:
                self.keys.append(name)
                self.model_refs.append(b)
                self.shadow_tensors.append(b.detach().clone())

    @torch.no_grad()
    def update(self):
        d = self.decay
        torch._foreach_mul_(self.shadow_tensors, d)
        torch._foreach_add_(self.shadow_tensors, self.model_refs, alpha=1 - d)

    def state_dict_with_ema(self, model) -> dict[str, torch.Tensor]:
        sd = dict(model.state_dict())
        for k, t in zip(self.keys, self.shadow_tensors):
            sd[k] = t
        return sd


def _cosine_with_warmup(step: int, total: int, warmup: int, min_factor: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return min_factor + (1 - min_factor) * cos


def _build_param_groups(
    model,
    base_lr: float,
    weight_decay: float,
    backbone_multiplier: float,
    no_decay_on_norm_bias: bool,
):
    backbone_ids: set[int] = set()
    try:
        bb = model.model.pixel_level_module.encoder
        for p in bb.parameters():
            backbone_ids.add(id(p))
    except AttributeError:
        pass

    bb_decay, bb_no_decay = [], []
    oth_decay, oth_no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm_or_bias = no_decay_on_norm_bias and (
            name.endswith(".bias") or ".norm" in name or "LayerNorm" in name or ".ln_" in name
        )
        is_backbone = id(p) in backbone_ids
        if is_backbone and is_norm_or_bias:
            bb_no_decay.append(p)
        elif is_backbone:
            bb_decay.append(p)
        elif is_norm_or_bias:
            oth_no_decay.append(p)
        else:
            oth_decay.append(p)

    groups = [
        {"params": bb_decay, "lr": base_lr * backbone_multiplier, "weight_decay": weight_decay, "_base_lr": base_lr * backbone_multiplier},
        {"params": bb_no_decay, "lr": base_lr * backbone_multiplier, "weight_decay": 0.0, "_base_lr": base_lr * backbone_multiplier},
        {"params": oth_decay, "lr": base_lr, "weight_decay": weight_decay, "_base_lr": base_lr},
        {"params": oth_no_decay, "lr": base_lr, "weight_decay": 0.0, "_base_lr": base_lr},
    ]
    return [g for g in groups if g["params"]]


@register_pipeline("mask2former", PipelineMode.TRAIN)
def run_train(model_name: str, output: str):
    from transformers import Mask2FormerImageProcessor

    mcfg = cfg.model_cfg(model_name)
    tcfg = cfg.train_cfg(model_name)

    torch.manual_seed(cfg.torch.seed)
    np.random.seed(cfg.torch.seed)
    device = torch.device(cfg.torch.device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    amp_dtype_name = mcfg.get("amp_dtype", "bf16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp_dtype_name]
    amp_enabled = amp_dtype != torch.float32
    use_scaler = amp_dtype == torch.float16

    grad_accum = mcfg.get("grad_accum", 1)
    weight_decay = mcfg.get("weight_decay", 0.05)
    warmup_steps = mcfg.get("warmup_steps", 1000)
    min_lr_factor = mcfg.get("min_lr_factor", 0.01)
    grad_clip = mcfg.get("grad_clip", 1.0)
    backbone_multiplier = mcfg.get("backbone_multiplier", 0.1)
    no_decay_on_norm_bias = mcfg.get("no_decay_on_norm_bias", True)
    ema_decay = mcfg.get("ema_decay", 0.9998)
    channels_last = mcfg.get("channels_last", True)
    val_every_epochs = mcfg.get("val_every_epochs", 2)

    model = create_model(model_name)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.to(device)

    train_loader = get_instance_seg_dataloader(model_name, Split.TRAIN)
    val_loader = get_instance_seg_dataloader(model_name, Split.VALID)

    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * tcfg.epochs

    param_groups = _build_param_groups(
        model,
        base_lr=tcfg.lr,
        weight_decay=weight_decay,
        backbone_multiplier=backbone_multiplier,
        no_decay_on_norm_bias=no_decay_on_norm_bias,
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=tcfg.lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = _EMA(model, decay=ema_decay)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor = Mask2FormerImageProcessor.from_pretrained(
        mcfg["model_name_or_path"],
        ignore_index=255,
        do_resize=False,
        do_rescale=True,
        do_normalize=True,
    )

    log_fh = open(out_dir / "train_log.jsonl", "a")
    best_val = float("inf")
    global_step = 0

    for epoch in range(tcfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        loss_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{tcfg.epochs}", leave=False)
        for i, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last if channels_last else torch.contiguous_format,
            )
            mask_labels = [m.to(device, non_blocking=True) for m in batch["mask_labels"]]
            class_labels = [c.to(device, non_blocking=True) for c in batch["class_labels"]]

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                out = model(
                    pixel_values=pixel_values,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = out.loss / grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum = loss_accum + loss.detach() * grad_accum
            loss_steps += 1

            if (i + 1) % grad_accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                lr_factor = _cosine_with_warmup(global_step, total_steps, warmup_steps, min_lr_factor)
                for pg in optimizer.param_groups:
                    base_lr = pg.get("_base_lr", tcfg.lr)
                    pg["lr"] = base_lr * lr_factor
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update()
                global_step += 1
                if global_step % 20 == 0:
                    avg_loss = (loss_accum / max(1, loss_steps)).item()
                    pbar.set_postfix(loss=f"{avg_loss:.3f}", lr=f"{tcfg.lr * lr_factor:.2e}", step=global_step)

        avg_train_loss = float(loss_accum / max(1, loss_steps))
        entry = {"epoch": epoch, "train_loss": avg_train_loss, "step": global_step}

        if (epoch + 1) % val_every_epochs == 0 or epoch == tcfg.epochs - 1:
            val_loss = compute_val_loss(model, val_loader, device, amp_enabled, amp_dtype)
            entry["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                best_dir = out_dir / "best_ema"
                best_dir.mkdir(parents=True, exist_ok=True)
                ema_sd = ema.state_dict_with_ema(model)
                model.save_pretrained(best_dir, state_dict=ema_sd, safe_serialization=True)
                processor.save_pretrained(best_dir)

        log_fh.write(json.dumps(entry) + "\n")
        log_fh.flush()
        logger.info(json.dumps(entry))

    final_dir = out_dir / "final_ema"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_sd = ema.state_dict_with_ema(model)
    model.save_pretrained(final_dir, state_dict=final_sd, safe_serialization=True)
    processor.save_pretrained(final_dir)
    log_fh.close()
    logger.info("Best EMA: %s (val_loss %.4f)", out_dir / "best_ema", best_val)
    logger.info("Final EMA: %s", final_dir)
