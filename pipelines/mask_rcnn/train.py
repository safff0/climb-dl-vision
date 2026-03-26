import logging
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode, SchedulerType, Split
from data.coco_dataset import get_coco_dataloader
from models import create_model
from pipelines import register_pipeline
from pipelines.mask_rcnn.validate import compute_val_loss, evaluate_coco

logger = logging.getLogger(__name__)


@register_pipeline("mask_rcnn", PipelineMode.TRAIN)
def run_train(model_name: str, output: str):
    device = torch.device(cfg.torch.device)
    torch.manual_seed(cfg.torch.seed)

    tcfg = cfg.train_cfg(model_name)

    model = create_model(model_name).to(device)
    loader = get_coco_dataloader(model_name, Split.TRAIN)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=tcfg.lr, momentum=0.9, weight_decay=0.0005)

    if tcfg.scheduler == SchedulerType.COSINE:
        scheduler = CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    else:
        scheduler = MultiStepLR(optimizer, milestones=tcfg.lr_milestones, gamma=tcfg.lr_gamma)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_ap = 0.0

    for epoch in range(tcfg.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{tcfg.epochs}", leave=False)

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            pbar.set_postfix(loss=f"{losses.item():.4f}")

        scheduler.step()
        avg_train_loss = total_loss / max(len(loader), 1)

        val_loss = compute_val_loss(model, model_name, device)
        metrics = evaluate_coco(model, model_name, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if metrics.segm_ap > best_ap:
            best_ap = metrics.segm_ap
            torch.save(model.state_dict(), str(out_path))

        logger.info(
            "Epoch %d/%d\n"
            "  train_loss: %.4f  val_loss: %.4f\n"
            "  bbox_AP: %.4f  segm_AP: %.4f  best_segm_AP: %.4f\n"
            "  lr: %.6f",
            epoch + 1, tcfg.epochs,
            avg_train_loss, val_loss,
            metrics.bbox_ap, metrics.segm_ap, best_ap,
            current_lr,
        )

    last_path = out_path.with_stem(out_path.stem + "_last")
    torch.save(model.state_dict(), str(last_path))
    logger.info("Best model: %s (segm AP: %.4f)", out_path, best_ap)
    logger.info("Last model: %s", last_path)
