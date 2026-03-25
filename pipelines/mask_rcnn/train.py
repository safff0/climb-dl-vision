from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from common.config import cfg
from data.coco_dataset import get_coco_dataloader
from models import create_model
from pipelines import register_pipeline
from pipelines.mask_rcnn.validate import evaluate_coco


@register_pipeline("mask_rcnn", "train")
def run_train(model_name: str, output: str):
    device = torch.device(cfg.torch.device)
    torch.manual_seed(cfg.torch.seed)

    model_cfg = cfg.model_cfg(model_name, "train")
    lr = model_cfg.get("lr", 0.005)
    epochs = model_cfg.get("epochs", 25)
    scheduler_type = model_cfg.get("scheduler", "cosine")
    milestones = model_cfg.get("lr_milestones", [16, 22])
    gamma = model_cfg.get("lr_gamma", 0.1)

    model = create_model(model_name).to(device)
    loader = get_coco_dataloader(model_name, "train")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_ap = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

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
        avg_loss = total_loss / max(len(loader), 1)

        metrics = evaluate_coco(model, model_name, device)
        segm_ap = metrics["segm_ap"]

        if segm_ap > best_ap:
            best_ap = segm_ap
            torch.save(model.state_dict(), str(out_path))

        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"loss: {avg_loss:.4f} — "
            f"segm_AP: {segm_ap:.4f} — "
            f"best_AP: {best_ap:.4f}"
        )

    last_path = out_path.with_stem(out_path.stem + "_last")
    torch.save(model.state_dict(), str(last_path))
    print(f"Best model: {out_path} (segm AP: {best_ap:.4f})")
    print(f"Last model: {last_path}")
