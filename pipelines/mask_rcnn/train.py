from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from common.config import cfg
from data.coco_dataset import get_coco_dataloader
from models import create_model
from pipelines import register_pipeline


@register_pipeline("mask_rcnn", "train")
def run_train(model_name: str, output: str):
    device = torch.device(cfg.torch.device)
    torch.manual_seed(cfg.torch.seed)

    model_cfg = cfg.model_cfg(model_name, "train")
    lr = model_cfg.get("lr", 0.005)
    epochs = model_cfg.get("epochs", 10)
    step_size = model_cfg.get("lr_step_size", 10)
    gamma = model_cfg.get("lr_gamma", 0.1)

    model = create_model(model_name).to(device)
    loader = get_coco_dataloader(model_name, "train")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        scheduler.step()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Saved to {output}")
