import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode, Split
from data.crop_dataset import get_crop_dataloader
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def _freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False


def _unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def _evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1), total_loss / max(len(loader), 1)


@register_pipeline("hold_classifier", PipelineMode.TRAIN)
def run_train(model_name: str, output: str):
    device = torch.device(cfg.torch.device)
    torch.manual_seed(cfg.torch.seed)

    tcfg = cfg.train_cfg(model_name)

    model = create_model(model_name).to(device)
    train_loader = get_crop_dataloader(model_name, Split.TRAIN)
    val_loader = get_crop_dataloader(model_name, Split.VALID)

    class_weights = train_loader.dataset.get_class_weights().to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    _freeze_backbone(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=tcfg.freeze_backbone_lr)

    for epoch in range(tcfg.freeze_backbone_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[freeze] Epoch {epoch + 1}/{tcfg.freeze_backbone_epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc, val_loss = _evaluate(model, val_loader, criterion, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(out_path))

        logger.info(
            "[freeze] Epoch %d/%d\n"
            "  train_loss: %.4f  val_loss: %.4f  val_acc: %.4f  best_acc: %.4f",
            epoch + 1, tcfg.freeze_backbone_epochs, avg_loss, val_loss, val_acc, best_acc,
        )

    _unfreeze_all(model)
    finetune_epochs = tcfg.epochs - tcfg.freeze_backbone_epochs
    optimizer = AdamW(model.parameters(), lr=tcfg.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    for epoch in range(finetune_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[finetune] Epoch {epoch + 1}/{finetune_epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc, val_loss = _evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), str(out_path))

        logger.info(
            "[finetune] Epoch %d/%d\n"
            "  train_loss: %.4f  val_loss: %.4f\n"
            "  val_acc: %.4f  best_acc: %.4f\n"
            "  lr: %.6f",
            epoch + 1, finetune_epochs, avg_loss, val_loss, val_acc, best_acc, current_lr,
        )

    last_path = out_path.with_stem(out_path.stem + "_last")
    torch.save(model.state_dict(), str(last_path))
    logger.info("Best model: %s (acc: %.4f)", out_path, best_acc)
    logger.info("Last model: %s", last_path)
