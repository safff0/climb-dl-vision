import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from common.config import cfg
from common.types import PipelineMode
from data.gnn_dataset import GNNGraphDataset
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


@register_pipeline("color_gnn", PipelineMode.TRAIN)
def run_train(model_name: str, output: str):
    device = torch.device(cfg.torch.device)
    torch.manual_seed(cfg.torch.seed)

    tcfg = cfg.train_cfg(model_name)
    mcfg = cfg.model_cfg(model_name)
    route_loss_weight = mcfg.get("route_loss_weight", 0.5)
    dataset_root = mcfg["dataset"]

    model = create_model(model_name).to(device)

    train_dataset = GNNGraphDataset(f"{dataset_root}/train_graphs.pt")
    val_dataset = GNNGraphDataset(f"{dataset_root}/valid_graphs.pt")
    train_loader = DataLoader(train_dataset, batch_size=tcfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=tcfg.batch_size)

    optimizer = AdamW(model.parameters(), lr=tcfg.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    color_criterion = nn.CrossEntropyLoss()
    route_criterion = nn.BCEWithLogitsLoss()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(tcfg.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{tcfg.epochs}", leave=False)

        for batch in pbar:
            batch = batch.to(device)
            color_logits, route_logits = model(batch)

            color_loss = color_criterion(color_logits, batch.y)
            route_loss = route_criterion(route_logits, batch.edge_labels)
            loss = color_loss + route_loss_weight * route_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)

        val_color_acc, val_route_acc = _evaluate(model, val_loader, device)

        if val_color_acc > best_acc:
            best_acc = val_color_acc
            torch.save(model.state_dict(), str(out_path))

        logger.info(
            "Epoch %d/%d\n"
            "  loss: %.4f\n"
            "  color_acc: %.4f  route_acc: %.4f  best_color_acc: %.4f\n"
            "  lr: %.6f",
            epoch + 1, tcfg.epochs,
            avg_loss,
            val_color_acc, val_route_acc, best_acc,
            optimizer.param_groups[0]["lr"],
        )

    last_path = out_path.with_stem(out_path.stem + "_last")
    torch.save(model.state_dict(), str(last_path))
    logger.info("Best model: %s (color acc: %.4f)", out_path, best_acc)


def _evaluate(model, loader, device):
    model.eval()
    color_correct = 0
    color_total = 0
    route_correct = 0
    route_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            color_logits, route_logits = model(batch)

            color_preds = color_logits.argmax(dim=1)
            color_correct += (color_preds == batch.y).sum().item()
            color_total += batch.y.size(0)

            route_preds = (route_logits > 0).float()
            route_correct += (route_preds == batch.edge_labels).sum().item()
            route_total += batch.edge_labels.size(0)

    color_acc = color_correct / max(color_total, 1)
    route_acc = route_correct / max(route_total, 1)
    return color_acc, route_acc
