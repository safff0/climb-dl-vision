import logging
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader

from common.config import cfg
from common.types import PipelineMode
from data.crop_dataset import get_dataset_info
from data.gnn_dataset import GNNGraphDataset
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


@register_pipeline("color_gnn", PipelineMode.VALIDATE)
def run_validate(model_name: str, weights: str):
    device = torch.device(cfg.torch.device)
    mcfg = cfg.model_cfg(model_name)
    dataset_root = mcfg["dataset"]
    color_model = mcfg.get("color_model", "hold_color_classifier")
    val_cfg = cfg.validate_cfg(model_name)

    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    dataset = GNNGraphDataset(f"{dataset_root}/valid_graphs.pt")
    loader = DataLoader(dataset, batch_size=val_cfg.batch_size)
    info = get_dataset_info(color_model)

    color_correct = 0
    color_total = 0
    route_correct = 0
    route_total = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

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

            for pred, label in zip(color_preds.cpu(), batch.y.cpu()):
                p, l = pred.item(), label.item()
                if p == l:
                    per_class[l]["tp"] += 1
                else:
                    per_class[p]["fp"] += 1
                    per_class[l]["fn"] += 1

    color_acc = color_correct / max(color_total, 1)
    route_acc = route_correct / max(route_total, 1)

    logger.info("Color accuracy: %.4f (%d/%d)", color_acc, color_correct, color_total)
    logger.info("Route accuracy: %.4f (%d/%d)", route_acc, route_correct, route_total)
    logger.info("")
    logger.info("%-15s  %8s  %8s  %8s", "Class", "Prec", "Recall", "F1")
    logger.info("-" * 47)

    for i, name in enumerate(info.class_names):
        tp = per_class[i]["tp"]
        fp = per_class[i]["fp"]
        fn = per_class[i]["fn"]
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-8)
        logger.info("%-15s  %8.4f  %8.4f  %8.4f", name, prec, recall, f1)
