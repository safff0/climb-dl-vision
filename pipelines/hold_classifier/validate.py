import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.config import cfg
from common.types import PipelineMode, Split
from data.crop_dataset import get_crop_dataloader, get_dataset_info
from models import create_model
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


@register_pipeline("hold_classifier", PipelineMode.VALIDATE)
def run_validate(model_name: str, weights: str):
    device = torch.device(cfg.torch.device)

    model = create_model(model_name).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    loader = get_crop_dataloader(model_name, Split.VALID)
    info = get_dataset_info(model_name)

    confusion = np.zeros((info.num_classes, info.num_classes), dtype=int)
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for pred, label in zip(preds.cpu(), labels.cpu()):
                p, l = pred.item(), label.item()
                confusion[l][p] += 1
                if p == l:
                    stats[l]["tp"] += 1
                else:
                    stats[p]["fp"] += 1
                    stats[l]["fn"] += 1

    total = confusion.sum()
    correct = confusion.diagonal().sum()
    accuracy = correct / max(total, 1)

    logger.info("Overall accuracy: %.4f (%d/%d)", accuracy, correct, total)
    logger.info("")
    logger.info("%-15s  %8s  %8s  %8s  %8s", "Class", "Prec", "Recall", "F1", "Support")
    logger.info("-" * 55)

    for i, name in enumerate(info.class_names):
        tp = stats[i]["tp"]
        fp = stats[i]["fp"]
        fn = stats[i]["fn"]
        support = tp + fn
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-8)
        logger.info("%-15s  %8.4f  %8.4f  %8.4f  %8d", name, prec, recall, f1, support)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(info.num_classes))
    ax.set_yticks(range(info.num_classes))
    ax.set_xticklabels(info.class_names, rotation=45, ha="right")
    ax.set_yticklabels(info.class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(info.num_classes):
        for j in range(info.num_classes):
            ax.text(j, i, str(confusion[i][j]), ha="center", va="center",
                    color="white" if confusion[i][j] > confusion.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()

    out_path = Path(weights).parent / f"{model_name}_confusion.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", out_path)
