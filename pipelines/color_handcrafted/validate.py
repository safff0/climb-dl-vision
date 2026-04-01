import logging
from collections import defaultdict

import numpy as np

from common.config import cfg
from common.types import PipelineMode, Split
from data.handcrafted_features import extract_features_from_dataset
from models.color_handcrafted import HandcraftedColorClassifier
from pipelines import register_pipeline

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


@register_pipeline("color_handcrafted", PipelineMode.VALIDATE)
def run_validate(model_name: str, weights: str):
    model = HandcraftedColorClassifier.load(weights)

    features, labels, class_names = extract_features_from_dataset(model_name, Split.VALID)
    mapped_labels = np.array([model.label_map.get(l, -1) for l in labels])
    preds = model.predict(features)

    correct = (preds == mapped_labels).sum()
    total = len(labels)
    accuracy = correct / max(total, 1)

    logger.info("Overall accuracy: %.4f (%d/%d)", accuracy, correct, total)
    logger.info("")
    logger.info("%-15s  %8s  %8s  %8s  %8s", "Class", "Prec", "Recall", "F1", "Support")
    logger.info("-" * 55)

    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p, l in zip(preds, mapped_labels):
        if p == l:
            stats[l]["tp"] += 1
        else:
            stats[p]["fp"] += 1
            stats[l]["fn"] += 1

    for i, name in enumerate(model.class_names):
        tp = stats[i]["tp"]
        fp = stats[i]["fp"]
        fn = stats[i]["fn"]
        support = tp + fn
        prec = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-8)
        logger.info("%-15s  %8.4f  %8.4f  %8.4f  %8d", name, prec, recall, f1, support)
