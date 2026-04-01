import logging

import numpy as np

from common.config import cfg
from common.types import PipelineMode, Split
from data.handcrafted_features import extract_features_from_dataset
from models.color_handcrafted import HandcraftedColorClassifier
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


@register_pipeline("color_handcrafted", PipelineMode.TRAIN)
def run_train(model_name: str, output: str):
    logger.info("Extracting train features...")
    train_features, train_labels, class_names = extract_features_from_dataset(model_name, Split.TRAIN)
    logger.info("Train: %d samples, %d features", len(train_labels), train_features.shape[1])

    logger.info("Extracting valid features...")
    val_features, val_labels, _ = extract_features_from_dataset(model_name, Split.VALID)
    logger.info("Valid: %d samples", len(val_labels))

    model = HandcraftedColorClassifier()
    model.fit(train_features, train_labels, class_names)

    mapped_train = np.array([model.label_map.get(l, -1) for l in train_labels])
    mapped_val = np.array([model.label_map.get(l, -1) for l in val_labels])
    train_acc = (model.predict(train_features) == mapped_train).mean()
    val_acc = (model.predict(val_features) == mapped_val).mean()

    logger.info("Train accuracy: %.4f", train_acc)
    logger.info("Valid accuracy: %.4f", val_acc)

    model.save(output)
    logger.info("Saved to %s", output)
