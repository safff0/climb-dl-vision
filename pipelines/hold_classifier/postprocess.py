import logging
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from common.types import Detection

logger = logging.getLogger(__name__)

MIN_CLUSTERS = 2
MAX_CLUSTERS = 8
MIN_HOLDS_FOR_CLUSTERING = 4


def cluster_colors(detections: list[Detection], n_clusters: int = None) -> list[Detection]:
    hold_indices = [i for i, d in enumerate(detections) if d.color_probs is not None]

    if len(hold_indices) < MIN_HOLDS_FOR_CLUSTERING:
        return detections

    prob_vectors = np.array([
        list(detections[i].color_probs.values()) for i in hold_indices
    ])

    if n_clusters is None:
        n_clusters = _find_best_k(prob_vectors)

    if n_clusters < 2 or n_clusters >= len(hold_indices):
        return detections

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(prob_vectors)

    for cluster_id in range(n_clusters):
        members = [hold_indices[j] for j, c in enumerate(cluster_ids) if c == cluster_id]
        if not members:
            continue

        colors = [detections[i].color for i in members]
        majority_color = Counter(colors).most_common(1)[0][0]

        for i in members:
            detections[i].color_cluster = int(cluster_id)
            detections[i].color_clustered = majority_color

    logger.debug("Clustered %d holds into %d groups", len(hold_indices), n_clusters)
    return detections


def _find_best_k(vectors: np.ndarray) -> int:
    max_k = min(MAX_CLUSTERS, len(vectors) - 1)
    if max_k < MIN_CLUSTERS:
        return MIN_CLUSTERS

    best_k = MIN_CLUSTERS
    best_score = -1.0

    for k in range(MIN_CLUSTERS, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k
