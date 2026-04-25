from dataclasses import dataclass

import numpy as np

from common.types import PhysicalHold


@dataclass
class HoldGraph:
    holds: list[PhysicalHold]
    edges: list[tuple[int, int]]
    adj: list[list[int]]
    radius: float


def _pairwise_min_dist(cs: np.ndarray) -> np.ndarray:
    if len(cs) < 2:
        return np.array([0.0])
    d = np.sqrt(((cs[:, None, :] - cs[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def build_graph(
    holds: list[PhysicalHold],
    radius_factor: float = 2.5,
    radius_min_px: float = 40.0,
    radius_max_px: float = 800.0,
) -> HoldGraph:
    if not holds:
        return HoldGraph(holds=[], edges=[], adj=[], radius=0.0)
    centers = np.asarray([h.center for h in holds], dtype=np.float64)
    min_dists = _pairwise_min_dist(centers)
    median_nn = float(np.median(min_dists[np.isfinite(min_dists)])) if np.isfinite(min_dists).any() else radius_min_px
    radius = float(np.clip(radius_factor * max(1.0, median_nn), radius_min_px, radius_max_px))
    d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1))
    edges: list[tuple[int, int]] = []
    adj: list[list[int]] = [[] for _ in holds]
    for i in range(len(holds)):
        for j in range(i + 1, len(holds)):
            if d[i, j] <= radius:
                edges.append((i, j))
                adj[i].append(j)
                adj[j].append(i)
    return HoldGraph(holds=holds, edges=edges, adj=adj, radius=radius)


def graph_consistency_score(graph: HoldGraph, target_color: str) -> np.ndarray:
    if not graph.holds:
        return np.zeros(0)
    n = len(graph.holds)
    scores = np.zeros(n, dtype=np.float64)
    for i, neigh in enumerate(graph.adj):
        if not neigh:
            scores[i] = 0.0
            continue
        wsum = 0.0
        wtotal = 0.0
        for j in neigh:
            h = graph.holds[j]
            p_target = h.color_probs_temporal.get(target_color, 0.0)
            w = max(1e-6, h.det_conf_mean)
            wsum += p_target * w
            wtotal += w
        scores[i] = wsum / max(wtotal, 1e-9)
    return scores


def connected_components(graph: HoldGraph) -> list[list[int]]:
    visited = [False] * len(graph.holds)
    comps: list[list[int]] = []
    for i in range(len(graph.holds)):
        if visited[i]:
            continue
        stack = [i]
        comp: list[int] = []
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            comp.append(v)
            stack.extend(graph.adj[v])
        comps.append(comp)
    return comps
