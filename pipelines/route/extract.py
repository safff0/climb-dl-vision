from dataclasses import dataclass

import numpy as np

from common.types import PhysicalHold, Route, RouteState
from pipelines.route.color_family import dominant_non_family, family_prob
from pipelines.route.graph import build_graph


@dataclass
class RouteExtractionConfig:
    target_color: str
    core_thr: float = 0.55
    possible_thr: float = 0.35
    rejected_strong_other_thr: float = 0.70
    track_k: float = 3.0
    weight_color: float = 0.50
    weight_graph: float = 0.25
    weight_track: float = 0.15
    weight_det: float = 0.10
    graph_radius_factor: float = 2.5
    min_frames_seen: int = 1
    colour_family_voting: bool = True
    propagation_iters: int = 3
    propagation_alpha: float = 0.5
    propagation_radius_factor: float = 2.0


def _track_score(frames_seen: int, total_keyframes: int, k: float) -> float:
    if total_keyframes <= 0:
        return 0.0
    ratio = min(1.0, frames_seen / max(1, total_keyframes))
    return float(1.0 - np.exp(-k * ratio))


def _propagate_label(
    initial: np.ndarray,
    graph,
    iters: int,
    alpha: float,
) -> np.ndarray:
    if not graph.holds:
        return initial
    current = initial.copy()
    for _ in range(iters):
        new = current.copy()
        for i, neigh in enumerate(graph.adj):
            if not neigh:
                continue
            w_sum = 0.0
            v_sum = 0.0
            for j in neigh:
                w = max(1e-6, graph.holds[j].det_conf_mean)
                v_sum += w * current[j]
                w_sum += w
            nbr = v_sum / max(w_sum, 1e-9)
            new[i] = (1.0 - alpha) * initial[i] + alpha * nbr
        current = new
    return current


def _family_graph_vote(graph, target_color: str, use_family: bool) -> np.ndarray:
    if not graph.holds:
        return np.zeros(0)
    n = len(graph.holds)
    scores = np.zeros(n, dtype=np.float64)
    for i, neigh in enumerate(graph.adj):
        if not neigh:
            continue
        wsum, wtotal = 0.0, 0.0
        for j in neigh:
            h = graph.holds[j]
            p = family_prob(target_color, h.color_probs_temporal, use_family)
            w = max(1e-6, h.det_conf_mean)
            wsum += p * w
            wtotal += w
        scores[i] = wsum / max(wtotal, 1e-9)
    return scores


def extract_route(
    holds: list[PhysicalHold],
    cfg: RouteExtractionConfig,
    total_keyframes: int,
) -> Route:
    if not holds:
        return Route(target_color=cfg.target_color, holds=[])

    graph = build_graph(holds, radius_factor=cfg.graph_radius_factor)
    prop_graph = build_graph(holds, radius_factor=cfg.propagation_radius_factor)

    p_color_local = np.array([
        family_prob(cfg.target_color, h.color_probs_temporal, cfg.colour_family_voting)
        for h in holds
    ], dtype=np.float64)

    p_color_final = _propagate_label(
        p_color_local, prop_graph, cfg.propagation_iters, cfg.propagation_alpha,
    )
    g_scores = _family_graph_vote(graph, cfg.target_color, cfg.colour_family_voting)

    for i, h in enumerate(holds):
        p_track = _track_score(len(h.frames_seen), total_keyframes, cfg.track_k)
        p_det = float(h.det_conf_mean)
        h.route_score = (
            cfg.weight_color * p_color_final[i]
            + cfg.weight_graph * g_scores[i]
            + cfg.weight_track * p_track
            + cfg.weight_det * p_det
        )
        h.color_score = float(p_color_final[i])
        h.graph_score = float(g_scores[i])
        h.track_score = float(p_track)
        h.det_score = float(p_det)

    for h in holds:
        p_local = family_prob(cfg.target_color, h.color_probs_temporal, cfg.colour_family_voting)
        other_name, other_p = dominant_non_family(cfg.target_color, h.color_probs_temporal)
        if h.route_score >= cfg.core_thr:
            h.route_state = RouteState.CORE.value
        elif h.route_score >= cfg.possible_thr:
            h.route_state = RouteState.POSSIBLE.value
        else:
            if other_p >= cfg.rejected_strong_other_thr and other_p > p_local * 1.5:
                h.route_state = RouteState.REJECTED.value
            else:
                h.route_state = RouteState.POSSIBLE.value if p_local > 0.15 else RouteState.UNKNOWN.value

    if not any(h.route_state == RouteState.CORE.value for h in holds):
        possible = [h for h in holds if h.route_state == RouteState.POSSIBLE.value]
        possible.sort(key=lambda h: h.route_score, reverse=True)
        for h in possible[:3]:
            h.route_state = RouteState.CORE.value

    return Route(target_color=cfg.target_color, holds=holds)
