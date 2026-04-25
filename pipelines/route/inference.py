import json
from pathlib import Path

from common.config import cfg
from common.types import BBox, PhysicalHold, Route
from pipelines.route.color_aggregate import entropy
from pipelines.route.extract import RouteExtractionConfig, extract_route


def detections_to_physical_holds(
    detections: list[dict],
    color_class_names: list[str],
) -> list[PhysicalHold]:
    holds: list[PhysicalHold] = []
    for i, det in enumerate(detections):
        bbox_list = det.get("bbox") or det.get("box")
        if bbox_list is None:
            continue
        bbox = BBox.from_list(bbox_list)
        seg = det.get("class_name") or det.get("seg_class") or "hold"
        det_conf = float(det.get("det_conf", det.get("score", 0.0)))
        color_label = det.get("color") or "UNKNOWN"
        color_conf = float(det.get("color_conf", 0.0))
        color_probs_list = det.get("color_probs") or []
        if isinstance(color_probs_list, dict):
            color_probs = {k: float(v) for k, v in color_probs_list.items()}
        else:
            color_probs = {
                color_class_names[k]: float(v)
                for k, v in enumerate(color_probs_list)
                if k < len(color_class_names)
            }
        type_label = det.get("type") or det.get("hold_type") or "unknown"
        type_conf = float(det.get("type_conf", 0.0))
        type_probs_field = det.get("type_probs") or {}
        if isinstance(type_probs_field, dict):
            type_probs = {k: float(v) for k, v in type_probs_field.items()}
        else:
            type_probs = {}

        import numpy as np

        probs_arr = np.array(
            [color_probs.get(c, 0.0) for c in color_class_names], dtype=np.float64
        )
        hold = PhysicalHold(
            physical_track_id=str(i),
            bbox=bbox,
            center=(bbox.cx, bbox.cy),
            area=float(bbox.area),
            seg_class=seg,
            color_label_raw=color_label,
            color_conf_raw=color_conf,
            color_probs_raw=color_probs,
            color_label_temporal=color_label,
            color_conf_temporal=color_conf,
            color_probs_temporal=color_probs,
            color_entropy=entropy(probs_arr) if probs_arr.size else 0.0,
            type_label=type_label,
            type_conf=type_conf,
            type_probs_raw=type_probs,
            type_probs_temporal=type_probs,
            frames_seen=[0],
            det_conf_mean=det_conf,
            det_conf_max=det_conf,
            mask_rle=det.get("mask_rle"),
        )
        holds.append(hold)
    return holds


def run_route_extraction(
    predictions_path: str,
    target_color: str,
    output: str,
    color_model_config: str = "eva02_color",
    core_thr: float | None = None,
    possible_thr: float | None = None,
    rejected_strong_other_thr: float | None = None,
    track_k: float | None = None,
    weight_color: float | None = None,
    weight_graph: float | None = None,
    weight_track: float | None = None,
    weight_det: float | None = None,
    graph_radius_factor: float | None = None,
    propagation_iters: int | None = None,
    propagation_alpha: float | None = None,
    propagation_radius_factor: float | None = None,
    colour_family_voting: bool | None = None,
):
    ccfg = cfg.model_cfg(color_model_config)
    color_class_names = list(ccfg["class_names"])

    raw = json.loads(Path(predictions_path).read_text())

    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = [(d.get("image", f"image_{i}"), d.get("detections", [])) for i, d in enumerate(raw)]
    else:
        raise ValueError(f"unsupported predictions format at {predictions_path}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_kwargs: dict = {"target_color": target_color}
    overrides = {
        "core_thr": core_thr,
        "possible_thr": possible_thr,
        "rejected_strong_other_thr": rejected_strong_other_thr,
        "track_k": track_k,
        "weight_color": weight_color,
        "weight_graph": weight_graph,
        "weight_track": weight_track,
        "weight_det": weight_det,
        "graph_radius_factor": graph_radius_factor,
        "propagation_iters": propagation_iters,
        "propagation_alpha": propagation_alpha,
        "propagation_radius_factor": propagation_radius_factor,
        "colour_family_voting": colour_family_voting,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg_kwargs[k] = v
    rcfg = RouteExtractionConfig(**cfg_kwargs)

    routes: dict[str, dict] = {}
    for image_name, detections in items:
        holds = detections_to_physical_holds(detections, color_class_names)
        route = extract_route(holds, rcfg, total_keyframes=1)
        routes[image_name] = route.to_dict()

    out_path = out_dir / "routes.json"
    out_path.write_text(json.dumps(routes, indent=2))
    return routes
