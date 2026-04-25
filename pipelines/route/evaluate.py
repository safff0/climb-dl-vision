from dataclasses import dataclass

from common.geometry import bbox_iou
from common.types import Route, RouteState


@dataclass
class RouteEvalResult:
    precision_core: float
    recall_core: float
    precision_core_plus_possible: float
    recall_core_plus_possible: float
    f1: float
    missed_required_holds_count: int
    gt_total: int
    core_total: int
    possible_total: int


def evaluate_route(
    route: Route,
    gt_boxes: list[tuple[float, float, float, float]],
    iou_thr: float = 0.3,
) -> RouteEvalResult:
    core = route.core_holds()
    active = route.active_holds()
    gt_used_core = [False] * len(gt_boxes)
    tp_core = 0
    for h in core:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if gt_used_core[j]:
                continue
            iou = bbox_iou(h.bbox.to_list(), g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            tp_core += 1
            gt_used_core[best_j] = True
    fp_core = len(core) - tp_core

    gt_used_active = [False] * len(gt_boxes)
    tp_active = 0
    for h in active:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if gt_used_active[j]:
                continue
            iou = bbox_iou(h.bbox.to_list(), g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            tp_active += 1
            gt_used_active[best_j] = True
    fp_active = len(active) - tp_active

    prec_core = tp_core / max(1, len(core))
    rec_core = tp_core / max(1, len(gt_boxes))
    prec_act = tp_active / max(1, len(active))
    rec_act = tp_active / max(1, len(gt_boxes))
    f1 = 2 * prec_act * rec_act / max(1e-9, (prec_act + rec_act))
    missed = len(gt_boxes) - tp_active

    return RouteEvalResult(
        precision_core=prec_core,
        recall_core=rec_core,
        precision_core_plus_possible=prec_act,
        recall_core_plus_possible=rec_act,
        f1=f1,
        missed_required_holds_count=int(missed),
        gt_total=len(gt_boxes),
        core_total=len(core),
        possible_total=sum(1 for h in route.holds if h.route_state == RouteState.POSSIBLE.value),
    )
