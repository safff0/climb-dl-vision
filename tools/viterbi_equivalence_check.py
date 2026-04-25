from __future__ import annotations

import math

import numpy as np

from pipeline.common.schemas import PhysicalHold, BBox, RouteState
from pipeline.contact.decode import (
    DecodeConfig, NO_CONTACT, OCCLUDED, PADDED, _build_state_list,
    _log_softmax, decode_limb,
)

def _reference_decode_limb(
    hold_logits: np.ndarray,
    cands_per_frame: list[list[PhysicalHold]],
    cfg: DecodeConfig | None = None,
    limb_conf: np.ndarray | None = None,
) -> list[int]:
    cfg = cfg or DecodeConfig()
    T, S = hold_logits.shape
    global_ids = _build_state_list(cands_per_frame)
    NG = len(global_ids)
    id_to_global = {g: i for i, g in enumerate(global_ids)}

    slot_to_global = np.full((T, S), id_to_global[PADDED], dtype=np.int64)
    slot_is_possible = np.zeros((T, S), dtype=bool)
    for t in range(T):
        cands = cands_per_frame[t]
        for s in range(S):
            if s == S - 2:
                slot_to_global[t, s] = id_to_global[NO_CONTACT]
            elif s == S - 1:
                slot_to_global[t, s] = id_to_global[OCCLUDED]
            elif s < len(cands):
                slot_to_global[t, s] = id_to_global[cands[s].physical_track_id]
                slot_is_possible[t, s] = cands[s].route_state == RouteState.POSSIBLE.value

    NEG = -1e9
    emissions = np.full((T, NG), NEG, dtype=np.float64)
    log_possible = math.log(max(1e-9, cfg.possible_prior))
    log_occluded = math.log(max(1e-9, cfg.occluded_prior))
    log_no_contact = math.log(max(1e-9, cfg.no_contact_prior))
    log_padded = math.log(max(1e-12, cfg.padded_prior))
    idx_no_contact = id_to_global[NO_CONTACT]
    idx_occluded = id_to_global[OCCLUDED]
    idx_padded = id_to_global[PADDED]

    def _is_real_hold_state(idx: int) -> bool:
        return idx not in {idx_no_contact, idx_occluded, idx_padded}

    for t in range(T):
        logp = _log_softmax(hold_logits[t], cfg.temperature)
        for s in range(S):
            gi = int(slot_to_global[t, s])
            logit = float(logp[s])
            if gi == idx_occluded:
                logit = logit + log_occluded
            elif gi == idx_no_contact:
                logit = logit + log_no_contact
            elif gi == idx_padded:
                logit = logit + log_padded
            else:
                if slot_is_possible[t, s]:
                    logit = logit + log_possible
            if logit > emissions[t, gi]:
                emissions[t, gi] = logit
        is_low = (
            limb_conf is not None
            and t < len(limb_conf)
            and limb_conf[t] < cfg.low_conf_threshold
        )
        if is_low:
            emissions[t, idx_occluded] += cfg.low_conf_occluded_bonus
            emissions[t, idx_no_contact] += cfg.low_conf_no_contact_bonus

    score = np.full((T, NG), NEG, dtype=np.float64)
    back = np.full((T, NG), -1, dtype=np.int64)
    score[0] = emissions[0]
    for t in range(1, T):
        prev = score[t - 1]
        prev_back = back[t - 1]
        is_low = (
            limb_conf is not None
            and t < len(limb_conf)
            and limb_conf[t] < cfg.low_conf_threshold
        )
        for g in range(NG):
            best = NEG
            best_gp = -1
            for gp in range(NG):
                s_ = prev[gp]
                if gp != g:
                    extra = 0.0
                    if is_low:
                        if _is_real_hold_state(gp) and _is_real_hold_state(g):
                            extra = cfg.low_conf_real_hold_switch_cost
                        else:
                            extra = cfg.low_conf_uncertain_switch_cost
                    s_ = s_ - cfg.switch_cost - extra
                    if t - 1 >= 1 and prev_back[gp] == g:
                        s_ = s_ - cfg.flap_cost
                if s_ > best:
                    best = s_
                    best_gp = gp
            if best_gp < 0:
                best_gp = 0
                best = prev[0]
            score[t, g] = best + emissions[t, g]
            back[t, g] = best_gp

    path = [int(score[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return path

def _fake_hold(i: int, state: str = "core") -> PhysicalHold:
    return PhysicalHold(
        physical_track_id=f"h{i}",
        bbox=BBox(0.0, 0.0, 1.0, 1.0),
        center=(float(i), float(i)),
        area=1.0,
        seg_class="hold",
        color_label_raw="Orange",
        color_conf_raw=1.0,
        color_probs_raw={"Orange": 1.0},
        color_label_temporal="Orange",
        color_conf_temporal=1.0,
        color_probs_temporal={"Orange": 1.0},
        color_entropy=0.0,
        type_label="jug",
        type_conf=0.9,
        route_state=state,
        route_score=0.9,
    )

def _run_case(seed: int, T: int, N_holds: int, K: int) -> None:
    rng = np.random.default_rng(seed)
    holds = [_fake_hold(i, "core" if i % 3 else "possible") for i in range(N_holds)]
    cands_per_frame: list[list[PhysicalHold]] = []
    for t in range(T):
        k = max(1, rng.integers(1, K))
        idx = rng.choice(N_holds, size=int(k), replace=False)
        cands_per_frame.append([holds[i] for i in idx])

    S = K + 2
    hold_logits = rng.normal(0.0, 2.0, size=(T, S)).astype(np.float64)
    limb_conf = rng.uniform(0.1, 1.0, size=T).astype(np.float64)

    cfg = DecodeConfig()

    segs = decode_limb(hold_logits, cands_per_frame, cfg, limb_conf=limb_conf)
                                                                      
    ref_path = _reference_decode_limb(hold_logits, cands_per_frame, cfg, limb_conf=limb_conf)

    ref_boundaries = [
        t for t in range(1, T) if ref_path[t] != ref_path[t - 1]
    ]
    seg_boundaries = sorted({seg.start_frame for seg in segs if seg.start_frame > 0})
                                                                         
    extras = [b for b in seg_boundaries if b not in ref_boundaries]
    assert not extras, f"seed={seed}: spurious segment boundaries {extras}"

def main() -> None:
                                                                          
    _run_case(seed=1, T=60, N_holds=20, K=6)
    _run_case(seed=2, T=120, N_holds=40, K=8)
    _run_case(seed=3, T=200, N_holds=80, K=8)
    print("viterbi equivalence: OK")

if __name__ == "__main__":
    main()
