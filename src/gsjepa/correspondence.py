"""Gaussian correspondence: index matching, count-aware OT, and hybrid persistent-ID + residual OT
(paper Sec. 4, App. D.4). Pure NumPy, no learning.

A Gaussian *state* is a dict of arrays: mu [N,3], logscale [N,3], sem [N,S], group [N], slot_id [N]
(-1 = no valid persistent identity). Source states also carry ``active`` [N]; synthetic targets carry
the hidden ground-truth ``lineage`` (list of sets of source slot ids) and ``birth`` [N].

Every plan is an (ns+1) x (nt+1) matrix whose last row / column are the source / target dustbins.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import CorrespondenceConfig

State = Dict[str, object]


def unit(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- synthetic source / target states
def make_source(seed: int, cfg: CorrespondenceConfig) -> State:
    """Active primitives + reserved inactive slots anchored near their motion groups."""
    rng = np.random.default_rng(seed)
    N = cfg.active_primitives + cfg.reserve_slots
    gids = np.repeat(np.arange(cfg.num_groups), math.ceil(N / cfg.num_groups))[:N]
    centers = rng.uniform(-1, 1, (cfg.num_groups, 3))
    mu = np.vstack([centers[g] + rng.normal(0, 0.18, 3) for g in gids]).astype(np.float64)
    logscale = rng.uniform(np.log(0.04), np.log(0.12), (N, 3)).astype(np.float64)
    sem = unit(rng.normal(size=(N, cfg.semantic_dim))).astype(np.float64)
    active = np.zeros(N, dtype=bool)
    active[:cfg.active_primitives] = True
    for i in range(cfg.active_primitives, N):
        mu[i] = centers[gids[i]] + rng.normal(0, 0.08, 3)
    return {"mu": mu, "logscale": logscale, "sem": sem, "group": gids.astype(int),
            "active": active, "slot_id": np.arange(N, dtype=int)}


def _item(src: State, i: int) -> Dict:
    return {"mu": src["mu"][i].copy(), "logscale": src["logscale"][i].copy(), "sem": src["sem"][i].copy(),
            "group": int(src["group"][i]), "slot_id": int(src["slot_id"][i]),
            "lineage": {int(src["slot_id"][i])}, "birth": not bool(src["active"][i])}


def _clone(x: Dict) -> Dict:
    return {k: v.copy() if isinstance(v, np.ndarray) else set(v) if isinstance(v, set) else v
            for k, v in x.items()}


def perturb_source(src: State, kind: str, severity: float, rng: np.random.Generator,
                   cfg: CorrespondenceConfig) -> State:
    """Apply small motion noise, then a topology event: permutation | split | merge | death | birth | mixed.

    Split children / merged primitives keep ground-truth lineage but lose their persistent id."""
    active_ids = np.flatnonzero(src["active"]).tolist()
    reserve_ids = np.flatnonzero(~src["active"]).tolist()
    items = [_item(src, i) for i in active_ids]
    for x in items:
        x["mu"] += rng.normal(0, cfg.position_noise_std, 3)
        x["logscale"] += rng.normal(0, cfg.logscale_noise_std, 3)

    n = max(1, int(round(severity * len(active_ids))))

    if kind in ("split", "mixed"):
        candidates = rng.choice(len(items), size=min(n, len(items)), replace=False).tolist()
        children = []
        for idx in sorted(candidates, reverse=True):
            parent = items.pop(idx)
            for sign in (-1, 1):
                child = _clone(parent)
                child["mu"] = parent["mu"] + sign * rng.normal(0.035, 0.008, 3)
                child["slot_id"], child["birth"] = -1, False
                children.append(child)
        items.extend(children)

    if kind in ("merge", "mixed") and len(items) >= 2:
        # Repeatedly merge the closest same-group pair.
        for _ in range(min(n, max(1, len(items) // 4))):
            pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))
                     if items[i]["group"] == items[j]["group"]]
            if not pairs:
                break
            i, j = min(pairs, key=lambda ij: np.linalg.norm(items[ij[0]]["mu"] - items[ij[1]]["mu"]))
            a, b = items[i], items[j]
            merged = {"mu": 0.5 * (a["mu"] + b["mu"]), "logscale": 0.5 * (a["logscale"] + b["logscale"]),
                      "sem": unit((a["sem"] + b["sem"])[None])[0], "group": a["group"], "slot_id": -1,
                      "lineage": set(a["lineage"]) | set(b["lineage"]), "birth": False}
            for idx in sorted((i, j), reverse=True):
                items.pop(idx)
            items.append(merged)

    if kind in ("death", "mixed") and items:
        k = min(n, len(items) // 3 if len(items) >= 3 else 1)
        if k > 0:
            remove = set(rng.choice(len(items), size=k, replace=False).tolist())
            items = [x for idx, x in enumerate(items) if idx not in remove]

    if kind in ("birth", "mixed") and reserve_ids:
        for rid in rng.choice(reserve_ids, size=min(n, len(reserve_ids)), replace=False):
            x = _item(src, int(rid))
            x["birth"] = True
            x["mu"] += rng.normal(0, 0.04, 3)
            x["logscale"] += rng.normal(0, cfg.logscale_noise_std, 3)
            items.append(x)

    if cfg.semantic_noise_std > 0:   # semantics are informative but not a perfect fingerprint
        for x in items:
            x["sem"] = unit((x["sem"] + rng.normal(0, cfg.semantic_noise_std, cfg.semantic_dim))[None])[0]

    if kind in ("permutation", "mixed"):   # destroy array-order identity
        rng.shuffle(items)

    tgt = {k: [x[k] for x in items] for k in ["mu", "logscale", "sem", "group", "slot_id", "lineage", "birth"]}
    for k in ["mu", "logscale", "sem"]:
        tgt[k] = np.asarray(tgt[k], dtype=np.float64)
    tgt["group"] = np.asarray(tgt["group"], dtype=int)
    tgt["slot_id"] = np.asarray(tgt["slot_id"], dtype=int)
    tgt["birth"] = np.asarray(tgt["birth"], dtype=bool)
    return tgt


def hide_persistent_ids(tgt: State, rate: float, rng: np.random.Generator) -> State:
    """Hide a fraction of valid target ids (geometry and lineage are unchanged)."""
    out = {k: v.copy() if isinstance(v, np.ndarray) else [set(x) for x in v] if k == "lineage" else v
           for k, v in tgt.items()}
    valid = np.flatnonzero(out["slot_id"] >= 0)
    k = int(round(float(rate) * len(valid)))
    if k > 0:
        out["slot_id"][rng.choice(valid, size=min(k, len(valid)), replace=False)] = -1
    return out


# ---------------------------------------------------------------- matching
def pair_cost(src: State, tgt: State) -> np.ndarray:
    """C_ij = 2|mu_i-mu_j|^2 + 0.25|s_i-s_j|^2 + 0.8 d_cos(f_i,f_j) + 0.8 [m(i) != m(j)]  (App. D.4.1)."""
    dm = ((src["mu"][:, None] - tgt["mu"][None]) ** 2).sum(-1)
    ds = ((src["logscale"][:, None] - tgt["logscale"][None]) ** 2).sum(-1)
    cos = 1 - np.clip(src["sem"] @ tgt["sem"].T, -1, 1)
    gp = (src["group"][:, None] != tgt["group"][None]).astype(float)
    return 2.0 * dm + 0.25 * ds + 0.8 * cos + 0.8 * gp


def sinkhorn(C: np.ndarray, p: np.ndarray, q: np.ndarray, eps: float, iters: int, tol: float) -> np.ndarray:
    """Balanced entropic OT (Cuturi 2013) with strictly positive marginals of equal mass."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    if not np.isclose(p.sum(), q.sum()):
        raise ValueError(f"Unequal OT mass: p={p.sum()} q={q.sum()}")
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("All Sinkhorn marginals must be strictly positive.")
    K = np.maximum(np.exp(-np.asarray(C, np.float64) / eps), 1e-300)
    u, v = np.ones_like(p), np.ones_like(q)
    for it in range(iters):
        u = p / (K @ v + 1e-300)
        v = q / (K.T @ u + 1e-300)
        if (it + 1) % 25 == 0 or it == iters - 1:
            P = (u[:, None] * K) * v[None, :]
            if max(np.abs(P.sum(1) - p).max(), np.abs(P.sum(0) - q).max()) < tol:
                break
    return (u[:, None] * K) * v[None, :]


def ot_plan(src: State, tgt: State, cfg: CorrespondenceConfig, src_mask=None, tgt_mask=None) -> np.ndarray:
    """Count-aware dustbin OT over the (masked) residual sets.

    Each real primitive has unit mass; dustbin capacities equal the opposite residual count:
    p = [1..1, nt], q = [1..1, ns]."""
    NS, NT = len(src["mu"]), len(tgt["mu"])
    si = np.arange(NS) if src_mask is None else np.flatnonzero(src_mask)
    tj = np.arange(NT) if tgt_mask is None else np.flatnonzero(tgt_mask)
    ns, nt = len(si), len(tj)
    P = np.zeros((NS + 1, NT + 1))
    if ns == 0 and nt == 0:
        return P
    if ns == 0:
        P[-1, tj] = 1.0
        return P
    if nt == 0:
        P[si, -1] = 1.0
        return P

    sub = lambda d, idx, n: {k: v[idx] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in d.items()}
    Ca = np.full((ns + 1, nt + 1), cfg.dustbin_cost)
    Ca[:ns, :nt] = pair_cost(sub(src, si, NS), sub(tgt, tj, NT))
    Ca[-1, -1] = 0.0
    Pr = sinkhorn(Ca, np.r_[np.ones(ns), float(nt)], np.r_[np.ones(nt), float(ns)],
                  cfg.sinkhorn_eps, cfg.sinkhorn_iters, cfg.sinkhorn_tol)
    P[np.ix_(np.r_[si, NS], np.r_[tj, NT])] = Pr
    return P


def index_plan(src: State, tgt: State) -> np.ndarray:
    """Naive baseline: array position i in source == position i in target."""
    ns, nt = len(src["mu"]), len(tgt["mu"])
    P = np.zeros((ns + 1, nt + 1))
    k = min(ns, nt)
    P[np.arange(k), np.arange(k)] = 1.0
    P[k:ns, -1] = 1.0
    P[-1, k:nt] = 1.0
    return P


def hybrid_plan(src: State, tgt: State, cfg: CorrespondenceConfig) -> np.ndarray:
    """Clamp valid unique persistent ids with unit mass (eq. 56), then OT on residual mass (eqs. 57-61)."""
    ns, nt = len(src["mu"]), len(tgt["mu"])
    P = np.zeros((ns + 1, nt + 1))
    used_s, used_t = np.zeros(ns, bool), np.zeros(nt, bool)
    slot_to_src = {int(s): i for i, s in enumerate(src["slot_id"])}
    ids, counts = np.unique(tgt["slot_id"][tgt["slot_id"] >= 0], return_counts=True)
    unique_ids = set(ids[counts == 1].tolist())
    for j, sid in enumerate(tgt["slot_id"]):
        sid = int(sid)
        if sid in unique_ids and sid in slot_to_src and not used_s[slot_to_src[sid]]:
            i = slot_to_src[sid]
            P[i, j], used_s[i], used_t[j] = 1.0, True, True
    return P + ot_plan(src, tgt, cfg, ~used_s, ~used_t)


# ---------------------------------------------------------------- metrics (App. D.4.2)
def lineage_matrix(src: State, tgt: State) -> np.ndarray:
    return np.array([[float(int(sid) in lin) for lin in tgt["lineage"]] for sid in src["slot_id"]])


def binary_prf(gt, pred):
    """Precision, recall, F1, #positives; NaN metrics when the event is absent from ground truth."""
    gt, pred = np.asarray(gt, bool), np.asarray(pred, bool)
    n_pos = int(gt.sum())
    if n_pos == 0:
        return np.nan, np.nan, np.nan, 0
    tp, fp, fn = int((gt & pred).sum()), int((~gt & pred).sum()), int((gt & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, n_pos


def event_metrics(src: State, tgt: State, P: np.ndarray) -> Dict[str, float]:
    ns, nt = len(src["mu"]), len(tgt["mu"])
    # Death: an active source slot absent from every target lineage; predicted when
    # its target-dustbin mass >= its mass on real targets.
    all_lineage = set().union(*tgt["lineage"]) if len(tgt["lineage"]) else set()
    gt_death = np.array([bool(src["active"][i]) and int(src["slot_id"][i]) not in all_lineage for i in range(ns)])
    pred_death = np.array([P[i, -1] >= P[i, :nt].sum() for i in range(ns)]) & src["active"]
    dp, dr, df1, nd = binary_prf(gt_death, pred_death)

    # Birth: target descends from a reserved slot; predicted when mass from reserved slots +
    # source dustbin >= mass from active sources.
    inactive_ids = set(src["slot_id"][~src["active"]].tolist())
    gt_birth = np.array([len(lin & inactive_ids) > 0 for lin in tgt["lineage"]])
    act, res = np.flatnonzero(src["active"]), np.flatnonzero(~src["active"])
    pred_birth = np.array([(P[res, j].sum() if len(res) else 0.0) + P[-1, j]
                           >= (P[act, j].sum() if len(act) else 0.0) for j in range(nt)], dtype=bool)
    bp, br, bf1, nb = binary_prf(gt_birth, pred_birth)

    f1s = np.array([bf1, df1], dtype=np.float64)
    return {"birth_precision": bp, "birth_recall": br, "birth_f1": bf1, "birth_gt_count": nb,
            "death_precision": dp, "death_recall": dr, "death_f1": df1, "death_gt_count": nd,
            "event_f1_macro": float(np.nanmean(f1s)) if np.isfinite(f1s).any() else np.nan}


def plan_metrics(src: State, tgt: State, P: np.ndarray) -> Dict[str, float]:
    """LineageMass (eq. 73), correspondence-induced geometry discrepancy D_corr, and event P/R/F1."""
    ns, nt = len(src["mu"]), len(tgt["mu"])
    real = P[:ns, :nt]
    dist = np.linalg.norm(src["mu"][:, None] - tgt["mu"][None], axis=-1)
    return {"lineage_mass": (lineage_matrix(src, tgt) * real).sum() / (real.sum() + 1e-12),
            "geometry_discrepancy": (real * dist).sum() / (real.sum() + 1e-12),
            **event_metrics(src, tgt, P)}
