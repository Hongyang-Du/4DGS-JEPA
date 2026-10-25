"""Experiment 1: path supervision and temporal composition (paper Sec. 5.2, Table 2, Table 7)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import ForecastConfig
from ..losses import hier_distance
from ..rollout import at_time, direct, get_history, rollout
from ..train import to_device


def _fde(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return torch.linalg.norm(a - b, dim=-1).mean(-1).cpu().numpy().astype(np.float64)


@torch.no_grad()
def evaluate(model, ds, cfg: ForecastConfig, device) -> Tuple[Dict, pd.DataFrame]:
    """Per-sample metrics for primary routes (all horizons) and held-out partitions.

    Rows have ``category`` in {primary, heldout_seen_segments, unseen_segments}.
    Composition metrics (CompErr eq. 68, GeoCompErr eq. 69) are NaN for one-segment routes."""
    model.eval()
    horizons = sorted(set(cfg.train_horizons) | set(cfg.unseen_horizons))
    routes: List[Tuple[str, int, List[int]]] = [("primary", H, cfg.primary_partitions[H]) for H in horizons]
    for cat, mapping in [("heldout_seen_segments", cfg.heldout_seen_segment_partitions),
                         ("unseen_segments", cfg.unseen_segment_partitions)]:
        routes += [(cat, H, p) for H, plist in mapping.items() for p in plist]

    rows = []
    for batch in DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False):
        batch = to_device(batch, device)
        t, gids = batch["t"], batch["group_ids"]
        z0 = model.encoder(*get_history(batch, t, cfg.context_len), gids)
        src_mu, src_cov = at_time(batch, "mu", t), at_time(batch, "cov", t)
        for cat, H, parts in routes:
            gt_mu = at_time(batch, "mu", t + H)
            zd, gd = direct(model, z0, src_mu, src_cov, gids, H)
            last = rollout(model, z0, src_mu, src_cov, gids, parts)[-1]
            zr, gr = last["z"], last["geom"]
            nan = np.full(len(t), np.nan)
            nontrivial = len(parts) > 1
            rows.append(pd.DataFrame({
                "category": cat, "horizon": H, "partition": "+".join(map(str, parts)),
                "direct_fde": _fde(gd["mu"], gt_mu), "rollout_fde": _fde(gr["mu"], gt_mu),
                "comp_err": hier_distance(zd, zr, cfg.num_groups, reduce=False).cpu().numpy().astype(np.float64) if nontrivial else nan,
                "geo_comp_err": _fde(gd["mu"], gr["mu"]) if nontrivial else nan,
            }))
    df = pd.concat(rows, ignore_index=True)

    prim = df[df.category == "primary"]
    seen = prim[prim.horizon.isin(cfg.train_horizons)]
    unseen = prim[prim.horizon.isin(cfg.unseen_horizons)]
    held = df[df.category == "heldout_seen_segments"]
    useg = df[df.category == "unseen_segments"]
    summary = {
        "direct_fde": seen.direct_fde.mean(),
        "rollout_fde": seen.rollout_fde.mean(),
        "delta_roll": seen.rollout_fde.mean() - seen.direct_fde.mean(),
        "comp_err": seen.comp_err.mean(),            # NaN rows (H=1) are skipped by mean()
        "geo_comp_err": seen.geo_comp_err.mean(),
        "unseen_fde": unseen.rollout_fde.mean(),
        "unseen_direct_fde": unseen.direct_fde.mean(),
        "heldout_comp_err": held.comp_err.mean(),
        "heldout_geo_comp_err": held.geo_comp_err.mean(),
        "unseen_seg_comp_err": useg.comp_err.mean(),
        "unseen_seg_geo_comp_err": useg.geo_comp_err.mean(),
    }
    return summary, df


@torch.no_grad()
def baselines(ds, cfg: ForecastConfig) -> pd.DataFrame:
    """Causal analytic baselines: static persistence and constant velocity (last two context frames)."""
    rows = []
    horizons = sorted(set(cfg.train_horizons) | set(cfg.unseen_horizons))
    for batch in DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False):
        t = batch["t"]
        cur, prev = at_time(batch, "mu", t), at_time(batch, "mu", t - 1)
        for H in horizons:
            tgt = at_time(batch, "mu", t + H)
            for name, pred in [("Static persistence", cur), ("Constant velocity", cur + H * (cur - prev))]:
                rows.append(pd.DataFrame({"baseline": name, "horizon": H, "fde": _fde(pred, tgt)}))
    return pd.concat(rows, ignore_index=True)


TABLE_COLUMNS = ["direct_fde", "rollout_fde", "delta_roll", "comp_err", "geo_comp_err", "unseen_fde"]
APPENDIX_COLUMNS = ["heldout_comp_err", "heldout_geo_comp_err", "unseen_seg_comp_err", "unseen_seg_geo_comp_err"]
