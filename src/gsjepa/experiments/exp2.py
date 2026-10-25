"""Experiment 2: does latent composition transfer to Gaussian geometry? (paper Sec. 5.3, Tables 3, 8)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..config import ForecastConfig
from ..geometry import affine_spd_distance, se3_distance
from ..losses import dmatch_metric
from ..rollout import at_time, direct, get_history, rollout, true_group_transform
from ..train import to_device


def _np(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy().astype(np.float64)


@torch.no_grad()
def evaluate(model, ds, cfg: ForecastConfig, device) -> Tuple[Dict, pd.DataFrame]:
    """Per-sample metrics on supervised horizons with primary composed routes.

    E_T^target (eq. 70) and E_T^dc (eq. 71) use the controlled SE(3) surrogate;
    cov_err and D_match use the affine-invariant SPD distance. Direct-composed metrics
    are NaN at H=1 where both routes coincide."""
    model.eval()
    rows = []
    for batch in DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False):
        batch = to_device(batch, device)
        t, gids = batch["t"], batch["group_ids"]
        B = len(t)
        z0 = model.encoder(*get_history(batch, t, cfg.context_len), gids)
        src_mu, src_cov = at_time(batch, "mu", t), at_time(batch, "cov", t)
        for H in cfg.train_horizons:
            parts = cfg.primary_partitions[H]
            _, gd = direct(model, z0, src_mu, src_cov, gids, H)
            last = rollout(model, z0, src_mu, src_cov, gids, parts)[-1]
            gc, R_c, d_c = last["geom"], last["R_total"], last["d_total"]
            gt_mu, gt_cov = at_time(batch, "mu", t + H), at_time(batch, "cov", t + H)
            R_true, d_true = true_group_transform(batch, t + H)
            nan = np.full(B, np.nan)
            nontrivial = len(parts) > 1
            rows.append(pd.DataFrame({
                "horizon": H,
                "direct_fde": _np(torch.linalg.norm(gd["mu"] - gt_mu, dim=-1).mean(-1)),
                "rollout_fde": _np(torch.linalg.norm(gc["mu"] - gt_mu, dim=-1).mean(-1)),
                "cov_err": _np(affine_spd_distance(gd["cov"], gt_cov).mean(-1)),
                "transform_target": _np(se3_distance(gd["R"], gd["d"], R_true, d_true).mean(-1)),
                "transform_target_rollout": _np(se3_distance(R_c, d_c, R_true, d_true).mean(-1)),
                "transform_dc": _np(se3_distance(gd["R"], gd["d"], R_c, d_c).mean(-1)) if nontrivial else nan,
                "d_match": _np(dmatch_metric(gd, gc, cfg)) if nontrivial else nan,
            }))
    df = pd.concat(rows, ignore_index=True)
    main = df[df.horizon == cfg.main_horizon]
    summary = {k: main[k].mean() for k in
               ["direct_fde", "rollout_fde", "cov_err", "transform_target", "transform_target_rollout",
                "transform_dc", "d_match"]}
    return summary, df


@torch.no_grad()
def baselines(ds, cfg: ForecastConfig) -> pd.DataFrame:
    """Static persistence / constant velocity FDE and identity-transform E_T^target."""
    rows = []
    for batch in DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False):
        t = batch["t"]
        B, G = len(t), cfg.num_groups
        cur, prev = at_time(batch, "mu", t), at_time(batch, "mu", t - 1)
        I = torch.eye(3).expand(B, G, 3, 3)
        zero = torch.zeros(B, G, 3)
        for H in cfg.train_horizons:
            tgt = at_time(batch, "mu", t + H)
            R_true, d_true = true_group_transform(batch, t + H)
            rows.append(pd.DataFrame({
                "horizon": H,
                "static_fde": _np(torch.linalg.norm(cur - tgt, dim=-1).mean(-1)),
                "constant_velocity_fde": _np(torch.linalg.norm(cur + H * (cur - prev) - tgt, dim=-1).mean(-1)),
                "identity_transform_target": _np(se3_distance(I, zero, R_true, d_true).mean(-1)),
            }))
    return pd.concat(rows, ignore_index=True)


TABLE_COLUMNS = ["direct_fde", "cov_err", "transform_target", "transform_dc", "d_match"]
