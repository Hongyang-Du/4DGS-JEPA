"""Training loop shared by Experiments 1 and 2.

Per optimiser step (App. D.2):
  * one terminal horizon Delta, cycling deterministically through ``train_horizons``;
  * one temporal partition pi of Delta, sampled from ``train_partitions[Delta]``;
  * loss = L_end + lambda_geo L_geo + lambda_rep L_rep
           [+ lambda_path L_path] [+ lambda_comp L_comp]
           [+ lambda_se3 L_gcomp^SE3] [+ lambda_match L_gcomp^match]
Checkpoint selection uses validation target accuracy only: 0.5 * (direct FDE + rollout FDE).
"""
from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ForecastConfig, Variant
from .geometry import se3_distance
from .losses import bidirectional_comp, geometry_loss, hier_distance, rep_regularizer
from .model import JEPA, HierEncoder, ema_update, make_teacher
from .rollout import at_time, direct, get_history, horizon, rollout


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_loss(model: JEPA, teacher: HierEncoder, batch, Delta: int, parts, variant: Variant,
                 cfg: ForecastConfig) -> Dict[str, torch.Tensor]:
    G, L = cfg.num_groups, cfg.context_len
    t, gids = batch["t"], batch["group_ids"]
    B, device = t.shape[0], t.device
    zero = torch.zeros((), device=device)

    def teacher_target(elapsed: int) -> torch.Tensor:
        with torch.no_grad():
            return teacher(*get_history(batch, t + elapsed, L), gids)

    z0 = model.encoder(*get_history(batch, t, L), gids)
    src_mu, src_cov = at_time(batch, "mu", t), at_time(batch, "cov", t)

    # L_end: direct terminal prediction vs EMA-teacher embedding of the future history (eq. 16)
    z_dir_T, g_dir_T = direct(model, z0, src_mu, src_cov, gids, Delta)
    l_end = hier_distance(z_dir_T, teacher_target(Delta), G)
    l_rep = 0.5 * (rep_regularizer(z0, cfg) + rep_regularizer(z_dir_T, cfg))

    needs_path = variant.path or variant.comp or cfg.geo_supervision == "both_routes" \
        or variant.gcomp_se3 or variant.gcomp_match
    steps = rollout(model, z0, src_mu, src_cov, gids, parts) if needs_path else []

    l_path, l_comp = zero, zero
    if variant.path:   # eq. 21: every intermediate composed state vs its teacher target
        l_path = torch.stack([hier_distance(s["z"], teacher_target(s["elapsed"]), G) for s in steps]).mean()

    # Direct predictions to each cumulative endpoint s_j of the partition.
    if variant.comp or cfg.geo_supervision == "both_routes":
        dirs = [direct(model, z0, src_mu, src_cov, gids, s["elapsed"]) for s in steps]
    if variant.comp:   # eq. 23
        l_comp = torch.stack([bidirectional_comp(zd, s["z"], G) for (zd, _), s in zip(dirs, steps)]).mean()

    # L_geo: target-anchored geometry (eq. 39)
    if cfg.geo_supervision == "direct":
        l_geo = geometry_loss(g_dir_T, at_time(batch, "mu", t + Delta), at_time(batch, "cov", t + Delta), cfg)
    elif cfg.geo_supervision == "both_routes":
        terms = []
        for (_, gd), s in zip(dirs, steps):
            tgt_mu, tgt_cov = at_time(batch, "mu", t + s["elapsed"]), at_time(batch, "cov", t + s["elapsed"])
            terms.append(0.5 * (geometry_loss(gd, tgt_mu, tgt_cov, cfg)
                                + geometry_loss(s["geom"], tgt_mu, tgt_cov, cfg)))
        l_geo = torch.stack(terms).mean()
    else:
        raise ValueError(cfg.geo_supervision)

    # L_gcomp: direct vs composed decoded motion at the terminal endpoint (eq. 43)
    l_se3, l_match = zero, zero
    if variant.gcomp_se3:
        l_se3 = se3_distance(g_dir_T["R"], g_dir_T["d"], steps[-1]["R_total"], steps[-1]["d_total"]).mean()
    if variant.gcomp_match:
        l_match = geometry_loss(g_dir_T, steps[-1]["geom"]["mu"], steps[-1]["geom"]["cov"], cfg)

    total = (l_end + cfg.lambda_geo * l_geo + cfg.lambda_rep * l_rep
             + cfg.lambda_path * l_path + cfg.lambda_comp * l_comp
             + cfg.lambda_gcomp_se3 * l_se3 + cfg.lambda_gcomp_match * l_match)
    return {"total": total, "end": l_end, "geo": l_geo, "rep": l_rep, "path": l_path,
            "comp": l_comp, "gcomp_se3": l_se3, "gcomp_match": l_match}


@torch.no_grad()
def validation_score(model: JEPA, ds, cfg: ForecastConfig, device) -> Dict[str, float]:
    """Mean direct and primary-route rollout FDE over the supervised horizons."""
    model.eval()
    direct_vals, rollout_vals = [], []
    for batch in DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False):
        batch = to_device(batch, device)
        t, gids = batch["t"], batch["group_ids"]
        z0 = model.encoder(*get_history(batch, t, cfg.context_len), gids)
        src_mu, src_cov = at_time(batch, "mu", t), at_time(batch, "cov", t)
        for H in cfg.train_horizons:
            gt_mu = at_time(batch, "mu", t + H)
            _, gd = direct(model, z0, src_mu, src_cov, gids, H)
            gr = rollout(model, z0, src_mu, src_cov, gids, cfg.primary_partitions[H])[-1]["geom"]
            direct_vals.append(torch.linalg.norm(gd["mu"] - gt_mu, dim=-1).mean(-1).cpu())
            rollout_vals.append(torch.linalg.norm(gr["mu"] - gt_mu, dim=-1).mean(-1).cpu())
    d, r = torch.cat(direct_vals).mean().item(), torch.cat(rollout_vals).mean().item()
    return {"val_score": 0.5 * (d + r), "val_direct_fde": d, "val_rollout_fde": r}


def train_variant(name: str, variant: Variant, seed: int, cfg: ForecastConfig, train_ds, val_ds,
                  device, log=print) -> Tuple[JEPA, HierEncoder, pd.DataFrame, Dict]:
    """Train one (variant, seed) run from scratch; returns the best-validation checkpoint."""
    seed_all(seed)
    model = JEPA(cfg).to(device)
    teacher = make_teacher(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.min_lr)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    partition_rng = random.Random(cfg.partition_seed_offset + seed)

    history, best, step = [], {"val_score": float("inf")}, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums: Dict[str, float] = {}
        for batch in loader:
            batch = to_device(batch, device)
            Delta = cfg.train_horizons[step % len(cfg.train_horizons)]
            step += 1
            parts = partition_rng.choice(cfg.train_partitions[Delta])
            losses = compute_loss(model, teacher, batch, Delta, parts, variant, cfg)

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema_update(teacher, model.encoder, cfg.ema_tau)
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())

        val = validation_score(model, val_ds, cfg, device)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
               **{f"train_{k}": v / len(loader) for k, v in sums.items()}, **val}
        history.append(row)
        if val["val_score"] < best["val_score"]:
            best = {**val, "epoch": epoch,
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "teacher": {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}}
        log(f"[{name} | seed {seed}] epoch {epoch:02d}/{cfg.epochs} train={row['train_total']:.4f} "
            f"val={val['val_score']:.5f} (direct={val['val_direct_fde']:.5f}, "
            f"rollout={val['val_rollout_fde']:.5f}) best_epoch={best['epoch']}")
        sched.step()

    model.load_state_dict(best["model"])
    teacher.load_state_dict(best["teacher"])
    info = {"best_epoch": best["epoch"], "best_val_score": best["val_score"],
            "best_val_direct_fde": best["val_direct_fde"], "best_val_rollout_fde": best["val_rollout_fde"]}
    return model.eval(), teacher.eval(), pd.DataFrame(history), info
