"""Controlled dynamic Gaussian worlds (paper Sec. 5.1, App. D.1, Table 5).

Each scene has ``num_groups`` motion groups of anisotropic Gaussians. Every Gaussian follows

    mu_{i,t}    = R_{m(i),t} mu_{i,0} + d_{m(i),t} + r_{i,t}                (eq. 64)
    Sigma_{i,t} = R_{m(i),t} U_{i,t} Sigma_{i,0} U_{i,t}^T R_{m(i),t}^T     (eq. 65)

where (R, d) is a smooth accelerated/oscillating rigid group motion and (r, U) is a small
Gaussian-specific non-rigid residual. Identities are persistent (no birth/death).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ForecastConfig
from .geometry import np_axis_angle_to_matrix, np_make_spd


def generate_scene(seed: int, cfg: ForecastConfig) -> Dict[str, np.ndarray]:
    """Generate one trajectory. Returns mu [T,N,3], cov [T,N,3,3], group_ids [N],
    and the ground-truth absolute group poses group_R [T,G,3,3], group_d [T,G,3]."""
    rng = np.random.default_rng(seed)
    G, P, T = cfg.num_groups, cfg.gaussians_per_group, cfg.seq_len
    N = G * P
    group_ids = np.repeat(np.arange(G), P).astype(np.int64)

    # Per-Gaussian canonical geometry and non-rigid residual parameters.
    group_centers = rng.uniform(-1.0, 1.0, size=(G, 3)).astype(np.float32)
    base_mu = np.zeros((N, 3), np.float32)
    base_cov = np.zeros((N, 3, 3), np.float32)
    local_phase = rng.uniform(0, 2 * np.pi, size=(N, 1)).astype(np.float32)
    local_dir = rng.normal(size=(N, 3)).astype(np.float32)
    local_dir /= np.linalg.norm(local_dir, axis=1, keepdims=True) + 1e-8
    local_amp = rng.uniform(0.0, 0.035, size=(N, 1)).astype(np.float32)
    local_freq = rng.uniform(0.10, 0.35, size=(N, 1)).astype(np.float32)
    scale_phase = rng.uniform(0, 2 * np.pi, size=(N, 3)).astype(np.float32)
    scale_amp = rng.uniform(0.0, 0.06, size=(N, 3)).astype(np.float32)
    for i in range(N):
        base_mu[i] = group_centers[group_ids[i]] + rng.normal(0, 0.16, size=3)
        base_cov[i] = np_make_spd(rng.uniform(np.log(0.04), np.log(0.10), size=3))

    # Per-group rigid motion parameters.
    axes = rng.normal(size=(G, 3)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-8
    omega = rng.uniform(-0.045, 0.045, size=(G, 1)).astype(np.float32)
    ang_acc = rng.uniform(-0.002, 0.002, size=(G, 1)).astype(np.float32)
    velocity = rng.uniform(-0.035, 0.035, size=(G, 3)).astype(np.float32)
    accel = rng.uniform(-0.0015, 0.0015, size=(G, 3)).astype(np.float32)
    trans_amp = rng.uniform(0.0, 0.08, size=(G, 3)).astype(np.float32)
    trans_freq = rng.uniform(0.08, 0.18, size=(G, 1)).astype(np.float32)
    trans_phase = rng.uniform(0, 2 * np.pi, size=(G, 3)).astype(np.float32)

    # Half of the scenes contain one static group.
    if seed % 2 == 0:
        omega[0] = ang_acc[0] = velocity[0] = accel[0] = trans_amp[0] = 0

    mu = np.zeros((T, N, 3), np.float32)
    cov = np.zeros((T, N, 3, 3), np.float32)
    group_R = np.zeros((T, G, 3, 3), np.float32)
    group_d = np.zeros((T, G, 3), np.float32)
    for t in range(T):
        tt = float(t)
        for g in range(G):
            angle = omega[g, 0] * tt + 0.5 * ang_acc[g, 0] * tt * tt
            group_R[t, g] = np_axis_angle_to_matrix(axes[g] * angle)
            group_d[t, g] = (velocity[g] * tt + 0.5 * accel[g] * tt * tt
                             + trans_amp[g] * np.sin(trans_freq[g, 0] * tt + trans_phase[g]))
        for i in range(N):
            g = group_ids[i]
            r = local_amp[i] * np.sin(local_freq[i] * tt + local_phase[i]) * local_dir[i]
            U = np.diag(np.exp(scale_amp[i] * np.sin(0.12 * tt + scale_phase[i])).astype(np.float32))
            Rg = group_R[t, g]
            mu[t, i] = Rg @ base_mu[i] + group_d[t, g] + r
            cov[t, i] = Rg @ U @ base_cov[i] @ U.T @ Rg.T

    return {"mu": mu, "cov": cov, "group_ids": group_ids, "group_R": group_R, "group_d": group_d}


class GaussianWorldDataset(Dataset):
    """One sample = (full trajectory, context endpoint t). The model only ever slices
    the causal history [t-L+1, t]; later frames are used as targets."""

    def __init__(self, scene_seeds: List[int], cfg: ForecastConfig):
        self.scenes = [generate_scene(s, cfg) for s in scene_seeds]
        max_h = max(max(cfg.train_horizons), max(cfg.unseen_horizons, default=0))
        self.samples = [(si, t) for si in range(len(self.scenes))
                        for t in range(cfg.context_len - 1, cfg.seq_len - max_h)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si, t = self.samples[idx]
        sc = self.scenes[si]
        out = {k: torch.from_numpy(sc[k]) for k in ("mu", "cov", "group_ids", "group_R", "group_d")}
        out["t"] = torch.tensor(t, dtype=torch.long)
        return out


def split_seeds(cfg: ForecastConfig) -> Tuple[List[int], List[int], List[int]]:
    """Disjoint scene seeds for train / val / test (no scene reuse or continuation)."""
    return (list(range(10_000, 10_000 + cfg.train_scenes)),
            list(range(20_000, 20_000 + cfg.val_scenes)),
            list(range(30_000, 30_000 + cfg.test_scenes)))


def build_datasets(cfg: ForecastConfig):
    return tuple(GaussianWorldDataset(s, cfg) for s in split_seeds(cfg))
