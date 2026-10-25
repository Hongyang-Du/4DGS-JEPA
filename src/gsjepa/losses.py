"""Loss terms of the 4DGS-JEPA objective (eq. 49), identity-aligned controlled versions."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .config import ForecastConfig
from .geometry import affine_spd_distance, cov_to_log_cholesky


# ---------------------------------------------------------------- latent distance (eq. 15)
def hier_distance(a: torch.Tensor, b: torch.Tensor, num_groups: int, reduce: bool = True) -> torch.Tensor:
    """D_hier with identity correspondence: cosine (scene) + cosine (groups) + SmoothL1 (Gaussians).

    With ``reduce=False`` returns one value per sample [B] (used for the CompErr metric)."""
    G = num_groups
    s = 1 - F.cosine_similarity(a[:, 0], b[:, 0], dim=-1)
    o = (1 - F.cosine_similarity(a[:, 1:1 + G], b[:, 1:1 + G], dim=-1))
    if reduce:
        return s.mean() + o.mean() + F.smooth_l1_loss(a[:, 1 + G:], b[:, 1 + G:])
    g = F.smooth_l1_loss(a[:, 1 + G:], b[:, 1 + G:], reduction="none").mean(dim=(-1, -2))
    return s + o.mean(dim=-1) + g


def bidirectional_comp(z_dir: torch.Tensor, z_comp: torch.Tensor, num_groups: int) -> torch.Tensor:
    """One term of L_comp (eq. 23) with u_j = v_j = 1/2: each route is a stop-grad target for the other."""
    return 0.5 * (hier_distance(z_dir, z_comp.detach(), num_groups)
                  + hier_distance(z_comp, z_dir.detach(), num_groups))


# ---------------------------------------------------------------- geometry
def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Element-wise Huber rho_delta on non-negative distances."""
    d = torch.as_tensor(delta, dtype=x.dtype, device=x.device)
    return torch.where(x <= d, 0.5 * x.pow(2), d * (x - 0.5 * d))


def geometry_loss(pred: Dict[str, torch.Tensor], tgt_mu, tgt_cov, cfg: ForecastConfig) -> torch.Tensor:
    """Controlled l_geo (eq. 37): Huber on Euclidean center error + w * SmoothL1 in log-Cholesky coords.

    Also used as the training-time D_match between two predicted states."""
    center = huber(torch.linalg.norm(pred["mu"] - tgt_mu, dim=-1), cfg.center_huber_delta).mean()
    cov = F.smooth_l1_loss(cov_to_log_cholesky(pred["cov"]), cov_to_log_cholesky(tgt_cov))
    return center + cfg.cov_loss_weight * cov


def dmatch_metric(g1: Dict[str, torch.Tensor], g2: Dict[str, torch.Tensor], cfg: ForecastConfig) -> torch.Tensor:
    """Evaluation D_match (eq. 42) per sample [B]: Huber center + w * affine-invariant SPD distance."""
    center = huber(torch.linalg.norm(g1["mu"] - g2["mu"], dim=-1), cfg.center_huber_delta).mean(dim=-1)
    cov = affine_spd_distance(g1["cov"], g2["cov"]).mean(dim=-1)
    return center + cfg.cov_loss_weight * cov


# ---------------------------------------------------------------- representation regulariser (eq. 48, App. D.1.2)
def _vicreg_level(x: torch.Tensor, cfg: ForecastConfig) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1])
    std = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-4)
    lvar = F.relu(cfg.var_floor - std).pow(2).mean()
    xc = x - x.mean(0, keepdim=True)
    C = (xc.T @ xc) / max(1, x.shape[0] - 1)
    off = C - torch.diag(torch.diagonal(C))
    return lvar + cfg.lambda_cov_rep * off.pow(2).sum() / x.shape[1]


def rep_regularizer(z: torch.Tensor, cfg: ForecastConfig) -> torch.Tensor:
    """VICReg-style variance + covariance penalty, averaged over scene / group / Gaussian levels."""
    G = cfg.num_groups
    return (_vicreg_level(z[:, :1], cfg) + _vicreg_level(z[:, 1:1 + G], cfg)
            + _vicreg_level(z[:, 1 + G:], cfg)) / 3.0
