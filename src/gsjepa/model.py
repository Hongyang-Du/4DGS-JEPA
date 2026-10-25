"""Small 4DGS-JEPA used in the controlled experiments (paper Sec. 3.2-3.8, App. D.1.1).

Latent state layout (eq. 9), a single token tensor z of shape [B, 1 + G + N, D]:
    z[:, 0]          scene token
    z[:, 1:1+G]      motion-group tokens
    z[:, 1+G:]       Gaussian tokens
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ForecastConfig
from .geometry import axis_angle_to_matrix, cov_to_log_cholesky


# ---------------------------------------------------------------- causal input features
def _normalize_cov_coord(coord: torch.Tensor) -> torch.Tensor:
    """NormCov (App. D.1.1): fixed a-priori normalisation of log-Cholesky coordinates."""
    diag = (coord[..., :3] - math.log(0.07)) / 0.5
    off = coord[..., 3:] / 0.05
    return torch.cat([diag, off], dim=-1)


def group_mean(x: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Average per-Gaussian features [B,N,C] within each motion group -> [B,G,C]."""
    out = []
    for g in range(num_groups):
        mask = (group_ids == g).float().unsqueeze(-1)
        out.append((x * mask).sum(1) / mask.sum(1).clamp_min(1.0))
    return torch.stack(out, dim=1)


def causal_gaussian_features(mu: torch.Tensor, cov: torch.Tensor, group_ids: torch.Tensor,
                             cfg: ForecastConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-Gaussian input features x_{i,t} from a causal history only.

    mu [B,L,N,3] and cov [B,L,N,3,3] end at the context endpoint t. Returns
      geom   [B,N,3+3+6+1+G]   current position, group-relative position, NormCov, existence, one-hot
      motion [B,N,3(L-1)+12]   relative history, velocity, acceleration, covariance change
    """
    B, L, N, _ = mu.shape
    assert L == cfg.context_len and L >= 3
    cov_coord = cov_to_log_cholesky(cov)
    cur_mu, cur_cov = mu[:, -1], cov_coord[:, -1]

    centers = group_mean(cur_mu, group_ids, cfg.num_groups)
    bidx = torch.arange(B, device=mu.device)[:, None]
    local_pos = cur_mu - centers[bidx, group_ids]
    onehot = F.one_hot(group_ids, num_classes=cfg.num_groups).float()
    exist = torch.ones(B, N, 1, device=mu.device, dtype=mu.dtype)
    geom = torch.cat([cur_mu / cfg.position_scale,
                      local_pos / cfg.relative_position_scale,
                      _normalize_cov_coord(cur_cov), exist, onehot], dim=-1)

    rel_hist = (mu[:, :-1] - cur_mu[:, None]) / cfg.relative_position_scale
    rel_hist = rel_hist.permute(0, 2, 1, 3).reshape(B, N, -1)
    velocity = (mu[:, -1] - mu[:, -2]) / cfg.velocity_scale
    prev_velocity = (mu[:, -2] - mu[:, -3]) / cfg.velocity_scale
    acceleration = (velocity - prev_velocity) * (cfg.velocity_scale / cfg.acceleration_scale)
    delta_cov = (cov_coord[:, -1] - cov_coord[:, -2]) / cfg.cov_delta_scale
    motion = torch.cat([rel_hist, velocity, acceleration, delta_cov], dim=-1)
    return geom, motion


def _transformer(cfg: ForecastConfig) -> nn.TransformerEncoder:
    D = cfg.token_dim
    layer = nn.TransformerEncoderLayer(D, cfg.transformer_heads, dim_feedforward=2 * D,
                                       batch_first=True, norm_first=True,
                                       dropout=cfg.transformer_dropout)
    return nn.TransformerEncoder(layer, cfg.transformer_layers, enable_nested_tensor=False)


# ---------------------------------------------------------------- modules
class HierEncoder(nn.Module):
    """E_theta: causal Gaussian history -> hierarchical latent state Z_t."""

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        D = cfg.token_dim
        self.cfg = cfg
        geom_dim = 3 + 3 + 6 + 1 + cfg.num_groups
        motion_dim = 3 * (cfg.context_len - 1) + 3 + 3 + 6
        self.geom_mlp = nn.Sequential(nn.Linear(geom_dim, D), nn.GELU(), nn.Linear(D, D))
        self.motion_mlp = nn.Sequential(nn.Linear(motion_dim, D), nn.GELU(), nn.Linear(D, D))
        self.fuse = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU())
        self.contextualizer = _transformer(cfg)
        self.type_embed = nn.Parameter(torch.randn(3, D) * 0.02)
        self.norm = nn.LayerNorm(D)

    def forward(self, mu_hist, cov_hist, group_ids) -> torch.Tensor:
        geom, motion = causal_gaussian_features(mu_hist, cov_hist, group_ids, self.cfg)
        gtok = self.fuse(self.geom_mlp(geom) + self.motion_mlp(motion))   # Gaussian tokens
        otok = group_mean(gtok, group_ids, self.cfg.num_groups)           # group tokens
        stok = otok.mean(dim=1, keepdim=True)                              # scene token
        tokens = torch.cat([stok + self.type_embed[0],
                            otok + self.type_embed[1],
                            gtok + self.type_embed[2]], dim=1)
        return self.norm(self.contextualizer(tokens))


class HorizonEmbedding(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float().view(-1, 1)
        return self.mlp(torch.cat([h / 8.0, torch.sin(h), torch.cos(h), torch.log1p(h)], dim=-1))


class Transition(nn.Module):
    """Phi_phi(Z, Delta): horizon-conditioned residual transformer, closed in latent space (eq. 12)."""

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        D = cfg.token_dim
        self.h = HorizonEmbedding(D)
        self.net = _transformer(cfg)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return z + self.out(self.net(z + self.h(h)[:, None, :]))


class GeometryDecoder(nn.Module):
    """Selective geometry decoder Q_psi (Sec. 3.8, restricted controlled version of App. D.1.1).

    Predicts per group an axis-angle rotation + translation, and per Gaussian a residual
    displacement r and diagonal log-scale deformation s, then applies eqs. (31)-(32):
        mu_b    = R mu_a + d + r
        Sigma_b = R U Sigma_a U^T R^T,   U = diag(exp(s))
    """

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        D, H = cfg.token_dim, cfg.horizon_dim
        self.cfg = cfg
        self.h = HorizonEmbedding(H)
        self.group_head = nn.Sequential(nn.Linear(2 * D + H, 2 * D), nn.GELU(), nn.Linear(2 * D, 6))
        self.gauss_head = nn.Sequential(nn.Linear(2 * D + H, 2 * D), nn.GELU(), nn.Linear(2 * D, 6))

    def forward(self, z0, z1, source_mu, source_cov, group_ids, h) -> Dict[str, torch.Tensor]:
        B, N, _ = source_mu.shape
        G = self.cfg.num_groups
        he = self.h(h)
        hscale = h.float().view(B, 1, 1)   # output bounds grow with elapsed time

        go = self.group_head(torch.cat([z0[:, 1:1 + G], z1[:, 1:1 + G],
                                        he[:, None].expand(B, G, -1)], dim=-1))
        R = axis_angle_to_matrix((0.10 * hscale) * torch.tanh(go[..., :3]))
        d = (0.10 * hscale) * torch.tanh(go[..., 3:])

        po = self.gauss_head(torch.cat([z0[:, 1 + G:1 + G + N], z1[:, 1 + G:1 + G + N],
                                        he[:, None].expand(B, N, -1)], dim=-1))
        r = (0.025 * hscale) * torch.tanh(po[..., :3])
        log_s = (0.04 * hscale) * torch.tanh(po[..., 3:])

        bidx = torch.arange(B, device=source_mu.device)[:, None]
        R_i, d_i = R[bidx, group_ids], d[bidx, group_ids]
        pred_mu = (R_i @ source_mu.unsqueeze(-1)).squeeze(-1) + d_i + r
        U = torch.diag_embed(torch.exp(log_s))
        pred_cov = R_i @ U @ source_cov @ U.transpose(-1, -2) @ R_i.transpose(-1, -2)
        if self.cfg.decoder_cov_stabilize:
            pred_cov = 0.5 * (pred_cov + pred_cov.transpose(-1, -2))
            pred_cov = pred_cov + 1e-6 * torch.eye(3, dtype=pred_cov.dtype, device=pred_cov.device)
        return {"mu": pred_mu, "cov": pred_cov, "R": R, "d": d, "r": r, "log_s": log_s}


class JEPA(nn.Module):
    """Student: encoder + transition + decoder. The EMA teacher is a separate HierEncoder."""

    def __init__(self, cfg: ForecastConfig):
        super().__init__()
        self.encoder = HierEncoder(cfg)
        self.transition = Transition(cfg)
        self.decoder = GeometryDecoder(cfg)


def make_teacher(model: JEPA, cfg: ForecastConfig) -> HierEncoder:
    teacher = HierEncoder(cfg).to(next(model.parameters()).device)
    teacher.load_state_dict(model.encoder.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.eval()


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float) -> None:
    """theta_bar <- tau * theta_bar + (1 - tau) * theta   (eq. 10)."""
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=1 - tau)
