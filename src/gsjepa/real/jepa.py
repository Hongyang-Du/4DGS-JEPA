"""4DGS-JEPA scaled to real Deformable-GS trajectories (tens/hundreds of thousands of Gaussians).

Same latent layout and objectives as the controlled model (gsjepa.model), with one change for scale:
self-attention runs only over the 1 + G scene/group tokens; Gaussian tokens are updated by
per-Gaussian MLPs conditioned on their group token. Time is measured in units of the median
prefix frame spacing dt, so a horizon h is a (float) number of frames.

Latent state z = (S, P): S [B, 1+G, D] scene + group tokens, P [B, n, D] Gaussian tokens.
Gaussian state: xyz [.., 3], quaternion rot [.., 4] (w, x, y, z; unnormalised, as rendered), scale [.., 3].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RealConfig:
    num_groups: int = 32
    context_len: int = 4
    context_stride: int = 1          # context frames are every context_stride-th prefix frame
    token_dim: int = 128
    layers: int = 4
    heads: int = 4
    max_horizon: int = 40            # largest training horizon (in prefix frames)
    targets_per_context: int = 4     # endpoint + path targets sampled per context
    batch_contexts: int = 16
    gaussians_per_step: int = 8192
    steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    ema_tau: float = 0.996
    w_end: float = 1.0
    w_comp: float = 0.5
    w_geo: float = 1.0
    w_geo_rollout: float = 0.5
    w_var: float = 0.1
    rollout_segment: float = 8.0     # segment length for composed rollouts at test time
    cv_k: int = 0                    # >0: decoder predicts a correction on top of constant velocity over k frames
    schedule: str = "onecycle"       # "onecycle" (first run) or "cosine" (AdamW, linear warmup + cosine decay)
    warmup_frac: float = 0.05
    min_lr_ratio: float = 0.01
    seed: int = 0
    log_every: int = 500
    extra: Dict = field(default_factory=dict)


# ------------------------------------------------------------------ quaternion helpers
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def axis_angle_to_quat(w: torch.Tensor) -> torch.Tensor:
    theta = w.norm(dim=-1, keepdim=True)
    axis = w / theta.clamp_min(1e-8)
    return torch.cat([torch.cos(theta / 2), torch.sin(theta / 2) * axis], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    conj = q * q.new_tensor([1.0, -1.0, -1.0, -1.0])
    return quat_mul(quat_mul(q, qv), conj)[..., 1:]


def scatter_mean(x: torch.Tensor, gid: torch.Tensor, G: int) -> torch.Tensor:
    """x [B, n, C], gid [n] -> [B, G, C]."""
    B, n, C = x.shape
    out = x.new_zeros(B, G, C).index_add_(1, gid, x)
    cnt = torch.bincount(gid, minlength=G).clamp_min(1).to(x.dtype)
    return out / cnt[None, :, None]


# ------------------------------------------------------------------ data statistics
@dataclass
class Stats:
    center: torch.Tensor     # [3]
    pos_scale: float
    vel_scale: float         # typical per-frame displacement
    vmax: float              # robust max per-frame displacement (decoder bound)
    logs_mean: torch.Tensor  # [3]
    logs_std: torch.Tensor   # [3]

    @staticmethod
    def from_prefix(xyz: torch.Tensor, scale: torch.Tensor, u: torch.Tensor) -> "Stats":
        """xyz [T, N, 3], scale [T, N, 3], u [T] frame times in dt units (prefix only)."""
        disp = (xyz[1:] - xyz[:-1]).norm(dim=-1) / (u[1:] - u[:-1]).clamp_min(1e-3)[:, None]
        sample = disp.flatten()[torch.randperm(disp.numel(), device=disp.device)[:2_000_000]]
        logs = torch.log(scale.abs().clamp_min(1e-8))
        return Stats(center=xyz.mean((0, 1)), pos_scale=float(xyz.std()),
                     vel_scale=float(sample.std()) + 1e-6,
                     vmax=float(torch.quantile(sample, 0.999)) + 1e-6,
                     logs_mean=logs.mean((0, 1)), logs_std=logs.std((0, 1)) + 1e-6)


# ------------------------------------------------------------------ modules
def _mlp(i: int, o: int, h: int | None = None) -> nn.Sequential:
    h = h or o
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def _transformer(cfg: RealConfig) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(cfg.token_dim, cfg.heads, dim_feedforward=2 * cfg.token_dim,
                                       batch_first=True, norm_first=True, dropout=0.0)
    return nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)


class HorizonEmbedding(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.mlp = _mlp(4, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float().view(-1, 1)
        return self.mlp(torch.cat([h / 16.0, torch.sin(h / 4), torch.cos(h / 4), torch.log1p(h)], dim=-1))


class Encoder(nn.Module):
    """E: causal history of L frames -> z = (S, P)."""

    def __init__(self, cfg: RealConfig, stats: Stats):
        super().__init__()
        D, G, L = cfg.token_dim, cfg.num_groups, cfg.context_len
        self.cfg, self.stats = cfg, stats
        in_dim = 3 + 3 + 3 * (L - 1) + (L - 1) + 4 + 3 + 4 + 3 + G
        self.gauss_mlp = nn.Sequential(_mlp(in_dim, D), nn.LayerNorm(D))
        self.group_in = _mlp(D + 6, D)
        self.ctx = _transformer(cfg)
        self.gauss_update = _mlp(2 * D, D)
        self.type_embed = nn.Parameter(torch.randn(2, D) * 0.02)
        self.norm_s, self.norm_p = nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, xyz, rot, scale, u, gid) -> Tuple[torch.Tensor, torch.Tensor]:
        """xyz [B,L,n,3], rot [B,L,n,4], scale [B,L,n,3], u [B,L] times, gid [n]."""
        st, G = self.stats, self.cfg.num_groups
        B, L, n, _ = xyz.shape
        cur = xyz[:, -1]
        centers = scatter_mean(cur, gid, G)
        gaps = (u[:, 1:] - u[:, :-1]).clamp_min(1e-3)                                 # [B, L-1]
        vel = (xyz[:, 1:] - xyz[:, :-1]) / gaps[:, :, None, None] / st.vel_scale      # [B, L-1, n, 3]
        logs = (torch.log(scale.abs().clamp_min(1e-8)) - st.logs_mean) / st.logs_std
        q_prev, q_cur = F.normalize(rot[:, -2], dim=-1), F.normalize(rot[:, -1], dim=-1)
        q_prev = q_prev * torch.sign((q_prev * q_cur).sum(-1, keepdim=True) + 1e-12)
        feats = torch.cat([
            (cur - st.center) / st.pos_scale,
            (cur - centers[:, gid]) / st.pos_scale * 4,
            vel.permute(0, 2, 1, 3).reshape(B, n, -1),
            (gaps - 1.0)[:, None, :].expand(B, n, L - 1),
            q_cur, logs[:, -1],
            (q_cur - q_prev) * 10, (logs[:, -1] - logs[:, -2]) * 10,
            F.one_hot(gid, G).float()[None].expand(B, n, G),
        ], dim=-1)
        P = self.gauss_mlp(feats)
        gvel = scatter_mean(vel[:, -1], gid, G)
        gpos = (centers - st.center) / st.pos_scale
        Sg = self.group_in(torch.cat([scatter_mean(P, gid, G), gvel, gpos], dim=-1))
        S = torch.cat([Sg.mean(1, keepdim=True) + self.type_embed[0], Sg + self.type_embed[1]], dim=1)
        S = self.norm_s(self.ctx(S))
        P = self.norm_p(P + self.gauss_update(torch.cat([P, S[:, 1:][:, gid]], dim=-1)))
        return S, P


class Transition(nn.Module):
    """Phi(z, h): horizon-conditioned residual update closed in latent space."""

    def __init__(self, cfg: RealConfig):
        super().__init__()
        D = cfg.token_dim
        self.h = HorizonEmbedding(D)
        self.net = _transformer(cfg)
        self.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))
        self.gauss = _mlp(3 * D, D, 2 * D)

    def forward(self, z, h, gid):
        S, P = z
        he = self.h(h)
        S1 = S + self.out(self.net(S + he[:, None]))
        n = P.shape[1]
        P1 = P + self.gauss(torch.cat([P, S1[:, 1:][:, gid], he[:, None].expand(-1, n, -1)], dim=-1))
        return S1, P1


class Decoder(nn.Module):
    """Selective geometry decoder: group SE(3) about the group centroid + per-Gaussian residual
    displacement r and log-scale change; Gaussian orientation follows its group rotation."""

    def __init__(self, cfg: RealConfig, stats: Stats):
        super().__init__()
        D = cfg.token_dim
        self.cfg, self.stats = cfg, stats
        self.h = HorizonEmbedding(D)
        self.group_head = _mlp(3 * D, 6, 2 * D)
        self.gauss_head = _mlp(3 * D, 6, 2 * D)
        for head in (self.group_head, self.gauss_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, z0, z1, src: Dict[str, torch.Tensor], h, gid) -> Dict[str, torch.Tensor]:
        st, G = self.stats, self.cfg.num_groups
        (S0, P0), (S1, P1) = z0, z1
        B, n, _ = P0.shape
        he = self.h(h)
        hs = h.float().view(B, 1, 1)
        go = self.group_head(torch.cat([S0[:, 1:], S1[:, 1:], he[:, None].expand(B, G, -1)], dim=-1))
        omega = torch.clamp(0.05 * hs, max=math.pi) * torch.tanh(go[..., :3])
        d = (2 * st.vmax * hs) * torch.tanh(go[..., 3:])
        po = self.gauss_head(torch.cat([P0, P1, he[:, None].expand(B, n, -1)], dim=-1))
        r = (st.vmax * hs) * torch.tanh(po[..., :3])
        log_s = torch.clamp(0.05 * hs, max=2.0) * torch.tanh(po[..., 3:])

        c = scatter_mean(src["xyz"], gid, G)
        q_g = axis_angle_to_quat(omega)[:, gid]
        xyz = quat_rotate(q_g, src["xyz"] - c[:, gid]) + c[:, gid] + d[:, gid] + r
        rot = quat_mul(q_g, src["rot"])
        scale = src["scale"] * torch.exp(log_s)
        out = {"xyz": xyz, "rot": rot, "scale": scale}
        if "vel" in src:  # constant-velocity base; zero-initialised heads start exactly at CV
            out["xyz"] = xyz + src["vel"] * hs
            out["vel"] = src["vel"]
        return out


class RealJEPA(nn.Module):
    def __init__(self, cfg: RealConfig, stats: Stats):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg, stats)
        self.transition = Transition(cfg)
        self.decoder = Decoder(cfg, stats)

    def source(self, ctx: Dict[str, torch.Tensor], uh: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decoder source state from a context window ctx [B, L, n, C] with times uh [B, L]:
        the last frame, plus its constant velocity over cv_k frames when cv_k > 0."""
        src = {k: v[:, -1] for k, v in ctx.items()}
        k = self.cfg.cv_k
        if k > 0:
            gap = (uh[:, -1] - uh[:, -1 - k]).clamp_min(1e-3)
            src["vel"] = (ctx["xyz"][:, -1] - ctx["xyz"][:, -1 - k]) / gap[:, None, None]
        return src

    def predict(self, z0, src, h_total: float, gid, segment: float | None = None):
        """Latent rollout from z0 to elapsed time h_total, decoding geometry segment by segment.
        segment=None -> one direct jump."""
        B = z0[0].shape[0]
        hs = [h_total] if segment is None else _segments(h_total, segment)
        z, geom = z0, src
        for h in hs:
            ht = torch.full((B,), h, device=z0[0].device)
            z1 = self.transition(z, ht, gid)
            geom = self.decoder(z, z1, geom, ht, gid)
            z = z1
        return geom


def _segments(total: float, seg: float) -> List[float]:
    k = max(1, math.ceil(total / seg - 1e-6))
    return [total / k] * k


# ------------------------------------------------------------------ losses
def latent_distance(za, zb) -> torch.Tensor:
    """Hierarchical latent distance: scene/group tokens and Gaussian tokens weighted equally."""
    return 0.5 * (F.smooth_l1_loss(za[0], zb[0]) + F.smooth_l1_loss(za[1], zb[1]))


def geometry_loss(pred, gt, stats: Stats) -> torch.Tensor:
    pos = (pred["xyz"] - gt["xyz"]).norm(dim=-1).mean() / stats.vel_scale
    rot = (1 - (F.normalize(pred["rot"], dim=-1) * F.normalize(gt["rot"], dim=-1)).sum(-1).abs()).mean() * 100
    logs = (torch.log(pred["scale"].abs().clamp_min(1e-8)) - torch.log(gt["scale"].abs().clamp_min(1e-8))).abs().mean() * 10
    return pos + rot + logs


def variance_loss(S: torch.Tensor) -> torch.Tensor:
    """VICReg variance hinge on group tokens (across batch and groups)."""
    x = S[:, 1:].reshape(-1, S.shape[-1])
    return F.relu(1.0 - torch.sqrt(x.var(0) + 1e-4)).mean()
