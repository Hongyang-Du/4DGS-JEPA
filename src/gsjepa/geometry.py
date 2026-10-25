"""Small geometry toolbox: SO(3)/SE(3) maps and SPD covariance coordinates.

Torch functions are batched over leading dims; ``np_*`` helpers are used only by
the NumPy data generator.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------- rotations
def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) cross-product matrix."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(*v.shape[:-1], 3, 3)


def axis_angle_to_matrix(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues map so(3) -> SO(3), with a Taylor expansion near zero."""
    W = skew(w)
    theta2 = (w * w).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    small = theta2 < 1e-6
    A = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
                    torch.sin(theta) / theta.clamp_min(1e-8))
    B = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
                    (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(*w.shape[:-1], 3, 3)
    return I + A[..., None] * W + B[..., None] * (W @ W)


def rotation_angle(R: torch.Tensor) -> torch.Tensor:
    """Geodesic angle of a rotation matrix (atan2 form: smooth gradient at 0)."""
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    s = 0.5 * torch.linalg.norm(v, dim=-1)
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.atan2(s, c)


# ---------------------------------------------------------------- SE(3)
def se3_distance(R1, d1, R2, d2) -> torch.Tensor:
    """Controlled SE(3) surrogate (paper App. D.3): sqrt(theta(R1^T R2)^2 + |d2-d1|^2)."""
    ang = rotation_angle(R1.transpose(-1, -2) @ R2)
    td = torch.linalg.norm(d2 - d1, dim=-1)
    return torch.sqrt(ang.pow(2) + td.pow(2) + 1e-12)


def compose_transform(R_prev, d_prev, R_seg, d_seg):
    """T_total = T_seg * T_prev (the rightmost transform acts first)."""
    return R_seg @ R_prev, (R_seg @ d_prev.unsqueeze(-1)).squeeze(-1) + d_seg


def relative_transform(R0, d0, R1, d1):
    """T_rel = T_1 * T_0^{-1}: maps a point posed at time 0 to its pose at time 1."""
    Rrel = R1 @ R0.transpose(-1, -2)
    drel = d1 - (Rrel @ d0.unsqueeze(-1)).squeeze(-1)
    return Rrel, drel


# ---------------------------------------------------------------- SPD
def cov_to_log_cholesky(cov: torch.Tensor) -> torch.Tensor:
    """6-D SPD coordinate: log of Cholesky diagonal + three off-diagonal entries."""
    I = torch.eye(3, dtype=cov.dtype, device=cov.device)
    L = torch.linalg.cholesky(cov + 1e-6 * I)
    diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-8).log()
    off = torch.stack([L[..., 1, 0], L[..., 2, 0], L[..., 2, 1]], dim=-1)
    return torch.cat([diag, off], dim=-1)


def affine_spd_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Affine-invariant SPD distance ||log(A^{-1/2} B A^{-1/2})||_F (used for evaluation)."""
    evals, evecs = torch.linalg.eigh(A)
    invsqrt = evecs @ torch.diag_embed(evals.clamp_min(1e-8).rsqrt()) @ evecs.transpose(-1, -2)
    lam = torch.linalg.eigvalsh(invsqrt @ B @ invsqrt).clamp_min(1e-8)
    return torch.linalg.norm(torch.log(lam), dim=-1)


# ---------------------------------------------------------------- NumPy (data generator)
def np_axis_angle_to_matrix(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-10:
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float32)
        return np.eye(3, dtype=np.float32) + K
    a = w / theta
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float32)
    return (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)).astype(np.float32)


def np_make_spd(log_scales: np.ndarray, rot: Optional[np.ndarray] = None) -> np.ndarray:
    D = np.diag(np.exp(2.0 * log_scales).astype(np.float32))
    if rot is None:
        return D
    return (rot @ D @ rot.T).astype(np.float32)
