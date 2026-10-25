"""Causal history slicing, direct prediction and recursive (composed) rollout."""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .geometry import compose_transform, relative_transform
from .model import JEPA


def get_history(batch: Dict[str, torch.Tensor], end: torch.Tensor, context_len: int):
    """Slice the L-frame history ending at absolute frame index ``end`` [B] (inclusive)."""
    B = end.shape[0]
    idx = end[:, None] - context_len + 1 + torch.arange(context_len, device=end.device)[None]
    assert int(idx.min()) >= 0
    bidx = torch.arange(B, device=end.device)[:, None]
    return batch["mu"][bidx, idx], batch["cov"][bidx, idx]


def at_time(batch: Dict[str, torch.Tensor], key: str, time: torch.Tensor) -> torch.Tensor:
    return batch[key][torch.arange(time.shape[0], device=time.device), time]


def horizon(B: int, value: float, device) -> torch.Tensor:
    return torch.full((B,), float(value), device=device)


def direct(model: JEPA, z0, src_mu, src_cov, group_ids, H: int):
    """One direct transition Z_t -> Z_{t+H} and its decoded geometry G_{t+H}."""
    h = horizon(z0.shape[0], H, z0.device)
    z = model.transition(z0, h)
    return z, model.decoder(z0, z, src_mu, src_cov, group_ids, h)


def rollout(model: JEPA, z0, src_mu, src_cov, group_ids, parts: Sequence[int]) -> List[Dict]:
    """Composed path along partition ``parts`` = (delta_1, ..., delta_J)  (eqs. 20, 36, 40).

    Each step returns the cumulative horizon, latent state, decoded geometry, and the
    chronologically composed group transform T_J ... T_1 accumulated so far."""
    z, mu, cov = z0, src_mu, src_cov
    B, G = z.shape[0], model.decoder.cfg.num_groups
    R_total = torch.eye(3, device=z.device, dtype=z.dtype).expand(B, G, 3, 3).clone()
    d_total = torch.zeros(B, G, 3, device=z.device, dtype=z.dtype)
    outputs, elapsed = [], 0
    for dlt in parts:
        h = horizon(B, dlt, z.device)
        zn = model.transition(z, h)
        geom = model.decoder(z, zn, mu, cov, group_ids, h)
        R_total, d_total = compose_transform(R_total, d_total, geom["R"], geom["d"])
        elapsed += dlt
        z, mu, cov = zn, geom["mu"], geom["cov"]
        outputs.append({"elapsed": elapsed, "z": z, "geom": geom, "R_total": R_total, "d_total": d_total})
    return outputs


def true_group_transform(batch: Dict[str, torch.Tensor], end: torch.Tensor):
    """Ground-truth relative group transform from each sample's context endpoint t to ``end``."""
    B, G = batch["mu"].shape[0], batch["group_R"].shape[2]
    bidx = torch.arange(B, device=end.device)[:, None]
    gidx = torch.arange(G, device=end.device)[None, :]
    t0 = batch["t"][:, None]
    return relative_transform(batch["group_R"][bidx, t0, gidx], batch["group_d"][bidx, t0, gidx],
                              batch["group_R"][bidx, end[:, None], gidx],
                              batch["group_d"][bidx, end[:, None], gidx])
