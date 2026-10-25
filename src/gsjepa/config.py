"""Experiment configs: plain dataclasses, filled from a YAML file plus ``key=value`` overrides."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


@dataclass
class Variant:
    """Which optional loss terms are active (L_end, L_geo and L_rep are always on)."""
    path: bool = False          # L_path      (Sec. 3.6)
    comp: bool = False          # L_comp      (Sec. 3.7)
    gcomp_se3: bool = False     # L_gcomp, group SE(3) term (Sec. 3.9)
    gcomp_match: bool = False   # L_gcomp, Gaussian-level D_match term


@dataclass
class ForecastConfig:
    """Experiments 1 and 2: controlled Gaussian worlds + small 4DGS-JEPA (paper Tables 5, 6)."""
    name: str = "exp"
    out_dir: str = "results/exp"

    # Controlled Gaussian world (Table 5)
    num_groups: int = 3
    gaussians_per_group: int = 6
    seq_len: int = 24
    context_len: int = 4
    train_scenes: int = 512
    val_scenes: int = 32
    test_scenes: int = 48

    # Model (Table 6)
    token_dim: int = 48
    horizon_dim: int = 16
    transformer_heads: int = 4
    transformer_layers: int = 1
    transformer_dropout: float = 0.0
    decoder_cov_stabilize: bool = False   # symmetrize + 1e-6 I after decoding (Exp2 only)

    # Optimisation (Table 6)
    batch_size: int = 48
    eval_batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 1e-5
    ema_tau: float = 0.99
    grad_clip: float = 1.0

    # Loss weights (App. D.2 / D.3)
    lambda_geo: float = 5.0
    lambda_path: float = 1.0
    lambda_comp: float = 0.5
    lambda_rep: float = 0.02
    lambda_cov_rep: float = 0.05
    var_floor: float = 0.5
    lambda_gcomp_se3: float = 0.4
    lambda_gcomp_match: float = 0.5
    center_huber_delta: float = 0.10
    cov_loss_weight: float = 0.25
    # "direct": L_geo on the direct terminal prediction only (Exp1).
    # "both_routes": L_geo on direct + composed predictions at every cumulative endpoint (Exp2).
    geo_supervision: str = "direct"

    # Fixed causal-feature scales (App. D.1.1)
    position_scale: float = 1.5
    relative_position_scale: float = 0.20
    velocity_scale: float = 0.05
    acceleration_scale: float = 0.01
    cov_delta_scale: float = 0.05

    # Horizons / temporal partitions (App. D.2)
    train_horizons: Tuple[int, ...] = (1, 2, 4, 8)
    unseen_horizons: Tuple[int, ...] = ()
    main_horizon: int = 8
    train_partitions: Dict[int, List[List[int]]] = field(default_factory=dict)
    primary_partitions: Dict[int, List[int]] = field(default_factory=dict)
    heldout_seen_segment_partitions: Dict[int, List[List[int]]] = field(default_factory=dict)
    unseen_segment_partitions: Dict[int, List[List[int]]] = field(default_factory=dict)
    partition_seed_offset: int = 100_000

    # Runs
    seeds: Tuple[int, ...] = (0, 1, 2)
    variants: Dict[str, Variant] = field(default_factory=dict)

    @property
    def num_gaussians(self) -> int:
        return self.num_groups * self.gaussians_per_group


@dataclass
class CorrespondenceConfig:
    """Experiment 3: correspondence stress test (paper Table 9)."""
    name: str = "exp3"
    out_dir: str = "results/exp3"

    active_primitives: int = 24
    reserve_slots: int = 6
    num_groups: int = 4
    semantic_dim: int = 6

    seeds: Tuple[int, ...] = (0, 1, 2)
    trials_per_setting: int = 50
    severities: Tuple[float, ...] = (0.10, 0.25, 0.40)

    sinkhorn_eps: float = 0.08
    sinkhorn_iters: int = 200
    sinkhorn_tol: float = 1e-7
    dustbin_cost: float = 1.20

    position_noise_std: float = 0.025
    logscale_noise_std: float = 0.020
    semantic_noise_std: float = 0.050

    id_dropout_rates: Tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)
    id_dropout_kind: str = "mixed"
    id_dropout_severity: float = 0.25

    bootstrap_samples: int = 2000
    bootstrap_seed: int = 12345


# ---------------------------------------------------------------- loading
def _coerce(cls, data: Dict[str, Any]):
    kwargs = {}
    names = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        if key == "variants":
            value = {k: Variant(**(v or {})) for k, v in value.items()}
        elif key.endswith("_partitions"):
            value = {int(k): v for k, v in value.items()}
        elif isinstance(value, list) and str(names[key].type).startswith("Tuple"):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str, overrides: Optional[Sequence[str]] = None):
    """Load a YAML config. ``kind: forecast`` or ``kind: correspondence`` selects the dataclass.

    ``overrides`` are ``key=value`` strings parsed as YAML, e.g. ``epochs=1`` or ``seeds=[0]``.
    """
    data = yaml.safe_load(Path(path).read_text())
    kind = data.pop("kind")
    for item in overrides or []:
        key, _, raw = item.partition("=")
        data[key.strip()] = yaml.safe_load(raw)
    cls = {"forecast": ForecastConfig, "correspondence": CorrespondenceConfig}[kind]
    return _coerce(cls, data)


def config_to_json(cfg) -> str:
    return json.dumps(dataclasses.asdict(cfg), indent=2, default=str)
