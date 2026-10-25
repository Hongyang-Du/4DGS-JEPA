"""Experiment 3: hybrid correspondence stress test (paper Sec. 5.4, Tables 4, 10, 11)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..config import CorrespondenceConfig
from ..correspondence import (hide_persistent_ids, hybrid_plan, index_plan, make_source, ot_plan,
                              perturb_source, plan_metrics)

METHODS = ["Index", "OT only", "Hybrid"]
KINDS = ["permutation", "split", "merge", "death", "birth", "mixed"]
CORE_METRICS = ["lineage_mass", "geometry_discrepancy", "birth_precision", "birth_recall", "birth_f1",
                "death_precision", "death_recall", "death_f1", "event_f1_macro"]
TABLE_COLUMNS = ["Permutation lineage", "Split/merge lineage", "Birth/death F1", "Geometry discrepancy"]


def trial_rng(seed, severity_idx, trial, kind_idx, stream=0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, severity_idx, trial, kind_idx, stream]))


def source_seed(seed, severity_idx, trial) -> int:
    return 100_000 + 100_000 * seed + 1_000 * severity_idx + trial


def run_trials(cfg: CorrespondenceConfig) -> pd.DataFrame:
    """seeds x severities x trials x kinds x methods. The source scene is shared across kinds (paired)."""
    rows = []
    for seed in cfg.seeds:
        for si, severity in enumerate(cfg.severities):
            for trial in range(cfg.trials_per_setting):
                src = make_source(source_seed(seed, si, trial), cfg)
                for ki, kind in enumerate(KINDS):
                    tgt = perturb_source(src, kind, severity, trial_rng(seed, si, trial, ki), cfg)
                    plans = {"Index": index_plan(src, tgt), "OT only": ot_plan(src, tgt, cfg),
                             "Hybrid": hybrid_plan(src, tgt, cfg)}
                    for method, P in plans.items():
                        rows.append({"seed": seed, "severity": severity, "trial": trial, "kind": kind,
                                     "method": method, **plan_metrics(src, tgt, P)})
    return pd.DataFrame(rows)


def run_id_dropout(cfg: CorrespondenceConfig) -> pd.DataFrame:
    """Hide a growing fraction of valid persistent ids under mixed perturbations (Table 11)."""
    si = list(cfg.severities).index(cfg.id_dropout_severity)
    ki = KINDS.index(cfg.id_dropout_kind)
    rows = []
    for seed in cfg.seeds:
        for trial in range(cfg.trials_per_setting):
            src = make_source(source_seed(seed, si, trial), cfg)
            tgt = perturb_source(src, cfg.id_dropout_kind, cfg.id_dropout_severity,
                                 trial_rng(seed, si, trial, ki), cfg)
            m_ot = plan_metrics(src, tgt, ot_plan(src, tgt, cfg))
            for ri, rate in enumerate(cfg.id_dropout_rates):
                hidden = hide_persistent_ids(tgt, rate, trial_rng(seed, si, trial, ki, stream=100 + ri))
                m_h = plan_metrics(src, hidden, hybrid_plan(src, hidden, cfg))
                base = {"seed": seed, "trial": trial, "id_dropout_rate": rate}
                rows.append({**base, "method": "Hybrid", **m_h})
                rows.append({**base, "method": "OT only", **m_ot})
    return pd.DataFrame(rows)


def _paper_metrics(d: pd.DataFrame) -> Dict[str, float]:
    return {"Permutation lineage": d.loc[d.kind == "permutation", "lineage_mass"].mean(),
            "Split/merge lineage": d.loc[d.kind.isin(["split", "merge"]), "lineage_mass"].mean(),
            "Birth/death F1": d.loc[d.kind.isin(["birth", "death"]), "event_f1_macro"].mean(),
            "Geometry discrepancy": d["geometry_discrepancy"].mean()}


def bootstrap_ci(values, rng, n_boot=2000, alpha=0.05):
    x = np.asarray(values, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boot = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def run(cfg: CorrespondenceConfig, out: Path) -> Dict[str, pd.DataFrame]:
    res = run_trials(cfg)
    res.to_csv(out / "all_trials.csv", index=False)

    # Table 4: per-seed aggregate, then mean / std over seeds.
    by_seed = pd.DataFrame([{"seed": s, "method": m, **_paper_metrics(res[(res.seed == s) & (res.method == m)])}
                            for s in cfg.seeds for m in METHODS])
    table4 = by_seed.groupby("method")[TABLE_COLUMNS].agg(["mean", "std"]).loc[METHODS]
    table4.to_csv(out / "table4_mean_std.csv")

    # Bootstrap CIs over (seed, trial) clusters.
    clusters = pd.DataFrame([{"seed": s, "trial": t, "method": m, **_paper_metrics(d)}
                             for (s, t, m), d in res.groupby(["seed", "trial", "method"])])
    rng = np.random.default_rng(cfg.bootstrap_seed)
    ci = []
    for m in METHODS:
        row = {"method": m}
        for col in TABLE_COLUMNS:
            row[f"{col} mean"], row[f"{col} ci_low"], row[f"{col} ci_high"] = bootstrap_ci(
                clusters.loc[clusters.method == m, col].values, rng, cfg.bootstrap_samples)
        ci.append(row)
    pd.DataFrame(ci).set_index("method").to_csv(out / "table4_bootstrap_ci.csv")

    # Table 10: mixed perturbations vs severity.
    mixed = res[res.kind == "mixed"].groupby(["seed", "severity", "method"])[CORE_METRICS].mean().reset_index()
    table10 = mixed.groupby(["severity", "method"])[["lineage_mass", "geometry_discrepancy", "event_f1_macro"]] \
        .agg(["mean", "std"])
    table10.to_csv(out / "table10_mixed_severity.csv")

    # Table 11: persistent-ID dropout.
    drop = run_id_dropout(cfg)
    drop.to_csv(out / "id_dropout_trials.csv", index=False)
    table11 = drop.groupby(["seed", "id_dropout_rate", "method"])[["lineage_mass", "geometry_discrepancy"]] \
        .mean().groupby(["id_dropout_rate", "method"]).mean().unstack("method")
    table11.to_csv(out / "table11_id_dropout.csv")

    return {"table4": table4, "table10": table10, "table11": table11}
