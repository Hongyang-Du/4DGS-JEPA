"""Check that the refactored package reproduces the original Colab notebooks bit-for-bit.

Executes the notebook's own code cells (definitions only) and the refactored package on a small
config, trains one (variant, seed) with each, and compares validation history + test metrics.

    python scripts/check_equivalence.py exp1 "Endpoint + Path + Composition"
    python scripts/check_equivalence.py exp2 "Full geometry composition"
    python scripts/check_equivalence.py exp3
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "exp1": "4DGS_JEPA_Exp1_Temporal_Composition_v3_MotionAware.ipynb",
    "exp2": "4DGS_JEPA_Exp2_Geometry_Composition_v2_MotionAware.ipynb",
    "exp3": "4DGS_JEPA_Exp3_Hybrid_Correspondence_v2_FairOT.ipynb",
}
DEF_CELLS = {"exp1": range(2, 12), "exp2": range(2, 12), "exp3": range(1, 8)}
SMALL = dict(epochs=2, train_scenes=24, val_scenes=4, test_scenes=4)


def notebook_namespace(exp):
    nb = json.loads((REPO / "notebooks" / NOTEBOOKS[exp]).read_text())
    ns = {"display": print}
    os.chdir(tempfile.mkdtemp())
    for i in DEF_CELLS[exp]:
        cell = nb["cells"][i]
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{exp}-cell{i}", "exec"), ns)
    return ns


def check_forecast(exp, variant):
    import torch
    from gsjepa.config import load_config
    from gsjepa.data import build_datasets
    from gsjepa.experiments import exp1 as e1, exp2 as e2
    from gsjepa.train import train_variant

    ns = notebook_namespace(exp)
    old_cfg = ns["Config"](**SMALL, seeds=(0,))
    old_variant = {"+SE(3) composition": "+SE3 composition"}.get(variant, variant)
    tr, va, te = (ns["GaussianWorldDataset"](s, old_cfg) for s in ns["build_splits"](old_cfg))
    _, _, old_hist, _ = old_model = ns["train_variant"](old_variant, 0, old_cfg, tr, va)
    old_summary = ns["evaluate_model"](old_model[0], te, old_cfg)[0]

    cfg = load_config(str(REPO / "configs" / f"{exp}.yaml"), [f"{k}={v}" for k, v in SMALL.items()])
    train_ds, val_ds, test_ds = build_datasets(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, new_hist, _ = train_variant(variant, cfg.variants[variant], 0, cfg, train_ds, val_ds, device)
    new_summary = (e1 if exp == "exp1" else e2).evaluate(model, test_ds, cfg, device)[0]

    rename = {"exp1": {"unseen_horizon_rollout_fde": "unseen_fde",
                       "heldout_seen_segments_comp_err": "heldout_comp_err",
                       "unseen_segments_comp_err": "unseen_seg_comp_err"},
              "exp2": {"fde": "direct_fde"}}[exp]
    worst = 0.0
    for col in ["val_score", "val_direct_fde", "val_rollout_fde"]:
        diff = np.abs(old_hist[col].values - new_hist[col].values).max()
        worst = max(worst, diff)
        print(f"  history {col:28s} max|diff| = {diff:.3e}")
    for k, v in old_summary.items():
        nk = rename.get(k, k)
        if nk in new_summary:
            diff = abs(float(v) - float(new_summary[nk]))
            worst = max(worst, diff / max(abs(float(v)), 1e-12))
            print(f"  test {nk:32s} notebook={float(v):.8f} ours={float(new_summary[nk]):.8f}")
    return worst


def check_exp3():
    from gsjepa.config import CorrespondenceConfig
    from gsjepa.experiments import exp3

    ns = notebook_namespace("exp3")
    cfg_old = ns["Config"](seeds=(0,), trials_per_setting=5)
    cfg = CorrespondenceConfig(seeds=(0,), trials_per_setting=5)
    old, new = [], []
    for si, sev in enumerate(cfg.severities):
        for trial in range(5):
            src_o = ns["make_source"](ns["source_seed"](0, si, trial), cfg_old)
            for ki, kind in enumerate(exp3.KINDS):
                tgt_o = ns["perturb_source"](src_o, kind, sev, ns["seed_trial_rng"](0, si, trial, ki), cfg_old)
                for P in (ns["index_plan"](src_o, tgt_o), ns["ot_plan"](src_o, tgt_o, cfg_old),
                          ns["hybrid_plan"](src_o, tgt_o, cfg_old)):
                    old.append(ns["metrics"](src_o, tgt_o, P)["lineage_mass"])
    res = exp3.run_trials(cfg)
    new = res.lineage_mass.values
    diff = np.nanmax(np.abs(np.asarray(old) - new))
    print(f"  exp3 lineage_mass over {len(new)} method-trials: max|diff| = {diff:.3e}")
    return diff


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    exp = sys.argv[1]
    worst = check_exp3() if exp == "exp3" else check_forecast(exp, sys.argv[2])
    print(f"WORST DIFF = {worst:.3e} -> {'EQUIVALENT' if worst < 1e-5 else 'MISMATCH'}")
