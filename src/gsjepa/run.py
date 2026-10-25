"""Single entry point for all experiments.

    python -m gsjepa.run configs/exp1.yaml                       # train + eval all variants x seeds
    python -m gsjepa.run configs/exp1.yaml --variants Endpoint --seeds 0   # one run (parallelise by hand)
    python -m gsjepa.run configs/exp1.yaml --aggregate           # rebuild tables from finished runs
    python -m gsjepa.run configs/exp3.yaml                       # correspondence stress test (CPU)
    python -m gsjepa.run configs/exp1.yaml --set epochs=1 train_scenes=16   # quick smoke test

Outputs go to ``<out_dir>/``: one sub-directory per (variant, seed) run, and a final
``report.md`` comparing reproduced numbers with the paper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from . import paper_numbers
from .config import ForecastConfig, config_to_json, load_config
from .data import build_datasets
from .experiments import exp1, exp2, exp3
from .train import train_variant

EXPERIMENTS = {"exp1": (exp1, paper_numbers.EXP1), "exp2": (exp2, paper_numbers.EXP2)}


def run_dir(out: Path, variant: str, seed: int) -> Path:
    stem = variant.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    return out / "runs" / f"{stem}_seed{seed}"


# ---------------------------------------------------------------- Exp1 / Exp2
def train_and_eval(cfg: ForecastConfig, out: Path, variants: List[str], seeds: List[int]) -> None:
    module, _ = EXPERIMENTS[cfg.name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    train_ds, val_ds, test_ds = build_datasets(cfg)
    print(f"device={device}  samples train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    for name in variants:
        for seed in seeds:
            rd = run_dir(out, name, seed)
            rd.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            model, teacher, hist, info = train_variant(name, cfg.variants[name], seed, cfg, train_ds, val_ds, device)
            summary, per_sample = module.evaluate(model, test_ds, cfg, device)
            summary = {"variant": name, "seed": seed, **info, **summary, "train_minutes": (time.time() - t0) / 60}
            hist.to_csv(rd / "history.csv", index=False)
            per_sample.to_csv(rd / "per_sample.csv", index=False)
            (rd / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
            torch.save({"model": model.state_dict(), "teacher": teacher.state_dict(), **info}, rd / "best.pt")
            print(f"==> {name} seed {seed}: " + ", ".join(
                f"{k}={summary[k]:.5f}" for k in module.TABLE_COLUMNS))


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.5f}±{std:.5f}"


def comparison_table(ours: pd.DataFrame, paper: Dict, columns: List[str]) -> str:
    """Markdown table with reproduced mean±std next to the paper's numbers."""
    lines = ["| Method | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for name, ref in paper.items():
        if name not in ours.index:
            continue
        cells = []
        for c in columns:
            m, s = ours.loc[name, (c, "mean")], ours.loc[name, (c, "std")]
            cells.append(f"{_fmt(m, s)}<br>paper {_fmt(*ref[c])}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def aggregate_forecast(cfg: ForecastConfig, out: Path) -> str:
    module, paper = EXPERIMENTS[cfg.name]
    summaries = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/summary.json"))]
    if not summaries:
        raise SystemExit(f"No finished runs under {out / 'runs'}")
    df = pd.DataFrame(summaries)
    df.to_csv(out / "summary_by_seed.csv", index=False)
    cols = module.TABLE_COLUMNS + getattr(module, "APPENDIX_COLUMNS", [])
    stats = df.groupby("variant")[cols].agg(["mean", "std"])
    stats.to_csv(out / "summary_mean_std.csv")

    _, _, test_ds = build_datasets(cfg)
    base = module.baselines(test_ds, cfg)
    base.to_csv(out / "baselines_per_sample.csv", index=False)

    # Horizon-wise curves (Fig. 2 / Table 8).
    per = pd.concat([pd.read_csv(p.parent / "per_sample.csv").assign(variant=s["variant"], seed=s["seed"])
                     for p, s in zip(sorted((out / "runs").glob("*/summary.json")), summaries)])
    metric_cols = [c for c in per.columns if c not in ("category", "horizon", "partition", "variant", "seed")]
    if "category" in per:
        per = per[per.category == "primary"]
    by_h = per.groupby(["variant", "seed", "horizon"])[metric_cols].mean().groupby(["variant", "horizon"]) \
        .agg(["mean", "std"])
    by_h.to_csv(out / "summary_by_horizon.csv")

    counts = df.groupby("variant").seed.nunique().to_dict()
    report = [f"# {cfg.name} reproduction report", "",
              f"Runs per variant: {counts}. Cells: reproduced mean±std over seeds, with the paper's value below.", ""]
    if cfg.name == "exp1":
        report += ["## Table 2", "", comparison_table(stats, paper, module.TABLE_COLUMNS), "",
                   "## Table 7 (held-out decompositions)", "", comparison_table(stats, paper, module.APPENDIX_COLUMNS),
                   ""]
        seen = base[base.horizon.isin(cfg.train_horizons)].groupby("baseline").fde.mean()
        h8 = base[base.horizon == 8].groupby("baseline").fde.mean()
        report += ["## Causal baselines (seen-horizon mean FDE)", ""]
        for name, ref in paper_numbers.EXP1_BASELINES.items():
            report.append(f"- {name}: {seen[name]:.5f} (paper {ref['seen_fde']:.5f}); at Δ=8: {h8[name]:.5f}")
        h8_model = per[per.horizon == 8].groupby(["variant", "seed"]).direct_fde.mean().groupby("variant").mean()
        report += ["", "Direct FDE at Δ=8 (paper: Endpoint + Composition 0.05687 vs constant velocity 0.07120):", ""]
        report += [f"- {v}: {x:.5f}" for v, x in h8_model.items()]
    else:
        report += ["## Table 3 (Δ=8)", "", comparison_table(stats, paper, module.TABLE_COLUMNS), ""]
        b8 = base[base.horizon == cfg.main_horizon].mean()
        report += ["## Reference baselines at Δ=8", ""]
        report += [f"- {k}: {b8[k]:.5f} (paper {v:.5f})" for k, v in paper_numbers.EXP2_BASELINES.items()]
        report += ["", "## Table 8 (horizon-wise direct–composed consistency)", "",
                   "| Δ | Variant | E_T^dc | D_match |", "|---|---|---|---|"]
        for (v, h), row in by_h.iterrows():
            if h > 1:
                report.append(f"| {h} | {v} | {_fmt(row[('transform_dc', 'mean')], row[('transform_dc', 'std')])} | "
                              f"{_fmt(row[('d_match', 'mean')], row[('d_match', 'std')])} |")
    text = "\n".join(report) + "\n"
    (out / "report.md").write_text(text)
    return text


# ---------------------------------------------------------------- Exp3
def run_exp3(cfg, out: Path) -> str:
    tables = exp3.run(cfg, out)
    t4 = tables["table4"]
    report = ["# exp3 reproduction report", "", "## Table 4", "",
              comparison_table(t4, paper_numbers.EXP3, exp3.TABLE_COLUMNS), "",
              "## Table 10 (mixed perturbations)", "", "```", tables["table10"].round(5).to_string(), "```", "",
              "## Table 11 (persistent-ID dropout, severity 0.25)", "", "```", tables["table11"].round(5).to_string(),
              "```", ""]
    text = "\n".join(report)
    (out / "report.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--variants", nargs="*", help="subset of variant names (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, help="subset of seeds (default: config seeds)")
    ap.add_argument("--aggregate", action="store_true", help="only rebuild tables from finished runs")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(config_to_json(cfg))

    if cfg.name == "exp3":
        print(run_exp3(cfg, out))
        return
    if not args.aggregate:
        train_and_eval(cfg, out, args.variants or list(cfg.variants), args.seeds or list(cfg.seeds))
    if args.aggregate or (not args.variants and not args.seeds):
        print(aggregate_forecast(cfg, out))


if __name__ == "__main__":
    main()
