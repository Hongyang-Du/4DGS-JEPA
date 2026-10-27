"""Aggregate the per-scene controlled comparisons into results/hypernerf/fair_summary_tables.md (+ .json).

    PYTHONPATH=src python scripts/fair_summary.py [--dataset dnerf]
Needs scripts/fair_report.py to have been run for every scene (reads <scene>/fair/summary.json).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SCENES = {"hypernerf": ["slice-banana", "cut-lemon1", "chickchicken", "split-cookie", "keyboard", "vrig-3dprinter"],
          "dnerf": ["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]}
# ODE-GS (ICLR'26) Table 3, HyperNeRF extrapolation (their own Stage 1 / protocol; reference only)
PAPER = {
    "slice-banana": {"ODE-GS": (21.29, .6437, .3230), "Deformable-GS": (21.73, .6530, .3241)},
    "cut-lemon1": {"ODE-GS": (21.69, .6964, .3098), "Deformable-GS": (21.36, .6950, .3207)},
    "chickchicken": {"ODE-GS": (20.29, .7254, .3023), "Deformable-GS": (19.03, .6996, .3167)},
    "split-cookie": {"ODE-GS": (20.72, .6593, .2406), "Deformable-GS": (17.84, .5698, .2945)},
    "keyboard": {"ODE-GS": (21.06, .7399, .2327), "Deformable-GS": (19.98, .6957, .2497)},
    "vrig-3dprinter": {"ODE-GS": (20.22, .6946, .2242), "Deformable-GS": (19.89, .6775, .2378)},
}
ROWS = [("full/jepa_full", "4DGS-JEPA (controlled, 90k)"), ("full/odegs_full", "ODE-GS (controlled, 90k)"),
        ("full/deform_query", "Deformable-GS (Stage 1 queried)"), ("full/static", "Static (last prefix state)"),
        ("full/rigid", "Group-rigid (k chosen on inner split)"), ("full/cv", "Constant velocity (k chosen on inner split)")]
KEYS = ["psnr", "ssim", "lpips", "dyn_psnr", "dyn_ssim", "dyn_lpips"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SCENES), default="hypernerf")
    ap.add_argument("--no_paper", action="store_true", help="leave out the ODE-GS paper reference rows (HyperNeRF)")
    args = ap.parse_args()
    D = f"results/{args.dataset}"
    paper = args.dataset == "hypernerf" and not args.no_paper
    data = {}
    for s in SCENES[args.dataset]:
        p = f"{D}/{s}/fair/summary.json"
        if not os.path.exists(p):
            continue
        summ = json.load(open(p))["summary"]
        base = json.load(open(f"{D}/{s}/fair/baselines/metrics.json"))
        summ["full/cv"] = {k: base["full"][f"cv{base['selected_k']}"].get(k) for k in KEYS}
        summ["_k"] = base["selected_k"]
        if "selected_k_rigid" in base:
            summ["full/rigid"] = {k: base["full"][f"rigid{base['selected_k_rigid']}"].get(k) for k in KEYS}
            summ["_k_rigid"] = base["selected_k_rigid"]
        data[s] = summ
    L = []
    for metric, title in [("psnr", "PSNR ↑ (full image)"), ("dyn_psnr", "PSNR ↑ (dynamic-region mask)"),
                          ("ssim", "SSIM ↑ (full image)"), ("lpips", "LPIPS ↓ (full image)"),
                          ("dyn_lpips", "LPIPS ↓ (dynamic-region mask)")]:
        L += [f"### {title}", "", "| method | " + " | ".join(data) + " | mean |", "|---|" + "---|" * (len(data) + 1)]
        for key, label in ROWS:
            vals = [data[s].get(key, {}).get(metric) for s in data]
            ok = [v for v in vals if v is not None]
            p = 2 if "psnr" in metric else 4
            cells = [f"{v:.{p}f}" if v is not None else "–" for v in vals]
            mean = f"{np.mean(ok):.{p}f}" if len(ok) == len(vals) and ok else "–"
            L.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")
        if paper and metric in ("psnr", "ssim", "lpips"):
            i = {"psnr": 0, "ssim": 1, "lpips": 2}[metric]
            for m in ["ODE-GS", "Deformable-GS"]:
                vals = [PAPER[s][m][i] for s in data]
                p = 2 if metric == "psnr" else 4
                L.append(f"| _paper ref: {m} (ODE-GS Table 3)_ | " + " | ".join(f"{v:.{p}f}" for v in vals)
                         + f" | {np.mean(vals):.{p}f} |")
        L.append("")
    L.append("CV window k chosen on each scene's inner split: " + ", ".join(f"{s}: k={data[s]['_k']}" for s in data))
    L.append("Group-rigid window k (same rule): " + ", ".join(f"{s}: k={data[s].get('_k_rigid', '–')}" for s in data))
    open(f"{D}/fair_summary_tables.md", "w").write("\n".join(L) + "\n")
    json.dump(data, open(f"{D}/fair_summary.json", "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
