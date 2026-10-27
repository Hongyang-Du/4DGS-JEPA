"""Level II temporal-consistency diagnostics (extra-experiments proposal, Sec. 1.3) on the trained runs of one scene.

    python -m gsjepa.real.consistency scene --source /data/HyperNeRF/interp/slice-banana \
        --root results/hypernerf/slice-banana --time_split 0.9 --inner_cut 0.8
    python -m gsjepa.real.consistency summary --dataset hypernerf
scene writes <root>/fair/level2.json; summary writes results/<dataset>/level2_summary.md (+ .json).

Every method forecasts from its split's origin t0 (last Stage-1 time it may see) on the uniform horizon grid
t0 + h * dt, h = 1..H, where H covers the split's scored cameras. Methods: JEPA (rollout = headline mode, and
direct), ODE-GS, Static, CV and group-rigid (windows chosen on the inner split), Deformable-GS query.

  target error   inner split only. Its times are Stage-1 training frames, so Stage-1 states there are a fitted
                 reference, while every inner stage-2 run and baseline has only seen t <= inner_cut.
                 ADE / FDE of Gaussian centres over all and over dynamic Gaussians, plus the ratio to Static.
  error growth   least-squares slope of ADE vs horizon (inner) and of per-camera PSNR vs elapsed time (both splits;
                 dB per 0.1 time, from the runs' metrics.json).
  bg drift       mean displacement from the origin of the static Gaussians (prefix motion below the median).
  group distort. RMS residual after the best rigid fit origin -> forecast, per motion group (the JEPA's groups),
                 averaged over Gaussians. Stage-1 reference: the same over the last H prefix frames.
  accel / jerk   norms of the 2nd / 3rd finite differences along the grid (origin included), dynamic Gaussians,
                 per dt^2 / dt^3. Stage-1 reference: the same over the last H prefix frames.
  JEPA CompErr   direct jump vs composed rollouts at horizons H/4, H/2, H over partitions into equal segments of
                 1/8, 1/4, 1/2 and 1 x the JEPA's maximum training horizon (frame counts differ a lot between scenes;
                 1/2 is the rollout segment used for the headline numbers) and 3 random partitions into 2-6 pieces:
                 latent distance (as in training) and centre distance,
                 reported with each path's target error (inner), as the proposal requires.
Dynamic / static Gaussians come from the prefix only (dynamic_gaussians score: top 10% dynamic, bottom 50% static).
Distances are in scene units; *_rel columns divide by the mean origin displacement of the dynamic Gaussians under
Stage 1 (inner) or under the method's own forecast (full), so scenes can be averaged.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .fair import (Split, build_odegs, dynamic_gaussians, group_rigid_motion, group_rigid_state, jepa_state_fn,
                   odegs_predictor, rigid_fit)
from .frontend import FrontEnd
from .jepa import RealConfig, RealJEPA, _segments, latent_distance

PARTITIONS = {"seg1/8": 1 / 8, "seg1/4": 1 / 4, "seg1/2": 1 / 2, "seg1": 1.0}  # x max training horizon
N_RANDOM = 3


def grid(fe: FrontEnd, sp: Split) -> np.ndarray:
    end = max(c.fid.item() for c in sp.cams)
    H = max(3, int(np.floor((end - sp.t_last) / sp.dt + 1e-6)))
    return sp.t_last + sp.dt * np.arange(1, H + 1)


def stack(fn, ts) -> torch.Tensor:
    return torch.stack([fn(float(t))["xyz"].float() for t in ts])  # [H, N, 3]


def ade(pred: torch.Tensor, ref: torch.Tensor, sel=None) -> torch.Tensor:
    d = (pred - ref).norm(dim=-1)  # [H, N]
    return (d if sel is None else d[:, sel]).mean(1)  # per horizon


def group_distortion(x0: torch.Tensor, x1: torch.Tensor, gid: torch.Tensor, G: int) -> float:
    tot, n = 0.0, 0
    for g in range(G):
        m = gid == g
        if m.sum() < 3:
            continue
        R, t = rigid_fit(x0[m], x1[m])
        tot += float(((x0[m] @ R.T + t - x1[m]) ** 2).sum(-1).sum())
        n += int(m.sum())
    return (tot / max(n, 1)) ** 0.5


def smoothness(traj: torch.Tensor, sel, dt: float) -> tuple[float, float]:
    """traj [H+1, N, 3] including the origin -> mean |2nd diff| / dt^2, mean |3rd diff| / dt^3 (dynamic Gaussians)."""
    x = traj[:, sel]
    a = (x[2:] - 2 * x[1:-1] + x[:-2]).norm(dim=-1).mean() / dt ** 2
    j = (x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]).norm(dim=-1).mean() / dt ** 3 if len(x) > 3 else torch.tensor(float("nan"))
    return float(a), float(j)


def psnr_slope(rows, t0: float) -> float:
    t = np.array([r["fid"] - t0 for r in rows])
    y = np.array([r["psnr"] for r in rows])
    return float(np.polyfit(t, y, 1)[0] * 0.1) if len(set(t.round(6))) > 1 else float("nan")


def load_jepa(path: str):
    ck = torch.load(path, weights_only=False)
    model = RealJEPA(RealConfig(**ck["cfg"]), ck["stats"]).cuda().eval()
    model.load_state_dict(ck["model"])
    return model, ck["gid"].cuda(), ck


def jepa_context(fe, model, gid, sp, dt):
    L, S = model.cfg.context_len, model.cfg.context_stride
    fids = sp.fids[len(sp.fids) - 1 - S * (L - 1)::S]
    traj = {k: v.cuda() for k, v in fe.trajectory(fids).items()}
    u = torch.tensor(fids / dt, dtype=torch.float32, device="cuda")[None]
    with torch.no_grad():
        ctx = {k: v[None] for k, v in traj.items()}
        return model.encoder(ctx["xyz"], ctx["rot"], ctx["scale"], u, gid), model.source(ctx, u)


@torch.no_grad()
def jepa_path(model, z0, src, hs, gid):
    """Latent rollout over the elapsed-time pieces hs; returns the final latent and decoded centres."""
    z, geom = z0, src
    for h in hs:
        ht = torch.full((1,), float(h), device="cuda")
        z1 = model.transition(z, ht, gid)
        geom = model.decoder(z, z1, geom, ht, gid)
        z = z1
    return z, geom["xyz"][0]


def random_partition(total: int, rng: np.random.Generator) -> list:
    k = int(rng.integers(2, min(total, 6) + 1))
    cuts = np.sort(rng.choice(np.arange(1, total), size=k - 1, replace=False))
    return np.diff(np.concatenate([[0], cuts, [total]])).astype(float).tolist()


def scene(args) -> dict:
    fe = FrontEnd(args.source, os.path.join(args.root, "stage1"), args.time_split)
    F = os.path.join(args.root, "fair")
    base = json.load(open(os.path.join(F, "baselines", "metrics.json")))
    kc, kr = base["selected_k"], base.get("selected_k_rigid")
    out = {"selected_k": kc, "selected_k_rigid": kr}
    for split in ["inner", "full"]:
        sp = Split(fe, split, args.inner_cut)
        ts = grid(fe, sp)
        H, dt = len(ts), sp.dt
        score_sel = dynamic_gaussians(fe, sp, 0.3, 0.9)
        static_sel = ~dynamic_gaussians(fe, sp, 0.3, 0.5)
        n = min(17, len(sp.fids))
        hist = fe.trajectory(sp.fids[-n:])
        last = {k: v[-1].cuda() for k, v in hist.items()}
        x0 = last["xyz"]
        u = sp.fids[-n:] / dt
        fns = {"static": lambda f: last}
        v = (hist["xyz"][-1] - hist["xyz"][-1 - kc]).cuda() / float(u[-1] - u[-1 - kc])
        fns[f"cv{kc}"] = lambda f: {**last, "xyz": last["xyz"] + v * ((f - sp.t_last) / dt)}
        jp, jo = os.path.join(F, "runs", f"jepa_{split}", "last.pt"), os.path.join(F, "runs", f"odegs_{split}", "last.pt")
        model = gid = None
        if os.path.exists(jp):
            model, gid, _ = load_jepa(jp)
            fns["jepa"] = jepa_state_fn(fe, model, gid, sp, model.cfg.rollout_segment, dt)
            fns["jepa_direct"] = jepa_state_fn(fe, model, gid, sp, None, dt)
        if gid is None:
            from .run_scene import kmeans_groups
            gid = kmeans_groups(fe.trajectory(sp.fids)["xyz"].cuda(), 32, seed=0)
        G = int(gid.max()) + 1
        if kr is not None:
            mo = group_rigid_motion(hist["xyz"][-1 - kr].cuda(), x0, gid, G)
            span = float(u[-1] - u[-1 - kr])
            fns[f"rigid{kr}"] = lambda f: group_rigid_state(last, mo, gid, (f - sp.t_last) / dt / span)
        if os.path.exists(jo):
            ck = torch.load(jo, weights_only=False)
            om = build_odegs(ck["cfg"])
            om.load_state_dict(ck["model_state_dict"])
            fns["odegs"] = odegs_predictor(fe, om, ck["cfg"], sp.t_last, ts)
        fns["deform_query"] = lambda f: {k: v.cuda() for k, v in fe.state_at(f).items()}

        # Stage-1 references: target states on the grid (inner only) and the last H prefix frames
        ref = stack(fe.state_at, ts) if split == "inner" else None
        pre_ts = sp.fids[-(H + 1):] if len(sp.fids) > H else sp.fids
        pre = stack(fe.state_at, pre_ts)
        s1 = {"accel": None, "jerk": None}
        s1["accel"], s1["jerk"] = smoothness(pre, score_sel, dt)
        s1["group_distortion"] = float(np.mean([group_distortion(pre[0], pre[h], gid, G) for h in range(1, len(pre))]))
        s1["bg_drift"] = float((pre[1:, static_sel] - pre[0, static_sel]).norm(dim=-1).mean())
        if ref is not None:
            s1["origin_disp_dyn"] = float((ref[:, score_sel] - x0[score_sel]).norm(dim=-1).mean())
        res = {"_grid": {"H": H, "dt": dt, "t0": sp.t_last, "n_dynamic": int(score_sel.sum()), "n_static": int(static_sel.sum())},
               "_stage1_reference": s1}
        for name, fn in fns.items():
            traj = stack(fn, ts)
            r = {}
            disp_dyn = float((traj[:, score_sel] - x0[score_sel]).norm(dim=-1).mean())
            norm = s1.get("origin_disp_dyn", disp_dyn) if split == "inner" else disp_dyn
            if ref is not None:
                a_all, a_dyn = ade(traj, ref), ade(traj, ref, score_sel)
                r.update(ade=float(a_all.mean()), fde=float(a_all[-1]), ade_dyn=float(a_dyn.mean()), fde_dyn=float(a_dyn[-1]),
                         ade_dyn_rel=float(a_dyn.mean()) / max(norm, 1e-12),
                         ade_growth_dyn=float(np.polyfit(np.arange(1, H + 1), a_dyn.cpu().numpy(), 1)[0]))
            r["origin_disp_dyn"] = disp_dyn
            r["bg_drift"] = float((traj[:, static_sel] - x0[static_sel]).norm(dim=-1).mean())
            r["group_distortion"] = float(np.mean([group_distortion(x0, traj[h], gid, G) for h in range(H)]))
            r["accel"], r["jerk"] = smoothness(torch.cat([x0[None], traj]), score_sel, dt)
            res[name] = r
        if ref is not None:
            for name in res:
                if not name.startswith("_") and "ade_dyn" in res.get("static", {}):
                    res[name]["ade_dyn_vs_static"] = res[name]["ade_dyn"] / res["static"]["ade_dyn"]

        # PSNR growth from the scored cameras (the runs' own evaluation)
        slopes = {}
        for name, p, mode in [("jepa", jp, "rollout"), ("odegs", jo, "default")]:
            mj = os.path.join(os.path.dirname(p), "metrics.json")
            if os.path.exists(mj):
                slopes[name] = psnr_slope(json.load(open(mj))["results"][mode]["per_image"], sp.t_last)
        for key in ["static", f"cv{kc}", "deform_query"] + ([f"rigid{kr}"] if kr is not None else []):
            if key in base[split]:
                slopes[key] = psnr_slope(base[split][key]["per_image"], sp.t_last)
        for name, s in slopes.items():
            res.setdefault(name, {})["psnr_slope_per_0.1"] = s

        # JEPA composition error: direct vs composed paths
        if model is not None:
            z0, src = jepa_context(fe, model, gid, sp, dt)
            rng = np.random.default_rng(0)
            comp = {}
            for hz in sorted({max(2, H // 4), max(2, H // 2), H}):
                zd, xd = jepa_path(model, z0, src, [float(hz)], gid)
                tgt = ref[hz - 1] if ref is not None else None
                row = {"direct": {}}
                if tgt is not None:
                    row["direct"]["target_ade_dyn"] = float((xd - tgt)[score_sel].norm(dim=-1).mean())
                parts = {}
                for pname, frac in PARTITIONS.items():
                    seg = max(1.0, frac * model.cfg.max_horizon)
                    if seg < hz:
                        parts[pname] = _segments(float(hz), seg)
                parts.update({f"rand{i}": random_partition(hz, rng) for i in range(N_RANDOM)})
                for pname, hs in parts.items():
                    zc, xc = jepa_path(model, z0, src, hs, gid)
                    d = (xd - xc).norm(dim=-1)
                    e = {"pieces": [round(h, 3) for h in hs], "latent": float(latent_distance(zd, zc)),
                         "centre": float(d.mean()), "centre_dyn": float(d[score_sel].mean()),
                         "centre_dyn_rel": float(d[score_sel].mean()) / max(res["jepa"]["origin_disp_dyn"], 1e-12)}
                    if tgt is not None:
                        e["target_ade_dyn"] = float((xc - tgt)[score_sel].norm(dim=-1).mean())
                    row[pname] = e
                comp[f"h{hz}"] = row
            res["jepa_comperr"] = comp
        out[split] = res
        print(f"[level2 {split}] H={H} " + " | ".join(
            f"{k}: ade_dyn_vs_static={v.get('ade_dyn_vs_static', float('nan')):.3f} distort={v['group_distortion']:.2e} "
            f"accel={v['accel']:.2e}" for k, v in res.items() if not k.startswith("_") and k != "jepa_comperr" and "accel" in v),
            flush=True)
    os.makedirs(F, exist_ok=True)
    json.dump(out, open(os.path.join(F, "level2.json"), "w"), indent=2)
    return out


SCENES = {"hypernerf": ["slice-banana", "cut-lemon1", "chickchicken", "split-cookie", "keyboard", "vrig-3dprinter"],
          "dnerf": ["bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex"]}
LABELS = [("jepa", "4DGS-JEPA (rollout)"), ("jepa_direct", "4DGS-JEPA (direct)"), ("odegs", "ODE-GS"),
          ("deform_query", "Deformable-GS query"), ("rigid", "Group-rigid"), ("cv", "Constant velocity"), ("static", "Static")]


def summary(dataset: str) -> None:
    D = f"results/{dataset}"
    data = {s: json.load(open(f"{D}/{s}/fair/level2.json")) for s in SCENES[dataset] if os.path.exists(f"{D}/{s}/fair/level2.json")}

    def key(d, m):
        return {"cv": f"cv{d['selected_k']}", "rigid": f"rigid{d['selected_k_rigid']}"}.get(m, m)

    def table(split, cols, title, skip=(), relative=()):
        """relative: metrics divided by the scene's Stage-1 reference value before averaging over scenes."""
        L = [f"### {title}", "", "| method | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
        for m, lab in [x for x in LABELS if x[0] not in skip]:
            cells = []
            for _, f in cols:
                v = []
                for s in data:
                    x = data[s][split].get(key(data[s], m), {}).get(f)
                    if x is not None and f in relative:
                        x = x / max(data[s][split]["_stage1_reference"][f], 1e-12)
                    v.append(x)
                v = [x for x in v if x is not None and np.isfinite(x)]
                cells.append(f"{np.mean(v):.3g}" if len(v) == len(data) and v else "–")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
        return L + [""]

    L = [f"# Level II diagnostics, {dataset} ({len(data)} scenes: {', '.join(data)})", "",
         "Means over scenes. Generated by `python -m gsjepa.real.consistency summary`; definitions in the module docstring.", ""]
    L += table("inner", [("ADE dyn / Static ↓", "ade_dyn_vs_static"), ("ADE dyn (rel.) ↓", "ade_dyn_rel"),
                         ("ADE growth / frame ↓", "ade_growth_dyn"), ("PSNR slope / 0.1 t ↑", "psnr_slope_per_0.1")],
               "Inner split: target error vs Stage-1 states (held out for every method; Deformable-GS query is the "
               "reference itself here and is left out)", skip=("deform_query",))
    rel = ("bg_drift", "group_distortion", "accel", "jerk")
    for split in ["inner", "full"]:
        L += table(split, [("bg drift ↓", "bg_drift"), ("group distortion", "group_distortion"), ("accel", "accel"),
                           ("jerk", "jerk")] + ([("PSNR slope / 0.1 t ↑", "psnr_slope_per_0.1")] if split == "full" else []),
                   f"{split} split: forecast consistency, as ratio to Stage 1 over the last H prefix frames "
                   "(1 = as smooth / rigid / drifting as the observed motion; Static and CV are 0 by construction)",
                   relative=rel)
    for split in ["inner", "full"]:
        L += [f"### JEPA composition error, {split} split (direct jump vs composed partitions; medians over the "
              "non-diverged cases; diverged = non-finite or composed centres off by > 10× the forecast motion)", "",
              "| horizon | partition | diverged | latent dist ↓ | centre dist dyn (rel.) ↓ | target ADE dyn: direct | composed |",
              "|---|---|---|---|---|---|---|"]
        for hk in ["short", "mid", "long"]:
            for p in list(PARTITIONS) + ["rand"]:
                acc = {"latent": [], "centre_dyn_rel": [], "td": [], "tc": []}
                for s in data:
                    comp = data[s][split].get("jepa_comperr")
                    if not comp:
                        continue
                    hs = sorted(comp, key=lambda h: int(h[1:]))
                    h = {"short": hs[0], "mid": hs[len(hs) // 2], "long": hs[-1]}[hk]
                    es = [e for n, e in comp[h].items() if n != "direct" and (n.startswith("rand") if p == "rand" else n == p)]
                    for e in es:
                        acc["n"] = acc.get("n", 0) + 1
                        if not (np.isfinite(e["latent"]) and np.isfinite(e["centre_dyn_rel"]) and e["centre_dyn_rel"] <= 10):
                            acc["div"] = acc.get("div", 0) + 1
                            continue
                        acc["latent"].append(e["latent"])
                        acc["centre_dyn_rel"].append(e["centre_dyn_rel"])
                        if "target_ade_dyn" in e:
                            acc["tc"].append(e["target_ade_dyn"])
                            acc["td"].append(comp[h]["direct"]["target_ade_dyn"])
                if acc.get("n"):
                    f = lambda x: f"{np.median(x):.3g}" if x else "–"
                    L.append(f"| {hk} | {p} | {acc.get('div', 0)}/{acc['n']} | {f(acc['latent'])} | {f(acc['centre_dyn_rel'])} | "
                             f"{f(acc['td'])} | {f(acc['tc'])} |")
        L.append("")
    open(f"{D}/level2_summary.md", "w").write("\n".join(L) + "\n")
    json.dump(data, open(f"{D}/level2_summary.json", "w"), indent=2)
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scene", "summary"])
    ap.add_argument("--source")
    ap.add_argument("--root")
    ap.add_argument("--time_split", type=float, default=0.9)
    ap.add_argument("--inner_cut", type=float, default=0.8)
    ap.add_argument("--dataset", default="hypernerf")
    args = ap.parse_args()
    if args.cmd == "scene":
        scene(args)
    else:
        summary(args.dataset)


if __name__ == "__main__":
    main()
