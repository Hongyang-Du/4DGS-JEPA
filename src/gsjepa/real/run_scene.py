"""Protocol-B temporal extrapolation on one D-NeRF scene with a trained Stage-1 front-end.

    python -m gsjepa.real.run_scene --scene mutant --data /data/D-NeRF \
        --stage1 results/dnerf/mutant/stage1 --out results/dnerf/mutant

Steps: extract prefix trajectories from Deformable-GS -> k-means motion groups -> train the
scaled 4DGS-JEPA on (context, target) windows inside the prefix -> roll out from the last
prefix frame to every evaluation camera time -> render and score (PSNR / SSIM / LPIPS-vgg),
together with the baselines that share the same front-end (Deformable-GS queried at t > 0.8,
static, constant velocity).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch

from .frontend import Evaluator, FrontEnd
from .jepa import (RealConfig, RealJEPA, Stats, geometry_loss, latent_distance, variance_loss)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kmeans_groups(xyz: torch.Tensor, G: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """Motion groups from prefix trajectories xyz [T, N, 3] (every 4th frame)."""
    feats = xyz[::4].permute(1, 0, 2).reshape(xyz.shape[1], -1)
    feats = (feats - feats.mean(0)) / (feats.std() + 1e-8)
    g = torch.Generator(device=feats.device).manual_seed(seed)
    cent = feats[torch.randperm(len(feats), generator=g, device=feats.device)[:G]].clone()
    for _ in range(iters):
        gid = torch.cdist(feats, cent).argmin(1)
        for k in range(G):
            m = gid == k
            if m.any():
                cent[k] = feats[m].mean(0)
    return torch.cdist(feats, cent).argmin(1)


def warmup_cosine(opt, steps: int, warmup_frac: float, min_ratio: float):
    """Linear warmup to the base LR, then cosine decay to min_ratio * base LR (per optimizer step)."""
    warm = max(1, int(warmup_frac * steps))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = min(1.0, (s - warm) / max(1, steps - warm))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def window(traj, idx, sub):
    """Gather frames idx [B, L] (or [B]) of Gaussians sub from traj [T, N, C] -> [B, (L,) n, C]."""
    return {k: v[:, sub][idx] for k, v in traj.items()}


def train(cfg: RealConfig, traj, u, gid, stats: Stats, log_path: str, callback=None) -> RealJEPA:
    """callback(steps_done, model), if given, is called before training and after every optimizer step
    (it decides itself whether to act, e.g. evaluate at fixed fractions of the budget)."""
    device = traj["xyz"].device
    T, N, _ = traj["xyz"].shape
    L, K, S = cfg.context_len, cfg.targets_per_context, cfg.context_stride
    offs = S * torch.arange(-L + 1, 1, device=device)[None]
    model = RealJEPA(cfg, stats).to(device)
    teacher = copy.deepcopy(model.encoder).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.schedule == "cosine":
        sched = warmup_cosine(opt, cfg.steps, cfg.warmup_frac, cfg.min_lr_ratio)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.05)
    log = open(log_path, "w")
    if callback is not None:
        callback(0, model)
    t0 = time.time()
    for step in range(cfg.steps):
        model.train()
        sub = torch.randperm(N, device=device)[:cfg.gaussians_per_step]
        g = gid[sub]
        B = cfg.batch_contexts
        t = torch.randint(S * (L - 1), T - 1, (B,), device=device)
        kmax = torch.clamp(T - 1 - t, max=cfg.max_horizon)
        # K sorted target offsets per context (endpoint = largest, the rest are path points)
        ks = torch.sort(torch.ceil(torch.rand(B, K, device=device) * kmax[:, None]).long().clamp_min(1), dim=1).values
        hist = t[:, None] + offs
        ctx = window(traj, hist, sub)
        z0 = model.encoder(ctx["xyz"], ctx["rot"], ctx["scale"], u[hist], g)
        src = model.source(ctx, u[hist])

        loss_end = loss_geo = 0.0
        z_direct = []
        for j in range(K):
            tgt_idx = t + ks[:, j]
            h = u[tgt_idx] - u[t]
            z1 = model.transition(z0, h, g)
            z_direct.append((z1, h, tgt_idx))
            with torch.no_grad():
                th = tgt_idx[:, None] + offs
                tw = window(traj, th, sub)
                zt = teacher(tw["xyz"], tw["rot"], tw["scale"], u[th], g)
            loss_end = loss_end + latent_distance(z1, zt) / K
            gt = {k: v[:, -1] for k, v in tw.items()}
            loss_geo = loss_geo + geometry_loss(model.decoder(z0, z1, src, h, g), gt, stats) / K

        # composition on the endpoint target: Phi(Phi(z, h_a), h - h_a) vs Phi(z, h)
        z_end, h_end, tgt_end = z_direct[-1]
        frac = torch.rand(B, device=device) * 0.8 + 0.1
        h_a = h_end * frac
        z_a = model.transition(z0, h_a, g)
        z_ab = model.transition(z_a, h_end - h_a, g)
        loss_comp = 0.5 * (latent_distance(z_ab, tuple(x.detach() for x in z_end))
                           + latent_distance(tuple(x.detach() for x in z_ab), z_end))
        geo_a = model.decoder(z0, z_a, src, h_a, g)
        geo_ab = model.decoder(z_a, z_ab, geo_a, h_end - h_a, g)
        gt_end = window(traj, tgt_end, sub)
        loss_geo_roll = geometry_loss(geo_ab, gt_end, stats)

        loss = (cfg.w_end * loss_end + cfg.w_comp * loss_comp + cfg.w_geo * loss_geo
                + cfg.w_geo_rollout * loss_geo_roll + cfg.w_var * variance_loss(z0[0]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pt, ps in zip(teacher.parameters(), model.encoder.parameters()):
                pt.mul_(cfg.ema_tau).add_(ps, alpha=1 - cfg.ema_tau)
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            msg = (f"step {step:6d} loss {loss.item():.4f} end {float(loss_end):.4f} comp {loss_comp.item():.4f} "
                   f"geo {float(loss_geo):.4f} geo_roll {loss_geo_roll.item():.4f} ({time.time() - t0:.0f}s)")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if callback is not None:
            callback(step + 1, model)
    log.close()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default="/data/D-NeRF")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--save_images", action="store_true")
    args = ap.parse_args()
    cfg = RealConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)

    fe = FrontEnd(os.path.join(args.data, args.scene), args.stage1)
    fids = fe.train_fids
    dt = float(np.median(np.diff(fids)))
    print(f"{fe.num_gaussians} Gaussians | {len(fe.train_cams)} prefix / {len(fe.eval_cams)} eval cameras | "
          f"{len(fids)} prefix times, dt={dt:.5f}, last prefix t={fids[-1]:.4f}", flush=True)

    traj = {k: v.cuda() for k, v in fe.trajectory(fids).items()}
    u = torch.tensor(fids / dt, dtype=torch.float32, device="cuda")
    stats = Stats.from_prefix(traj["xyz"], traj["scale"], u)
    gid = kmeans_groups(traj["xyz"], cfg.num_groups, seed=cfg.seed)
    print(f"stats: pos_scale={stats.pos_scale:.4f} vel_scale={stats.vel_scale:.5f} vmax={stats.vmax:.5f}; "
          f"group sizes min/max {torch.bincount(gid).min().item()}/{torch.bincount(gid).max().item()}", flush=True)

    model = train(cfg, traj, u, gid, stats, os.path.join(args.out, "train_log.txt"))
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "gid": gid.cpu(), "dt": dt, "stats": stats}, os.path.join(args.out, "jepa.pt"))

    # ---------------------------------------------------------------- evaluation (protocol B)
    model.eval()
    L, T = cfg.context_len, len(fids)
    last = {k: v[-1] for k, v in traj.items()}
    allg = torch.arange(fe.num_gaussians, device="cuda")
    hist = torch.arange(T - L, T, device="cuda")[None]
    with torch.no_grad():
        ctx = window(traj, hist, allg)
        z0 = model.encoder(ctx["xyz"], ctx["rot"], ctx["scale"], u[hist], gid)
        src = model.source(ctx, u[hist])

    cache = {}

    def jepa_fn(segment):
        def fn(fid):
            key = (segment, fid)
            if key not in cache:
                with torch.no_grad():
                    geo = model.predict(z0, src, (fid - fids[-1]) / dt, gid, segment)
                cache[key] = {k: v[0] for k, v in geo.items()}
            return cache[key]
        return fn

    def const_vel(k):
        v = (traj["xyz"][-1] - traj["xyz"][-1 - k]) / (u[-1] - u[-1 - k])
        return lambda fid: {**last, "xyz": last["xyz"] + v * ((fid - fids[-1]) / dt)}

    methods = {
        "Deformable-GS (query t>0.8)": fe.state_at,
        "Static (last prefix frame)": lambda fid: last,
        "Constant velocity (1 frame)": const_vel(1),
        "Constant velocity (4 frames)": const_vel(4),
        "4DGS-JEPA direct": jepa_fn(None),
        f"4DGS-JEPA rollout (seg {cfg.rollout_segment:g})": jepa_fn(cfg.rollout_segment),
    }
    ev = Evaluator(fe)
    results = {}
    for name, fn in methods.items():
        save = os.path.join(args.out, "images", name.split(" ")[0]) if args.save_images else None
        results[name] = ev.evaluate(fe.eval_cams, fn, save)
        r = results[name]
        print(f"{name:34s} PSNR {r['psnr']:.2f} (img {r['psnr_img']:.2f})  SSIM {r['ssim']:.4f}  LPIPS {r['lpips']:.4f}", flush=True)
    sanity = ev.evaluate(fe.train_cams[::8], fe.state_at)
    print(f"[sanity] Deformable-GS on prefix cameras: PSNR {sanity['psnr']:.2f} SSIM {sanity['ssim']:.4f}", flush=True)
    results["_sanity_prefix_deform"] = sanity
    results["_meta"] = {"scene": args.scene, "num_gaussians": fe.num_gaussians, "dt": dt,
                        "prefix_times": len(fids), "eval_cameras": len(fe.eval_cams),
                        "last_prefix_fid": float(fids[-1]), "cfg": asdict(cfg)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
