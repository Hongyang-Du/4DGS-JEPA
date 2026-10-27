"""Apple-to-apple protocol-B benchmark on one scene (shared Stage 1, same budget / tuning / selection rules).

Two splits, both inside the observed prefix or on the protocol-B test cameras:
  inner  stage 2 sees Stage-1 states for t <= --inner_cut only; scored on the prefix cameras with
         inner_cut < t <= last prefix time (GT images exist, no t > split data involved) -> used for tuning
  full   stage 2 sees the whole prefix; scored ONCE on the t > split evaluation cameras

    python -m gsjepa.real.fair jepa  --split inner --cfg '{"lr": 3e-4}' --steps 12000 --out <dir>
    python -m gsjepa.real.fair odegs --split full  --cfg '{"lr": 1e-3}' --steps 60000 --seed 1 --out <dir>
    python -m gsjepa.real.fair baselines --out <dir>
    python -m gsjepa.real.fair masks                                    # dynamic-region masks (once per scene)
    python -m gsjepa.real.fair bench --method odegs --cfg '{...}'      # throughput (it/s) for budget calibration
  D-NeRF mutant is the default; HyperNeRF e.g.
    --source /data/HyperNeRF/interp/slice-banana --root results/hypernerf/slice-banana --time_split 0.9 --inner_cut 0.8

Every learned run uses AdamW + linear warmup + cosine decay, gradient clipping 1.0, a fixed optimizer-step
budget and the LAST checkpoint (no selection on any evaluation camera). During training both the inner
validation cameras (forecast from the inner-cut context) and the test cameras (forecast from the end of the
prefix) are scored (PSNR) at every 5% of the budget -> curve.csv; the test curve is for display only.

Dynamic-region mask (identical for all methods, computed once per scene from GT + the Stage-1 static state):
pixels where max_c |GT_t - R_t0| > --mask_tau after opening, dilated by --mask_dilate px (see compute_masks).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from .frontend import Evaluator, FrontEnd, mask_key
from .jepa import RealConfig, Stats, axis_angle_to_quat, quat_mul, quat_rotate
from .run_scene import kmeans_groups, set_seed, train as train_jepa, warmup_cosine, window

ODEGS_DEFAULT = {  # ODE-GS official D-NeRF config (configs/config_DNerf_default.yaml) unless overridden
    "d_model": 128, "latent_dim": 64, "nhead": 8, "num_encoder_layers": 4, "num_decoder_layers": 4,
    "ode_nhidden": 64, "decoder_nhidden": 128, "ode_layers": 4, "rtol": 1e-3, "atol": 1e-4, "use_tanh": True,
    "obs_time_span": 0.6, "obs_points": 30, "extrap_points": 10, "max_extrap_time_span": 0.2,
    "num_time_windows": 50, "batch_size": 1024, "reg_weight": 1e-5, "xyz_reg_weight": 1e-3,
    "reg_temperature": 0.05, "expected_traj_loss_init": 0.05, "expected_traj_loss_end": 0.0,
    "reg_ema_decay": 0.9, "lr": 1e-3, "weight_decay": 1e-4, "warmup_frac": 0.05, "min_lr_ratio": 0.01,
}
ODEGS_HYPERNERF = {  # configs/config_HyperNeRF_default.yaml data / regulariser settings
    "obs_time_span": 0.4, "obs_points": 30, "extrap_points": 10, "max_extrap_time_span": 0.1, "batch_size": 512,
    "xyz_reg_weight": 0.0, "reg_temperature": 0.15, "expected_traj_loss_init": 0.02,
}
CURVE_POINTS = 21  # 0%, 5%, ..., 100% of the step budget


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class Split:
    """Training times and scoring cameras for the inner (tuning) or full (final) split."""

    def __init__(self, fe: FrontEnd, split: str, inner_cut: float):
        prefix = fe.train_fids
        t_eval0 = min(c.fid.item() for c in fe.eval_cams)
        if prefix.max() >= t_eval0:  # multi-camera rigs (vrig): the camera-count split can cut inside one time
            print(f"[split] prefix time {prefix.max():.4f} also has evaluation cameras -> dropped from stage-2 data", flush=True)
            prefix = prefix[prefix < t_eval0]
        if split == "inner":
            self.fids = prefix[prefix <= inner_cut]
            self.cams = [c for c in fe.train_cams if self.fids[-1] < c.fid.item() <= prefix[-1]]
        else:
            self.fids = prefix
            self.cams = fe.eval_cams
        self.name, self.t_last = split, float(self.fids[-1])
        self.dt = float(np.median(np.diff(prefix)))  # same time unit for both splits
        print(f"[split {split}] train times {len(self.fids)} (t <= {self.t_last:.4f}) | scored cameras "
              f"{len(self.cams)} (t in [{self.cams[0].fid.item():.4f}, {self.cams[-1].fid.item():.4f}])", flush=True)


def summarize(res: dict) -> str:
    s = f"PSNR {res['psnr']:.2f} (img {res['psnr_img']:.2f})  SSIM {res['ssim']:.4f}  LPIPS {res['lpips']:.4f}"
    if "dyn_psnr" in res:
        s += f" | dyn PSNR {res['dyn_psnr']:.2f} SSIM {res['dyn_ssim']:.4f} LPIPS {res['dyn_lpips']:.4f}"
    return s


def per_time(res: dict, key: str = "psnr") -> dict:
    out = {}
    for r in res["per_image"]:
        out.setdefault(round(r["fid"], 4), []).append(r[key])
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


# ================================================================== dynamic-region masks
def rigid_fit(a: torch.Tensor, b: torch.Tensor):
    """Least-squares rigid transform (R, t) with a @ R.T + t ~ b (Kabsch)."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = torch.linalg.svd((a - ca).T @ (b - cb))
    D = torch.diag(torch.tensor([1.0, 1.0, float(torch.sign(torch.det(Vt.T @ U.T)))], device=a.device))
    R = Vt.T @ D @ U.T
    return R, cb - ca @ R.T


def dynamic_gaussians(fe: FrontEnd, sp: Split, window: float, quantile: float) -> torch.Tensor:
    """Prefix-only motion score: for every prefix time in the last `window` of sp's training range, residual
    displacement of each Gaussian w.r.t. the state at sp.t_last after removing the best global rigid motion
    (Stage 1 also explains camera/pose drift as scene motion); max over time. Top (1 - quantile) -> dynamic."""
    fs = sp.fids[sp.fids >= sp.t_last - window]
    xyz = fe.trajectory(fs)["xyz"].cuda()
    ref = xyz[-1]
    score = torch.zeros(len(ref), device="cuda")
    for x in xyz:
        R, t = rigid_fit(x, ref)
        score = torch.maximum(score, (x @ R.T + t - ref).norm(dim=-1))
    thr = torch.quantile(score[torch.randperm(len(score), device="cuda")[:500000]], quantile)
    return score > thr


def compute_masks(fe: FrontEnd, ev: Evaluator, inner_cut: float, mode: str, tau: float, window: float,
                  quantile: float, dilate: int, path: str) -> dict:
    """Dynamic-region mask of each scored camera (time t, forecast origin t0 = last training time of its split).

    mode "gt" (default): pixels where max_c |GT_t - R_t0| > tau, R_t0 = Stage-1 render of the static state at t0
      from the same camera; morphological opening (radius 2) removes isolated background reconstruction errors,
      then dilation by `dilate` px. By construction this is where the static forecast is wrong.
    mode "gauss": visible footprint at t0 of the Gaussians with the largest prefix-only motion
      (dynamic_gaussians), thresholded at 0.5 coverage and dilated. On real scenes Stage 1 under-models fast
      hand motion, so this misses hands; kept for reference."""
    from utils.sh_utils import RGB2SH
    from gaussian_renderer import render_ode
    out, vis = {}, []
    feats = fe.gaussians.get_features
    for name in ["inner", "full"]:
        sp = Split(fe, name, inner_cut)
        ref = fe.state_at(sp.t_last)
        if mode == "gauss":
            dyn = dynamic_gaussians(fe, sp, window, quantile)
            f = torch.zeros_like(feats)
            f[:, 0] = RGB2SH(torch.where(dyn[:, None], 1.0, 0.0).expand(-1, 3))
        for cam in sp.cams:
            cam.load2device("cuda")
            with torch.no_grad():
                if mode == "gauss":
                    cov = render_ode(cam, fe.gaussians, fe.pipe, fe.background, ref["xyz"], ref["rot"], ref["scale"],
                                     new_features=f)["render"][0]
                    m = (cov > 0.5).float()[None, None]
                else:
                    d = (cam.original_image.cuda() - ev.render(cam, ref)).abs().amax(0)
                    m = (d > tau).float()[None, None]
                    m = -F.max_pool2d(-m, 5, stride=1, padding=2)
                    m = F.max_pool2d(m, 5, stride=1, padding=2)
            if dilate > 0:
                m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
            out[mask_key(cam)] = m[0, 0].bool().cpu()
            cam.load2device("cpu")
        fr = [out[mask_key(c)].float().mean().item() for c in sp.cams]
        print(f"[masks {name}] mode {mode}; pixel fraction mean {np.mean(fr):.3f} "
              f"(min {np.min(fr):.3f}, max {np.max(fr):.3f})", flush=True)
        vis += [sp.cams[0], sp.cams[-1]]
    torch.save({"masks": out, "mode": mode, "tau": tau, "window": window, "quantile": quantile, "dilate": dilate,
                "inner_cut": inner_cut}, path)
    import imageio.v2 as imageio
    for cam in vis:  # GT with the mask overlaid in red, for the report
        gt = cam.original_image.clone()
        m = out[mask_key(cam)]
        gt[0][m] = 0.5 * gt[0][m] + 0.5
        imageio.imwrite(path.replace(".pt", f"_{cam.image_name}.png"),
                        (gt.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    return out


def load_masks(root: str):
    p = os.path.join(root, "fair", "masks.pt")
    return torch.load(p)["masks"] if os.path.exists(p) else None


# ================================================================== training curves
class Curve:
    """Score val (inner cameras, forecast from the inner cut) and test (t > split, forecast from the prefix end)
    at every 5% of the step budget; time spent here is excluded from the reported training time."""

    def __init__(self, fe, ev, inner_cut, steps, make_fn, masks, path):
        self.sps = {"val": Split(fe, "inner", inner_cut), "test": Split(fe, "full", inner_cut)}
        self.at = set(np.linspace(0, steps, CURVE_POINTS).round().astype(int).tolist())
        self.fe, self.ev, self.make_fn, self.masks, self.path = fe, ev, make_fn, masks, path
        self.rows, self.eval_sec, self.t0 = [], 0.0, None

    def __call__(self, step, model):
        if self.t0 is None:
            torch.cuda.synchronize()
            self.t0 = time.time()
        if step not in self.at:
            return
        torch.cuda.synchronize()
        t = time.time()
        model.eval()
        row = {"step": step, "train_sec": t - self.t0 - self.eval_sec}
        for k, sp in self.sps.items():
            r = self.ev.evaluate(sp.cams, self.make_fn(model, sp), masks=self.masks, fast=True)
            row[f"{k}_psnr"] = r["psnr"]
            if "dyn_psnr" in r:
                row[f"{k}_dyn_psnr"] = r["dyn_psnr"]
        model.train()
        torch.cuda.synchronize()
        self.eval_sec += time.time() - t
        self.rows.append(row)
        print("[curve] " + " ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)


# ================================================================== 4DGS-JEPA
def jepa_state_fn(fe, model, gid, sp: Split, segment, dt):
    """Roll out the JEPA from the last context_len Stage-1 frames of sp's training range."""
    L, S = model.cfg.context_len, model.cfg.context_stride
    fids = sp.fids[len(sp.fids) - 1 - S * (L - 1)::S]
    assert len(fids) == L and fids[-1] == sp.fids[-1]
    traj = {k: v.cuda() for k, v in fe.trajectory(fids).items()}
    u = torch.tensor(fids / dt, dtype=torch.float32, device="cuda")[None]
    with torch.no_grad():
        ctx = {k: v[None] for k, v in traj.items()}
        z0 = model.encoder(ctx["xyz"], ctx["rot"], ctx["scale"], u, gid)
        src = model.source(ctx, u)
    cache = {}

    def f(fid):
        if fid not in cache:
            with torch.no_grad():
                g = model.predict(z0, src, (fid - sp.t_last) / dt, gid, segment)
            cache[fid] = {k: v[0] for k, v in g.items() if k != "vel"}
        return cache[fid]
    return f


def run_jepa(fe, sp, cfg: RealConfig, out, args, masks) -> dict:
    set_seed(cfg.seed)
    traj = {k: v.cuda() for k, v in fe.trajectory(sp.fids).items()}
    u = torch.tensor(sp.fids / sp.dt, dtype=torch.float32, device="cuda")
    stats = Stats.from_prefix(traj["xyz"], traj["scale"], u)
    gid = kmeans_groups(traj["xyz"], cfg.num_groups, seed=cfg.seed)
    ev = Evaluator(fe)
    curve = Curve(fe, ev, args.inner_cut, cfg.steps,
                  lambda m, s: jepa_state_fn(fe, m, gid, s, cfg.rollout_segment, sp.dt), masks, os.path.join(out, "curve.csv"))
    torch.cuda.synchronize()
    t0 = time.time()
    model = train_jepa(cfg, traj, u, gid, stats, os.path.join(out, "train_log.txt"), callback=curve)
    torch.cuda.synchronize()
    train_sec = time.time() - t0 - curve.eval_sec
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "gid": gid.cpu(), "dt": sp.dt, "stats": stats,
                "split": sp.name, "t_last": sp.t_last}, os.path.join(out, "last.pt"))
    del traj
    model.eval()
    res = {}
    for mode, seg in [("rollout", cfg.rollout_segment), ("direct", None)]:
        res[mode] = ev.evaluate(sp.cams, jepa_state_fn(fe, model, gid, sp, seg, sp.dt),
                                os.path.join(out, "images", mode) if args.save_images else None, masks=masks)
        print(f"4DGS-JEPA {mode:8s} [{sp.name}] {summarize(res[mode])}", flush=True)
    return {"method": "jepa", "params": n_params(model), "train_sec": train_sec, "steps": cfg.steps,
            "samples_per_step": cfg.batch_contexts * cfg.gaussians_per_step,
            "observation": f"context_len={cfg.context_len} prefix frames", "cfg": asdict(cfg), "results": res,
            "curve": curve.rows}


# ================================================================== ODE-GS (official model + loss, our budget/optimizer)
def odegs_windows(fe: FrontEnd, fids: np.ndarray, c: dict):
    """GPU-resident copy of ODE-GS DynamicLengthDataset: num_time_windows observation windows of span
    obs_time_span starting in [min_t, max_t - span], each followed by extrap_points targets up to
    min(max_t, end + max_extrap_time_span); all Gaussians, states (xyz, rot, scale) from Stage 1."""
    min_t, max_t = float(fids[0]), float(fids[-1])
    starts = torch.linspace(min_t, max_t - c["obs_time_span"] - 1e-6, steps=c["num_time_windows"])
    obs, tgt, times = [], [], []
    st = lambda ts: torch.stack([torch.cat(list(fe.state_at(float(t)).values()), -1) for t in ts], 1)
    for s in starts.tolist():
        e = s + c["obs_time_span"]
        ot = torch.linspace(s, e, steps=c["obs_points"], device="cuda")
        et = torch.linspace(e, min(max_t, e + c["max_extrap_time_span"]), steps=c["extrap_points"] + 1, device="cuda")[1:]
        obs.append(st(ot).cpu())
        tgt.append(st(et).cpu())
        times.append(torch.cat([ot, et]))
    # windows live in host memory (W x N x 40 x 10 floats exceeds GPU memory on large scenes); batches are
    # gathered on the CPU and copied to the GPU each step
    return torch.stack(obs), torch.stack(tgt), torch.stack(times)  # [W,N,To,10], [W,N,Te,10] (cpu), [W,To+Te] (cuda)


def build_odegs(c: dict, obs_dim: int = 10):
    from scene.extrapolation_ode_model import TransformerLatentODEWrapper
    return TransformerLatentODEWrapper(
        latent_dim=c["latent_dim"], d_model=c["d_model"], nhead=c["nhead"], num_encoder_layers=c["num_encoder_layers"],
        num_decoder_layers=c["num_decoder_layers"], ode_nhidden=c["ode_nhidden"], decoder_nhidden=c["decoder_nhidden"],
        obs_dim=obs_dim, noise_std=1.0, ode_layers=c["ode_layers"], reg_weight=c["reg_weight"], kl_beta=1.0,
        variational_inference=False, use_torchode=True, rtol=c["rtol"], atol=c["atol"], use_tanh=c["use_tanh"],
        xyz_reg_weight=c["xyz_reg_weight"]).cuda()


def train_odegs(fe: FrontEnd, sp: Split, c: dict, steps: int, seed: int, log_path: str, callback=None):
    from train_extrapolation import _compute_adaptive_regularizer_scale
    set_seed(seed)
    obs, tgt, times = odegs_windows(fe, sp.fids, c)
    W, N, To, D = obs.shape
    Te = tgt.shape[2]
    # one contiguous row (To + Te states) per (window, Gaussian): a single index_select into a pinned buffer per step
    rows = torch.cat([obs.reshape(W * N, To * D), tgt.reshape(W * N, Te * D)], 1)
    del obs, tgt
    buf = torch.empty(c["batch_size"], rows.shape[1], pin_memory=True)
    model = build_odegs(c)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = warmup_cosine(opt, steps, c["warmup_frac"], c["min_lr_ratio"])
    log = open(log_path, "w")
    if callback is not None:
        callback(0, model)
    torch.cuda.synchronize()
    ema, t0 = None, time.time()
    for step in range(steps):
        model.train()
        w = torch.randint(0, W, (c["batch_size"],))
        g = torch.randint(0, N, (c["batch_size"],))
        torch.index_select(rows, 0, w * N + g, out=buf)
        b = buf.cuda()  # synchronous: buf is reused next step
        o, y, f = b[:, :To * D].view(-1, To, D), b[:, To * D:].view(-1, Te, D), times[w.cuda()]
        pred_x, pred_z, ut, m, lv = model(o, y, f)
        loss, _, pred_loss, _, reg, xreg = model.compute_loss(pred_x, y, o, pred_z, ut, o.shape[1], m, lv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        # ODE-GS adaptive regulariser scaling (EMA of the trajectory loss), as in train_extrapolation.py
        pl = float(pred_loss.detach())
        ema = pl if ema is None else c["reg_ema_decay"] * ema + (1 - c["reg_ema_decay"]) * pl
        s = _compute_adaptive_regularizer_scale(ema, c["reg_temperature"], c["expected_traj_loss_init"],
                                                c["expected_traj_loss_end"])
        model.reg_weight, model.xyz_reg_weight = c["reg_weight"] * s, c["xyz_reg_weight"] * s
        if step % 500 == 0 or step == steps - 1:
            msg = (f"step {step:6d} loss {loss.item():.5f} pred {pl:.5f} reg {reg.item():.2e} xyz_reg {xreg.item():.2e} "
                   f"lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if callback is not None:
            callback(step + 1, model)
    log.close()
    torch.cuda.synchronize()
    loop_sec = time.time() - t0 - (callback.eval_sec if callback is not None else 0.0)
    del rows
    torch.cuda.empty_cache()
    return model, loop_sec


def odegs_predictor(fe: FrontEnd, model, c: dict, t_last: float, targets):
    from evaluate_extrapolation import predict_trajectories_at_times
    model.eval()
    obs_t = torch.linspace(t_last - c["obs_time_span"], t_last, steps=c["obs_points"], device="cuda")
    targets = sorted(set(float(t) for t in targets))
    with torch.no_grad():
        pred = predict_trajectories_at_times(model, fe.deform, fe.gaussians, obs_t,
                                             torch.tensor(targets, device="cuda"), batch_size=20000)
    lookup = {round(k, 6): v for k, v in pred.items()}

    def fn(f):
        p = lookup[round(float(torch.tensor(f, dtype=torch.float32, device="cuda").item()), 6)]
        return {"xyz": p[:, :3], "rot": p[:, 3:7], "scale": p[:, 7:10]}
    return fn


def run_odegs(fe, sp, c, steps, seed, out, args, masks) -> dict:
    ev = Evaluator(fe)
    curve = Curve(fe, ev, args.inner_cut, steps,
                  lambda m, s: odegs_predictor(fe, m, c, s.t_last, [cam.fid.item() for cam in s.cams]),
                  masks, os.path.join(out, "curve.csv"))
    model, train_sec = train_odegs(fe, sp, c, steps, seed, os.path.join(out, "train_log.txt"), callback=curve)
    torch.save({"model_state_dict": model.state_dict(), "cfg": c, "seed": seed, "steps": steps, "split": sp.name,
                "t_last": sp.t_last}, os.path.join(out, "last.pt"))
    fn = odegs_predictor(fe, model, c, sp.t_last, [cam.fid.item() for cam in sp.cams])
    res = {"default": ev.evaluate(sp.cams, fn, os.path.join(out, "images") if args.save_images else None, masks=masks)}
    print(f"ODE-GS [{sp.name}] {summarize(res['default'])}", flush=True)
    return {"method": "odegs", "params": n_params(model), "train_sec": train_sec, "steps": steps,
            "samples_per_step": c["batch_size"],
            "observation": f"last {c['obs_time_span']} of time, {c['obs_points']} points", "cfg": c, "results": res,
            "curve": curve.rows}


# ================================================================== analytic baselines
def rot_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [..., 3, 3] -> axis-angle vectors [..., 3] (angle < pi)."""
    th = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    f = torch.where(th < 1e-4, torch.full_like(th, 0.5), th / (2 * th.sin().clamp_min(1e-8)))
    return v * f[..., None]


def group_rigid_motion(x0: torch.Tensor, x1: torch.Tensor, gid: torch.Tensor, G: int):
    """Per-group rigid motion x0 -> x1 about the group centroids: (axis-angle w [G,3], c0 [G,3], c1 [G,3])."""
    w, c0, c1 = x0.new_zeros(G, 3), x0.new_zeros(G, 3), x0.new_zeros(G, 3)
    for g in range(G):
        m = gid == g
        if m.sum() >= 3:
            R, _ = rigid_fit(x0[m], x1[m])
            w[g], c0[g], c1[g] = rot_to_axis_angle(R), x0[m].mean(0), x1[m].mean(0)
    return w, c0, c1


def group_rigid_state(last: dict, motion, gid: torch.Tensor, s: float) -> dict:
    """Continue each group's rigid motion (constant angular + linear velocity about its centroid) for s times the
    fitted interval; Gaussians keep their scale, rotations are composed with the group rotation."""
    w, c0, c1 = motion
    q = axis_angle_to_quat(w * s)[gid]
    xyz = quat_rotate(q, last["xyz"] - c1[gid]) + c1[gid] + (c1 - c0)[gid] * s
    return {**last, "xyz": xyz, "rot": quat_mul(q, last["rot"])}


def run_baselines(fe: FrontEnd, inner_cut: float, masks, ks=(1, 2, 4, 8, 16), num_groups: int = 32) -> dict:
    """CV and group-rigid windows k chosen on the inner split; static has no free parameter.
    Group-rigid uses the JEPA's motion groups (k-means on the split's prefix trajectories, same G and seed)."""
    ev = Evaluator(fe)
    out = {}
    for name in ["inner", "full"]:
        sp = Split(fe, name, inner_cut)
        n = min(17, len(sp.fids))
        traj = {k: v.cuda() for k, v in fe.trajectory(sp.fids[-n:]).items()}
        u = torch.tensor(sp.fids[-n:] / sp.dt, dtype=torch.float32, device="cuda")
        last = {k: v[-1] for k, v in traj.items()}
        res = {"static": ev.evaluate(sp.cams, lambda f: last, masks=masks)}
        for k in [k for k in ks if k < n]:
            v = (traj["xyz"][-1] - traj["xyz"][-1 - k]) / (u[-1] - u[-1 - k])
            res[f"cv{k}"] = ev.evaluate(sp.cams, lambda f, v=v: {**last, "xyz": last["xyz"] + v * ((f - sp.t_last) / sp.dt)},
                                        masks=masks)
        gid = kmeans_groups(fe.trajectory(sp.fids)["xyz"].cuda(), num_groups, seed=0)
        for k in [k for k in ks if k < n]:
            mo = group_rigid_motion(traj["xyz"][-1 - k], traj["xyz"][-1], gid, num_groups)
            span = float(u[-1] - u[-1 - k])
            res[f"rigid{k}"] = ev.evaluate(sp.cams, lambda f, mo=mo, span=span: group_rigid_state(
                last, mo, gid, (f - sp.t_last) / sp.dt / span), masks=masks)
        res["deform_query"] = ev.evaluate(sp.cams, fe.state_at, masks=masks)
        for m, r in res.items():
            print(f"[{name}] {m:12s} {summarize(r)}", flush=True)
        out[name] = res
    for pre, key, label in [("cv", "selected_k", "CV"), ("rigid", "selected_k_rigid", "Group-rigid")]:
        cand = [k for k in ks if f"{pre}{k}" in out["inner"]]
        out[key] = max(cand, key=lambda k: out["inner"][f"{pre}{k}"]["psnr"])
        print(f"{label} window selected on the inner split: k={out[key]}", flush=True)
    return out


# ================================================================== per-scene controlled configuration
# Shared stage-2 settings, frozen from slice-banana (dt = 0.006098): observation window 0.3537 time with 30 points
# (= 30 Stage-1 frames, stride 2), training horizon 0.0976 (16 frames), JEPA rollout segment 0.0488 (8 frames).
# Other scenes keep the same time spans; JEPA's stride / frame counts are derived from the scene's dt and the
# ODE-GS window is set to exactly the JEPA context span so both observe the same 30 time points.
OBS_SPAN, OBS_POINTS, HORIZON, SEGMENT = 0.35365853658536583, 30, 0.0975609756097561, 0.04878048780487805
JEPA_SHARED = {"token_dim": 192, "layers": 5, "heads": 6, "num_groups": 32, "cv_k": 0, "batch_contexts": 4,
               "gaussians_per_step": 1024, "weight_decay": 1e-4, "schedule": "cosine", "warmup_frac": 0.05,
               "min_lr_ratio": 0.01, "log_every": 1000}
ODEGS_SHARED = {"d_model": 256, "latent_dim": 128, "ode_nhidden": 256, "decoder_nhidden": 256, "extrap_points": 10,
                "batch_size": 4096, "weight_decay": 1e-4, "warmup_frac": 0.05, "min_lr_ratio": 0.01}


def plan(fe: FrontEnd, lr: float, out: str, obs_span: float = OBS_SPAN, horizon: float = HORIZON) -> dict:
    """Shared stage-2 configs. HyperNeRF defaults; D-NeRF (time_split 0.8) uses horizon 0.2 = its test range."""
    dt = float(np.median(np.diff(fe.train_fids)))
    stride = max(1, round(obs_span / ((OBS_POINTS - 1) * dt)))
    hor = max(1, round(horizon / dt))
    segment = SEGMENT * horizon / HORIZON
    jepa = {**JEPA_SHARED, "context_len": OBS_POINTS, "context_stride": stride, "max_horizon": hor,
            "rollout_segment": float(max(1, round(segment / dt))), "lr": lr}
    odegs = {**ODEGS_SHARED, "obs_time_span": (OBS_POINTS - 1) * stride * dt, "obs_points": OBS_POINTS,
             "max_extrap_time_span": hor * dt, "lr": lr}
    os.makedirs(out, exist_ok=True)
    json.dump(jepa, open(os.path.join(out, "jepa.json"), "w"))
    json.dump(odegs, open(os.path.join(out, "odegs.json"), "w"))
    info = {"dt": dt, "stride": stride, "horizon_frames": hor, "num_gaussians": fe.num_gaussians,
            "prefix_cams": len(fe.train_cams), "eval_cams": len(fe.eval_cams), "prefix_times": len(fe.train_fids),
            "last_prefix_t": float(fe.train_fids[-1]), "first_eval_t": float(min(c.fid.item() for c in fe.eval_cams)),
            "image_hw": list(fe.eval_cams[0].original_image.shape[1:])}
    json.dump(info, open(os.path.join(out, "scene.json"), "w"), indent=2)
    print(f"[plan] {json.dumps(info)}\n[plan] jepa {json.dumps(jepa)}\n[plan] odegs {json.dumps(odegs)}", flush=True)
    return info


# ================================================================== throughput benchmark
def bench(fe: FrontEnd, method: str, cfg: dict, base: dict, n: int = 300) -> float:
    sp = Split(fe, "full", 0.0)
    if method == "jepa":
        c = RealConfig(**{**cfg, "steps": n, "log_every": 10**9})
        traj = {k: v.cuda() for k, v in fe.trajectory(sp.fids).items()}
        u = torch.tensor(sp.fids / sp.dt, dtype=torch.float32, device="cuda")
        stats = Stats.from_prefix(traj["xyz"], traj["scale"], u)
        gid = kmeans_groups(traj["xyz"], c.num_groups)
        train_jepa(RealConfig(**{**asdict(c), "steps": 30}), traj, u, gid, stats, "/dev/null")  # warm-up
        torch.cuda.synchronize()
        t0 = time.time()
        train_jepa(c, traj, u, gid, stats, "/dev/null")
        torch.cuda.synchronize()
        sec = time.time() - t0
    else:
        c = {**base, **cfg}
        _, sec = train_odegs(fe, sp, c, n + 50, 0, "/dev/null")  # incl. torch.compile warm-up
        _, sec50 = train_odegs(fe, sp, c, 50, 0, "/dev/null")
        sec -= sec50
    its = n / sec
    print(f"[bench] {method} {json.dumps(cfg)} -> {its:.2f} it/s", flush=True)
    return its


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["jepa", "odegs", "baselines", "bench", "masks", "plan"])
    ap.add_argument("--source", default="/data/D-NeRF/mutant")
    ap.add_argument("--root", default="results/dnerf/mutant", help="scene results folder (holds stage1/)")
    ap.add_argument("--time_split", type=float, default=0.8)
    ap.add_argument("--split", choices=["inner", "full"], default="inner")
    ap.add_argument("--inner_cut", type=float, default=0.64)
    ap.add_argument("--cfg", default="{}", help="JSON overrides, or @path/to/file.json")
    ap.add_argument("--lr", type=float, default=3e-4, help="plan: shared stage-2 peak LR")
    ap.add_argument("--obs_span", type=float, default=OBS_SPAN, help="plan: observation window (time units)")
    ap.add_argument("--horizon", type=float, default=HORIZON, help="plan: training forecast horizon (time units)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="jepa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--mask_mode", choices=["gt", "gauss"], default="gt")
    ap.add_argument("--mask_tau", type=float, default=0.2)
    ap.add_argument("--mask_window", type=float, default=0.3)
    ap.add_argument("--mask_quantile", type=float, default=0.9)
    ap.add_argument("--mask_dilate", type=int, default=4)
    args = ap.parse_args()
    over = json.load(open(args.cfg[1:])) if args.cfg.startswith("@") else json.loads(args.cfg)
    fe = FrontEnd(args.source, os.path.join(args.root, "stage1"), args.time_split)
    odegs_base = {**ODEGS_DEFAULT, **({} if fe.is_blender else ODEGS_HYPERNERF)}

    if args.cmd == "bench":
        bench(fe, args.method, over, odegs_base)
        return
    if args.cmd == "plan":
        plan(fe, args.lr, os.path.join(args.root, "fair", "configs"), args.obs_span, args.horizon)
        return
    if args.cmd == "masks":
        os.makedirs(os.path.join(args.root, "fair"), exist_ok=True)
        compute_masks(fe, Evaluator(fe), args.inner_cut, args.mask_mode, args.mask_tau, args.mask_window,
                      args.mask_quantile, args.mask_dilate,
                      os.path.join(args.root, "fair", "masks.pt"))
        return
    masks = load_masks(args.root)
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "baselines":
        res = run_baselines(fe, args.inner_cut, masks)
    else:
        sp = Split(fe, args.split, args.inner_cut)
        if args.cmd == "jepa":
            cfg = RealConfig(**{**over, "seed": args.seed, **({"steps": args.steps} if args.steps else {})})
            res = run_jepa(fe, sp, cfg, args.out, args, masks)
        else:
            res = run_odegs(fe, sp, {**odegs_base, **over}, args.steps, args.seed, args.out, args, masks)
        res.update(split=args.split, seed=args.seed, inner_cut=args.inner_cut, t_last=sp.t_last,
                   gpu=torch.cuda.get_device_name(0), selection="last checkpoint",
                   per_time={k: per_time(v) for k, v in res["results"].items()})
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)


if __name__ == "__main__":
    main()
