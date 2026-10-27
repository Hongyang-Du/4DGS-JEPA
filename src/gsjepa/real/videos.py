"""Render protocol-B comparison frames/videos for one scene.

Frames are written to <out>/<method>/{eval,sweep}/NNN.png:
  eval   the evaluation cameras (t > 0.8) in time order, each from its own GT camera pose
  sweep  one fixed evaluation camera, t swept over [0, 1]; for t inside the prefix every
         Gaussian method shows the Stage-1 Deformable-GS fit, after it each method's forecast

    python -m gsjepa.real.videos render --scene mutant --methods gt deform cv4 jepa odegs
    python -m gsjepa.real.videos render --scene mutant --methods dnerf
    python -m gsjepa.real.videos compose --scene mutant --methods gt deform cv4 odegs dnerf jepa

Fair-benchmark runs (gsjepa.real.fair) are rendered as jepa_fair / odegs_fair (their last.pt given by
--jepa_ckpt / --odegs_ckpt), cvK for any window K, and D-NeRF from a chosen checkpoint (--dnerf_ckpt,
--dnerf_name).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

LABELS = {
    "gt": "Ground truth",
    "deform": "Deformable-GS",
    "static": "Static",
    "cv4": "Const. velocity",
    "odegs": "ODE-GS",
    "dnerf": "D-NeRF",
    "jepa": "4DGS-JEPA (ours)",
    "jepa_fair": "4DGS-JEPA (ours)",
    "odegs_fair": "ODE-GS (fair)",
    "odegs_official": "ODE-GS official",
}
NUM_SWEEP = 101


ROOT = None  # scene results folder; default results/dnerf/<scene> (set by --root)


def _paths(scene: str):
    root = ROOT or f"results/dnerf/{scene}"
    return root, os.path.join(root, "videos")


def _save(img: torch.Tensor | np.ndarray, path: str) -> None:
    import imageio.v2 as imageio
    if torch.is_tensor(img):
        img = img.permute(1, 2, 0).cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, (np.clip(img, 0, 1) * 255).round().astype(np.uint8))


# ---------------------------------------------------------------- Gaussian-based methods
def render_gaussian_methods(scene: str, methods, data: str, sweep_cam: int, jepa_ckpt: str | None = None,
                            jepa_segment: float | None = -1, odegs_ckpt: str | None = None,
                            time_split: float = 0.8) -> None:
    from .frontend import Evaluator, FrontEnd
    from .jepa import RealConfig, RealJEPA, Stats
    from .run_scene import window

    root, out = _paths(scene)
    fe = FrontEnd(data if os.path.exists(os.path.join(data, "dataset.json")) else os.path.join(data, scene),
                  os.path.join(root, "stage1"), time_split)
    ev = Evaluator(fe)
    fids = fe.train_fids
    t_last = float(fids[-1])
    cam = fe.eval_cams[sweep_cam]
    sweep = np.linspace(0.0, 1.0, NUM_SWEEP)
    meta = {"eval_fids": [c.fid.item() for c in fe.eval_cams], "sweep_times": sweep.tolist(),
            "sweep_cam": sweep_cam, "last_prefix_fid": t_last}
    os.makedirs(out, exist_ok=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)

    traj = None
    cvs = [m for m in methods if m.startswith("cv")]
    if cvs or any(m in methods for m in ["static", "jepa", "jepa_fair"]):
        traj = {k: v.cuda() for k, v in fe.trajectory(fids).items()}
    dt = float(np.median(np.diff(fids)))

    fns = {"deform": fe.state_at}
    if traj is not None:
        last = {k: v[-1] for k, v in traj.items()}
        u_last = fids[-1] / dt
        v4 = (traj["xyz"][-1] - traj["xyz"][-5]) / ((fids[-1] - fids[-5]) / dt)
        fns["static"] = lambda f: last
        fns["cv4"] = lambda f: {**last, "xyz": last["xyz"] + v4 * ((f - fids[-1]) / dt)}
        for m in cvs:
            k = int(m[2:])
            vk = (traj["xyz"][-1] - traj["xyz"][-1 - k]) / ((fids[-1] - fids[-1 - k]) / dt)
            fns[m] = lambda f, vk=vk: {**last, "xyz": last["xyz"] + vk * ((f - fids[-1]) / dt)}
    for jm in [m for m in methods if m in ("jepa", "jepa_fair")]:
        path = os.path.join(root, "jepa", "jepa.pt") if jm == "jepa" else jepa_ckpt
        ck = torch.load(path, weights_only=False)
        cfg = RealConfig(**ck["cfg"])
        gid = ck["gid"].cuda()
        u = torch.tensor(fids / dt, dtype=torch.float32, device="cuda")
        stats = ck.get("stats")
        if stats is None:  # older checkpoints: recompute exactly as run_scene did (seeded)
            from .run_scene import set_seed
            set_seed(cfg.seed)
            stats = Stats.from_prefix(traj["xyz"], traj["scale"], u)
        model = RealJEPA(cfg, stats).cuda().eval()
        model.load_state_dict(ck["model"])
        L, T = cfg.context_len, len(fids)
        S = cfg.context_stride
        hist = torch.arange(T - 1 - S * (L - 1), T, S, device="cuda")[None]
        allg = torch.arange(fe.num_gaussians, device="cuda")
        with torch.no_grad():
            ctx = window(traj, hist, allg)
            z0 = model.encoder(ctx["xyz"], ctx["rot"], ctx["scale"], u[hist], gid)
            src = model.source(ctx, u[hist])

        seg = cfg.rollout_segment if jm == "jepa" or jepa_segment == -1 else jepa_segment

        def jepa_fn(f, model=model, z0=z0, src=src, gid=gid, seg=seg):
            with torch.no_grad():
                g = model.predict(z0, src, (f - t_last) / dt, gid, seg)
            return {k: v[0] for k, v in g.items() if k != "vel"}
        fns[jm] = jepa_fn
    if "odegs_fair" in methods:
        from .fair import build_odegs, odegs_predictor as fair_predictor
        ck = torch.load(odegs_ckpt, weights_only=False)
        model = build_odegs(ck["cfg"])
        model.load_state_dict(ck["model_state_dict"])
        fns["odegs_fair"] = fair_predictor(fe, model, ck["cfg"], t_last,
                                           meta["eval_fids"] + [t for t in meta["sweep_times"] if t > t_last])
    for m in methods:
        if m == "odegs_official":
            fns[m] = odegs_predictor(fe, root, meta, None)
        elif m.startswith("odegs") and m != "odegs_fair":   # "odegs" = last epoch, "odegs@N" = epoch N
            fns[m] = odegs_predictor(fe, root, meta, int(m.split("@")[1]) if "@" in m else None)

    for m in methods:
        if m == "gt":
            for i, c in enumerate(fe.eval_cams):
                _save(c.original_image, os.path.join(out, "gt", "eval", f"{i:03d}.png"))
            continue
        fn = fns[m]
        res = ev.evaluate(fe.eval_cams, fn)
        print(f"{m:8s} PSNR {res['psnr']:.2f} (img {res['psnr_img']:.2f}) SSIM {res['ssim']:.4f} LPIPS {res['lpips']:.4f}", flush=True)
        json.dump(res, open(os.path.join(out, f"metrics_{m}.json"), "w"), indent=2)
        for i, c in enumerate(fe.eval_cams):
            _save(ev.render(c, fn(c.fid.item())), os.path.join(out, m, "eval", f"{i:03d}.png"))
            c.load2device("cpu")
        for i, t in enumerate(sweep):
            state = fe.state_at(t) if t <= t_last else fn(float(t))
            _save(ev.render(cam, state), os.path.join(out, m, "sweep", f"{i:03d}.png"))
        cam.load2device("cpu")


def odegs_predictor(fe, root: str, meta: dict, epoch: int | None = None):
    """ODE-GS forecasts with the official model and observation window (last 0.6 of the prefix,
    30 uniform points), predicted for all eval and sweep times at once.

    epoch=None takes the last epoch. (ODE-GS itself reports the epoch with the best PSNR on the
    t > 0.8 evaluation cameras, i.e. selected on the test set.)"""
    from omegaconf import OmegaConf
    from evaluate_extrapolation import load_checkpoint, predict_trajectories_at_times
    from utils.config_utils import DotDict

    run = glob.glob(os.path.join(root, "stage1", "*", "config.yaml"))[0]
    cfg = DotDict(OmegaConf.to_container(OmegaConf.load(run), resolve=True))
    ckpts = glob.glob(os.path.join(os.path.dirname(run), "epoch_*", "model.pth"))
    by_epoch = {int(os.path.basename(os.path.dirname(c)).split("_")[1]): c for c in ckpts}
    ckpt = by_epoch[max(by_epoch) if epoch is None else epoch]
    print(f"ODE-GS checkpoint: {ckpt}", flush=True)
    model = load_checkpoint(ckpt, cfg, fe.gaussians)
    t_last = meta["last_prefix_fid"]
    obs = torch.linspace(t_last - cfg.data.obs_time_span, t_last, steps=cfg.data.obs_points, device="cuda")
    targets = sorted(set(meta["eval_fids"]) | {t for t in meta["sweep_times"] if t > t_last})
    with torch.no_grad():
        pred = predict_trajectories_at_times(model, fe.deform, fe.gaussians, obs,
                                             torch.tensor(targets, device="cuda"), batch_size=cfg.data.batch_size)
    lookup = {round(k, 6): v for k, v in pred.items()}

    def fn(f):
        p = lookup[round(float(torch.tensor(f, dtype=torch.float32, device="cuda").item()), 6)]
        return {"xyz": p[:, :3], "rot": p[:, 3:7], "scale": p[:, 7:10]}
    return fn


# ---------------------------------------------------------------- D-NeRF
def render_dnerf(scene: str, data_b: str, ckpt: str | None = None, name: str = "dnerf") -> None:
    """Official D-NeRF network (trained on the protocol-B prefix) rendered at full 800x800 on the
    same cameras, composited on black (rgb_black = rgb_white - (1 - acc))."""
    from .frontend import Evaluator  # noqa: F401  (imports lpips etc. via ODE-GS path)
    import lpips
    from utils.image_utils import psnr as psnr_per_channel
    from utils.loss_utils import ssim

    root, out = _paths(scene)
    meta = json.load(open(os.path.join(out, "meta.json")))
    dn = os.path.abspath("third_party/D-NeRF")
    sys.path.insert(0, dn)
    cwd = os.getcwd()
    os.chdir(dn)
    from run_dnerf import config_parser, create_nerf, render
    os.chdir(cwd)

    expdir = os.path.join(root, "dnerf")
    ckpts = [ckpt] if ckpt else sorted(glob.glob(os.path.join(expdir, "*.tar")))
    args = config_parser().parse_args(["--config", os.path.join(dn, "configs", f"{scene}.txt"),
                                       "--datadir", os.path.join(data_b, scene),
                                       "--basedir", root, "--expname", "dnerf", "--ft_path", ckpts[-1]])
    print(f"D-NeRF checkpoint: {ckpts[-1]}", flush=True)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    _, render_kwargs_test, start, _, _ = create_nerf(args)
    render_kwargs_test.update(near=2.0, far=6.0)  # Blender bounds, as in run_dnerf.train
    torch.set_default_tensor_type("torch.FloatTensor")
    test = json.load(open(os.path.join(data_b, scene, "transforms_test.json")))
    assert np.allclose([f["time"] for f in test["frames"]], meta["eval_fids"], atol=1e-6)
    H = W = 800
    focal = 0.5 * W / np.tan(0.5 * test["camera_angle_x"])
    def draw(c2w, t):
        with torch.no_grad():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            rgb, _, acc, _ = render(H, W, focal, chunk=args.chunk, c2w=torch.tensor(c2w[:3, :4], dtype=torch.float32),
                                    frame_time=torch.tensor(t, dtype=torch.float32), **render_kwargs_test)
            torch.set_default_tensor_type("torch.FloatTensor")
        rgb = rgb - (1.0 - acc[..., None]) if args.white_bkgd else rgb
        return rgb.clamp(0, 1).permute(2, 0, 1)

    import imageio.v2 as imageio
    lp = lpips.LPIPS(net="vgg", verbose=False).cuda()
    rows = []
    for i, f in enumerate(test["frames"]):
        img = draw(np.array(f["transform_matrix"]), f["time"])
        gt_rgba = imageio.imread(os.path.join(data_b, scene, f["file_path"] + ".png")).astype(np.float32) / 255.0
        gt = torch.tensor(gt_rgba[..., :3] * gt_rgba[..., 3:4], device="cuda").permute(2, 0, 1)
        mse = ((img - gt) ** 2).mean()
        rows.append({"fid": f["time"], "psnr": psnr_per_channel(img, gt).mean().item(),
                     "psnr_img": (-10 * torch.log10(mse)).item(),
                     "ssim": ssim(img[None], gt[None]).item(), "lpips": lp(img[None], gt[None]).item()})
        _save(img, os.path.join(out, name, "eval", f"{i:03d}.png"))
    res = {k: float(np.mean([r[k] for r in rows])) for k in ["psnr", "psnr_img", "ssim", "lpips"]}
    res["per_image"], res["checkpoint"], res["iteration"] = rows, ckpts[-1], int(start)
    print(f"{name}    PSNR {res['psnr']:.2f} (img {res['psnr_img']:.2f}) SSIM {res['ssim']:.4f} LPIPS {res['lpips']:.4f} @ iter {start}", flush=True)
    json.dump(res, open(os.path.join(out, f"metrics_{name}.json"), "w"), indent=2)
    c2w = np.array(test["frames"][meta["sweep_cam"]]["transform_matrix"])
    for i, t in enumerate(meta["sweep_times"]):
        _save(draw(c2w, t), os.path.join(out, name, "sweep", f"{i:03d}.png"))


# ---------------------------------------------------------------- composition
def compose(scene: str, methods, fps_eval: int = 6, fps_sweep: int = 20, prefix: str | None = None,
            labels: dict | None = None, cols: int | None = None) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    _, out = _paths(scene)
    meta = json.load(open(os.path.join(out, "meta.json")))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default(size=26)
    metrics = {}
    for m in methods:
        p = os.path.join(out, f"metrics_{m}.json")
        if os.path.exists(p):
            metrics[m] = json.load(open(p))

    def tile(path, label, sub=None, size=400):  # height `size`, aspect ratio kept
        im = Image.open(path).convert("RGB")
        w = int(round(size * im.width / im.height / 8)) * 8
        im = im.resize((w, size), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.text((10, 8), label, fill=(255, 255, 255), font=font)
        if sub:
            d.text((10, size - 36), sub, fill=(255, 220, 120), font=font)
        return np.asarray(im)

    for kind, fps in [("eval", fps_eval), ("sweep", fps_sweep)]:
        n = len(meta["eval_fids"]) if kind == "eval" else len(meta["sweep_times"])
        ms = [m for m in methods if os.path.isdir(os.path.join(out, m, kind))]
        frames = []
        for i in range(n):
            if kind == "eval":
                t = meta["eval_fids"][i]
                head = f"t={t:.3f}"
            else:
                t = meta["sweep_times"][i]
                head = f"t={t:.2f} " + ("observed" if t <= meta["last_prefix_fid"] else "FORECAST")
            row = []
            for m in ms:
                sub = None
                if kind == "eval" and m in metrics:
                    sub = f"{metrics[m]['per_image'][i]['psnr']:.1f} dB"
                label = (labels or {}).get(m) or LABELS.get(m, f"ODE-GS ep{m.split('@')[1]} (test-selected)" if m.startswith("odegs@") else m)
                row.append(tile(os.path.join(out, m, kind, f"{i:03d}.png"), label, sub))
            if cols and len(row) > cols:  # grid layout, padded with black tiles
                row += [np.zeros_like(row[0])] * (-len(row) % cols)
                grid = np.concatenate([np.concatenate(row[r:r + cols], axis=1) for r in range(0, len(row), cols)], axis=0)
            else:
                grid = np.concatenate(row, axis=1)
            bar = Image.new("RGB", (grid.shape[1], 44), (40, 40, 40))
            ImageDraw.Draw(bar).text((10, 8), head, fill=(255, 255, 255), font=font)
            frames.append(np.concatenate([np.asarray(bar), grid], axis=0))
        path = os.path.join(out, f"{prefix or scene}_{kind}.mp4")
        imageio.mimwrite(path, frames, fps=fps, quality=8, macro_block_size=8)
        print(f"wrote {path} ({len(frames)} frames, methods: {', '.join(ms)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "compose"])
    ap.add_argument("--scene", default="mutant")
    ap.add_argument("--methods", nargs="+", default=["gt", "deform", "cv4", "jepa"])
    ap.add_argument("--data", default="/data/D-NeRF")
    ap.add_argument("--data_b", default="data/protocol_b")
    ap.add_argument("--sweep_cam", type=int, default=30)
    ap.add_argument("--jepa_ckpt", default=None)
    ap.add_argument("--jepa_segment", type=float, default=-1, help="-1: cfg.rollout_segment, 0: direct jump")
    ap.add_argument("--odegs_ckpt", default=None)
    ap.add_argument("--dnerf_ckpt", default=None)
    ap.add_argument("--dnerf_name", default="dnerf")
    ap.add_argument("--prefix", default=None, help="output video name prefix (default: scene)")
    ap.add_argument("--labels", default="{}", help="JSON {method: label}")
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--root", default=None, help="scene results folder (default results/dnerf/<scene>)")
    ap.add_argument("--time_split", type=float, default=0.8)
    args = ap.parse_args()
    global ROOT
    ROOT = args.root
    if args.cmd == "compose":
        compose(args.scene, args.methods, prefix=args.prefix, labels=json.loads(args.labels), cols=args.cols)
        return
    gs = [m for m in args.methods if not m.startswith("dnerf")]
    if gs:
        render_gaussian_methods(args.scene, gs, args.data, args.sweep_cam, args.jepa_ckpt,
                                None if args.jepa_segment == 0 else args.jepa_segment, args.odegs_ckpt, args.time_split)
    if any(m.startswith("dnerf") for m in args.methods):
        render_dnerf(args.scene, args.data_b, args.dnerf_ckpt, args.dnerf_name)


if __name__ == "__main__":
    main()
