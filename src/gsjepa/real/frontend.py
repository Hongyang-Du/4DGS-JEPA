"""Bridge to the ODE-GS code base (third_party/ODE-GS): Stage-1 front-end loading,
per-time Gaussian state extraction and the evaluation protocol B renderer/metrics.

Protocol B (ODE-GS, ICLR'26): pool the D-NeRF train+test cameras, normalise fids to [0, 1],
sort by time, first int(0.8 N) cameras are the observed prefix, the rest are evaluated.
HyperNeRF (Nerfies format, detected by dataset.json) uses the same rule with time_split 0.9; ODE-GS picks the
camera subset / resolution from the parent folder name (interp: ids[::4] + ids[2::4] at rgb/2x, vrig: train+val
at rgb/4x, misc: all ids at rgb/2x), so scenes must live at <root>/{interp,misc,vrig}/<scene>.
"""
from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
import torch

ODE_GS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../third_party/ODE-GS"))
if ODE_GS not in sys.path:
    sys.path.insert(0, ODE_GS)

import lpips  # noqa: E402
from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render_ode  # noqa: E402
from scene import DeformModel, GaussianModel, Scene  # noqa: E402
from utils.image_utils import psnr as psnr_per_channel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402
from utils.ode_load_utils import compute_deformed_gaussian_state  # noqa: E402


class FrontEnd:
    """Trained Deformable-GS (Stage 1) plus the protocol-B camera split."""

    def __init__(self, source_path: str, model_path: str, time_split: float = 0.8):
        self.is_blender = os.path.exists(os.path.join(source_path, "transforms_train.json"))
        parser = ArgumentParser()
        lp, pp = ModelParams(parser), PipelineParams(parser)
        args = parser.parse_args(["-s", source_path, "-m", model_path, "--eval"] + (["--is_blender"] if self.is_blender else []))
        self.dataset, self.pipe = lp.extract(args), pp.extract(args)
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        scene = Scene(self.dataset, self.gaussians, load_iteration=-1, shuffle=False)
        self.deform = DeformModel(is_blender=self.is_blender, is_6dof=False)
        self.deform.load_weights(model_path, iteration=-1)
        self.deform.deform.eval()

        cams = scene.getTrainCameras() + scene.getTestCameras()
        cams.sort(key=lambda c: c.fid.item())
        split = int(len(cams) * time_split)
        self.train_cams, self.eval_cams = cams[:split], cams[split:]
        self.time_split = time_split
        self.background = torch.zeros(3, device="cuda")
        self.num_gaussians = self.gaussians.get_xyz.shape[0]

    @property
    def train_fids(self) -> np.ndarray:
        return np.unique([c.fid.item() for c in self.train_cams])

    @property
    def eval_fids(self) -> np.ndarray:
        return np.unique([c.fid.item() for c in self.eval_cams])

    @torch.no_grad()
    def state_at(self, fid: float) -> Dict[str, torch.Tensor]:
        """Deformed Gaussian state at time fid: xyz [N,3], quaternion [N,4], scale [N,3].

        The quaternion is kept unnormalised: the rasterizer does not normalise it and Deformable-GS
        was trained on get_rotation + d_rotation as is."""
        t = torch.full((self.num_gaussians, 1), float(fid), device="cuda")
        xyz, rot, scale, _, _ = compute_deformed_gaussian_state(self.gaussians, self.deform, t)
        return {"xyz": xyz, "rot": rot, "scale": scale}

    @torch.no_grad()
    def trajectory(self, fids: np.ndarray) -> Dict[str, torch.Tensor]:
        states = [self.state_at(f) for f in fids]
        return {k: torch.stack([s[k] for s in states]).cpu() for k in states[0]}


class Evaluator:
    """Protocol-B metrics: SSIM (3DGS), LPIPS-vgg on [0,1] inputs, PSNR.

    ``psnr`` follows ODE-GS (per-channel PSNR averaged); ``psnr_img`` is the usual whole-image PSNR."""

    def __init__(self, fe: FrontEnd):
        self.fe = fe
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).cuda()
        self._lpips_spatial = None

    def lpips_spatial(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._lpips_spatial is None:
            self._lpips_spatial = lpips.LPIPS(net="vgg", verbose=False, spatial=True).cuda()
        return self._lpips_spatial(a, b)

    @torch.no_grad()
    def render(self, cam, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        cam.load2device("cuda")
        out = render_ode(cam, self.fe.gaussians, self.fe.pipe, self.fe.background,
                         state["xyz"].cuda(), state["rot"].cuda(), state["scale"].cuda())
        return out["render"].clamp(0.0, 1.0)

    @torch.no_grad()
    def evaluate(self, cams: List, state_fn, save_dir: str | None = None, masks: Dict[str, torch.Tensor] | None = None,
                 fast: bool = False) -> Dict[str, float]:
        """state_fn(fid) -> Gaussian state; averages metrics over all cameras (per image).

        masks: {mask_key(cam) (or image_name): bool [H, W]} dynamic-region masks -> adds dyn_psnr (MSE over mask pixels),
        dyn_ssim (SSIM map averaged over the mask) and dyn_lpips (spatial LPIPS map averaged over the mask).
        fast: PSNR only (training curves)."""
        rows = []
        for cam in cams:
            fid = cam.fid.item()
            img = self.render(cam, state_fn(fid))
            gt = cam.original_image.cuda()
            mse = ((img - gt) ** 2).mean()
            row = {"fid": fid, "psnr": psnr_per_channel(img, gt).mean().item(), "psnr_img": (-10 * torch.log10(mse)).item()}
            if not fast:
                row["ssim"] = ssim(img[None], gt[None]).item()
                row["lpips"] = self.lpips(img[None], gt[None]).item()
            m = None if masks is None else masks.get(mask_key(cam), masks.get(cam.image_name))
            if m is not None and m.any():  # an empty mask (no change yet) has no dynamic-region score
                m = m.cuda().float()
                row["dyn_frac"] = m.mean().item()
                row["dyn_psnr"] = (-10 * torch.log10((((img - gt) ** 2).mean(0) * m).sum() / m.sum().clamp_min(1))).item()
                if not fast:
                    row["dyn_ssim"] = ((ssim_map(img, gt) * m).sum() / m.sum().clamp_min(1)).item()
                    lp = self.lpips_spatial(img[None], gt[None])
                    row["dyn_lpips"] = ((lp[0, 0] * m).sum() / m.sum().clamp_min(1)).item()
            rows.append(row)
            if save_dir is not None:
                import imageio.v2 as imageio
                os.makedirs(save_dir, exist_ok=True)
                imageio.imwrite(os.path.join(save_dir, f"{cam.image_name}_{fid:.4f}.png"),
                                (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            cam.load2device("cpu")
        keys = dict.fromkeys(k for r in rows for k in r if k != "fid")
        summary = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
        summary["per_image"] = rows
        return summary


def mask_key(cam) -> str:
    """D-NeRF reuses image names across its train/test folders, so masks are keyed by name and time."""
    return f"{cam.image_name}@{cam.fid.item():.6f}"


def ssim_map(img: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Per-pixel SSIM map [H, W] (3DGS SSIM, channel-averaged)."""
    from utils.loss_utils import create_window
    import torch.nn.functional as F
    C = img.shape[0]
    win = create_window(window_size, C).to(img.device).type_as(img)
    x, y = img[None], gt[None]
    mu1, mu2 = F.conv2d(x, win, padding=window_size // 2, groups=C), F.conv2d(y, win, padding=window_size // 2, groups=C)
    s11 = F.conv2d(x * x, win, padding=window_size // 2, groups=C) - mu1 ** 2
    s22 = F.conv2d(y * y, win, padding=window_size // 2, groups=C) - mu2 ** 2
    s12 = F.conv2d(x * y, win, padding=window_size // 2, groups=C) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return m[0].mean(0)
