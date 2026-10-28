import sys, torch, numpy as np
from gsjepa.real.frontend import FrontEnd
from gsjepa.real.fair import Split, jepa_state_fn, build_odegs, odegs_predictor
from gsjepa.real.consistency import load_jepa
s = sys.argv[1]; F = f"results/dnerf/{s}/fair"
fe = FrontEnd(f"/data/D-NeRF/{s}", f"results/dnerf/{s}/stage1", 0.8)
sp = Split(fe, "full", 0.64)
m, gid, ck = load_jepa(f"{F}/runs/jepa_full/last.pt")
jf = jepa_state_fn(fe, m, gid, sp, m.cfg.rollout_segment, sp.dt)
oc = torch.load(f"{F}/runs/odegs_full/last.pt", weights_only=False); om = build_odegs(oc["cfg"]); om.load_state_dict(oc["model_state_dict"])
ts = fe.eval_fids
of = odegs_predictor(fe, om, oc["cfg"], sp.t_last, ts)
last = fe.state_at(sp.t_last)
print("cfg max_horizon", m.cfg.max_horizon, "segment", m.cfg.rollout_segment, "ctx", m.cfg.context_len, m.cfg.context_stride, "| horizon frames to end", (ts[-1]-sp.t_last)/sp.dt)
sl = last["scale"]; print("last scale stats: min %.3f med %.3f max %.3f" % (sl.min(), sl.median(), sl.max()))
print(" t     h  | ADE j    ADE o    ADE last | p99 j   p99 o   p99 gt | scale-max j  o  gt | fracfar j o")
for t in ts[::3]:
    g = fe.state_at(t); J = jf(t); O = of(t)
    e = lambda S: (S["xyz"].cuda() - g["xyz"]).norm(dim=-1)
    d = lambda S: (S["xyz"].cuda() - last["xyz"]).norm(dim=-1)
    print(f"{t:.3f} {(t-sp.t_last)/sp.dt:5.1f} | {e(J).mean():.4f} {e(O).mean():.4f} {e(last).mean():.4f} | "
          f"{d(J).quantile(.99):.3f} {d(O).quantile(.99):.3f} {d(g).quantile(.99):.3f} | "
          f"{J['scale'].max():.2f} {O['scale'].max():.2f} {g['scale'].max():.2f} | "
          f"{(e(J)>0.3).float().mean():.4f} {(e(O)>0.3).float().mean():.4f}")
from gsjepa.real.fair import rigid_fit
print("\n t     h | mean disp vec J (from last)  | GT mean disp | J rigid-residual | J spread ratio | prefix vel of centroid")
c = fe.trajectory(sp.fids[-20:])["xyz"].mean(1); print("prefix centroid last 20 frames (y):", np.round(c[:, 1].numpy()[::3], 3), "z:", np.round(c[:, 2].numpy()[::3], 3))
for t in ts[::4]:
    g = fe.state_at(t); J = jf(t)
    dj = (J["xyz"].cuda() - last["xyz"]).mean(0); dg = (g["xyz"] - last["xyz"]).mean(0)
    R, tt = rigid_fit(last["xyz"].float(), J["xyz"].cuda().float()); res = (last["xyz"] @ R.T + tt - J["xyz"].cuda()).norm(dim=-1).mean()
    spread = (J["xyz"].cuda() - J["xyz"].cuda().mean(0)).norm(dim=-1).mean() / (last["xyz"] - last["xyz"].mean(0)).norm(dim=-1).mean()
    print(f"{t:.3f} {(t-sp.t_last)/sp.dt:5.1f} | {np.round(dj.cpu().numpy(),3)} | {np.round(dg.cpu().numpy(),3)} | {res:.3f} | {spread:.2f}")
