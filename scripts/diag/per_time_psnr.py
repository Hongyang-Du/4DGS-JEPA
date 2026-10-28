import json, sys, numpy as np
s = sys.argv[1]; F = f"results/dnerf/{s}/fair"
def pt(res, k="psnr"):
    d = {}
    for r in res["per_image"]: d.setdefault(round(r["fid"], 3), []).append(r[k])
    return {t: np.mean(v) for t, v in sorted(d.items())}
j = json.load(open(f"{F}/runs/jepa_full/metrics.json")); o = json.load(open(f"{F}/runs/odegs_full/metrics.json"))
b = json.load(open(f"{F}/baselines/metrics.json"))
print("jepa keys", list(j["results"].keys()), "| baseline keys", [k for k in b if not k.startswith("selected")][:20])
for m in j["results"]: print("jepa", m, round(j["results"][m]["psnr"], 2))
rows = {"jepa_roll": pt(j["results"]["rollout"]), "jepa_dir": pt(j["results"]["direct"]), "odegs": pt(o["results"] if "per_image" in o["results"] else list(o["results"].values())[0])}
base = b.get("results", b)
for k in ["static", "deform", "cv1", "rigid1"]:
    if k in base: rows[k] = pt(base[k])
ts = list(rows["jepa_roll"])
print("t      " + " ".join(f"{k:>9s}" for k in rows))
for t in ts[::max(1, len(ts)//16)]: print(f"{t:.3f} " + " ".join(f"{rows[k].get(t, float('nan')):9.2f}" for k in rows))
