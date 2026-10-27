"""Write a protocol-B copy of a D-NeRF scene for baselines that read the Blender format directly.

Pools transforms_train + transforms_test (train first, then test), stable-sorts by time and puts the
first int(0.8 N) cameras in transforms_train.json and the rest in transforms_test.json, exactly as
ODE-GS does. transforms_val.json gets a few prefix cameras (D-NeRF needs a val split for logging).
Image folders are symlinked.

    python scripts/prepare_protocol_b.py /data/D-NeRF/mutant data/protocol_b/mutant
"""
import json
import os
import sys


def main(src: str, dst: str, split: float = 0.8) -> None:
    os.makedirs(dst, exist_ok=True)
    metas = {s: json.load(open(os.path.join(src, f"transforms_{s}.json"))) for s in ["train", "test"]}
    frames = metas["train"]["frames"] + metas["test"]["frames"]
    frames = sorted(frames, key=lambda f: f["time"])
    k = int(len(frames) * split)
    out = {"train": frames[:k], "val": frames[:k][::27], "test": frames[k:]}
    for s, fr in out.items():
        meta = {key: v for key, v in metas["train"].items() if key != "frames"}
        meta["frames"] = fr
        json.dump(meta, open(os.path.join(dst, f"transforms_{s}.json"), "w"), indent=2)
    for d in ["train", "test"]:
        link = os.path.join(dst, d)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(src, d)), link)
    print(f"{len(out['train'])} prefix cameras (t <= {out['train'][-1]['time']:.4f}), "
          f"{len(out['test'])} eval cameras (t >= {out['test'][0]['time']:.4f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
