"""Fixed evaluation of a stage-1 checkpoint on the eval split (same frames and same camera perturbations every time,
so checkpoints / data stages are directly comparable).

Frames: every --stride-th time step of data/eval (all 54 robots: 4 real + 50 procgen, never trained on).
Hypothesis noise and correspondence sampling are seeded per frame (SegLoomFrames deterministic=True).
Metrics: robot segmentation IoU (amodal = incl. hidden parts, visible), camera pose error after the solve
(median deg / cm; before = the perturbed initial camera), per group all / real / procgen and per camera.

    python scripts/eval_stage1.py out/stage1/last.pt --out out/stage1/eval_stage1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from shadow.learn.data import SegLoomFrames, collate  # noqa: E402
from train_stage1 import Stage1Net, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="data/eval")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    s = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    ta = s["args"]
    net = Stage1Net(pretrained=False).to(a.device)
    net.load_state_dict(s["net"])
    hyp = dict(rot_deg=ta["rot_deg"], trans_m=ta["trans_m"], spacing=ta["spacing"], r_max=ta["r_max"], corr=ta["corr"])
    ds = SegLoomFrames(a.data, scale=ta["scale"], hypothesis=hyp, seed=20261126, deterministic=True)
    sub = Subset(ds, list(range(0, len(ds), a.stride)))
    dl = DataLoader(sub, a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    res = evaluate(net, dl, a.device)
    rec = dict(ckpt=os.path.abspath(a.ckpt), step=s["step"], ckpt_time=s.get("time"), data=s.get("data"),
               eval_data=a.data, frames=len(sub), eval_time=time.strftime("%Y-%m-%d %H:%M:%S"),
               minutes=(time.time() - t0) / 60, metrics=res)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)
    print(f"step {s['step']} | {len(sub)} frames x 3 cams | {rec['minutes']:.1f} min")
    for g in ("all", "real", "procgen"):
        m = res[g]
        print(f"  {g:8s} IoU amodal {m['iou_amodal']:.3f} visible {m['iou_visible']:.3f} | rot {m['rot_deg_before']:.2f} -> "
              f"{m['rot_deg_after']:.2f} deg | trans {m['trans_cm_before']:.2f} -> {m['trans_cm_after']:.2f} cm (median)")


if __name__ == "__main__":
    main()
