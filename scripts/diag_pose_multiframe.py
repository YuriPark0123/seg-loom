"""Two follow-ups of diag_pose.py on static cameras of the eval split:

  1. displacement scale: solve with uvh + alpha * disp (alpha > 1 helping = the network under-predicts, i.e. its
     matches are biased toward the hypothesis, which a correlated error looks like)
  2. multi-frame: one pose from the matches of K frames of the same static camera (the robot moves, so the points
     cover far more of the workspace than in one frame). uvh + disp estimates the TRUE pixel of each base-frame
     point whatever the frame's own hypothesis, so the frames' matches can be stacked.

    python scripts/diag_pose_multiframe.py out/stage1_main/stages/stage3.5_final_step209628.pt --episodes 60 --frames 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from shadow.learn.data import SegLoomFrames, collate  # noqa: E402
from shadow.learn.geometry import rot_angle  # noqa: E402
from shadow.learn.register import solve_pose  # noqa: E402
from train_stage1 import CAMS, POSE_PRIOR, REAL, Stage1Net, prepare, sample_flow  # noqa: E402


def err(T, T_gt):
    return (torch.rad2deg(rot_angle(T[:, :3, :3], T_gt[:, :3, :3])).cpu().numpy(),
            ((T[:, :3, 3] - T_gt[:, :3, 3]).norm(dim=-1) * 100).cpu().numpy())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="data/eval")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    s = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta = s["args"]
    net = Stage1Net(pretrained=False).to(dev).eval()
    net.load_state_dict(s["net"])
    hyp = dict(rot_deg=ta["rot_deg"], trans_m=ta["trans_m"], spacing=ta["spacing"], r_max=ta["r_max"], corr=ta["corr"])
    ds = SegLoomFrames(a.data, scale=ta["scale"], hypothesis=hyp, seed=20261126, deterministic=True)
    # episodes with at least one static top/front camera
    start, cands = 0, []
    for i, (root, ep, robot, T) in enumerate(ds.episodes):
        meta = json.load(open(os.path.join(root, "episodes", ep, "meta.json")))
        st = [c for c in ("top", "front") if meta["cameras"][c].get("motion") == "static"]
        if st and T >= a.frames:
            cands.append((start, T, robot, st))
        start += T
    random.Random(0).shuffle(cands)
    # real robots first so they are represented
    cands = sorted(cands, key=lambda c: c[2] not in REAL)[: a.episodes // 4] + \
        [c for c in cands if c[2] not in REAL][: a.episodes - a.episodes // 4]
    alphas = (1.0,)
    res = {}
    def add(group, key, r, t):
        for g in ("all", group):
            d = res.setdefault(g, {}).setdefault(key, [[], []])
            d[0] += list(np.atleast_1d(r)); d[1] += list(np.atleast_1d(t))
    for (s0, T, robot, st) in cands:
        idx = [s0 + int(t) for t in np.linspace(0, T - 1, a.frames).round()]
        b = next(iter(DataLoader(Subset(ds, idx), len(idx), collate_fn=collate, num_workers=0)))
        x, t = prepare(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, flow = net(x)
        disp = sample_flow(flow.float(), t["uvh"], *x.shape[-2:])
        grp = "real" if robot in REAL else "procgen"
        C = len(CAMS)
        for c in st:
            ci = CAMS.index(c)
            v = torch.arange(len(idx), device=dev) * C + ci           # views of this camera, one per frame
            Tg = t["T_gt"][v]
            if (Tg - Tg[:1]).abs().max() > 1e-4:
                continue                                              # not actually static
            cams = {k: val[v] for k, val in t["cams"].items()}
            X, w, uvh, uvt, Tn = t["X"][v], t["w"][v], t["uvh"][v], t["uvt"][v], t["T_noisy"][v]
            add(grp, "single/before", *err(Tn, Tg))
            for al in alphas:
                for pr in (None, POSE_PRIOR):
                    Ts, _ = solve_pose(Tn, X, uvh + al * disp[v], w, cams, prior_sigma=pr)
                    add(grp, f"single/a{al}/{'prior' if pr else 'noprior'}", *err(Ts, Tg))
            # joint: stack all K frames' matches, init from frame 0's noisy camera
            cam1 = {k: val[:1] for k, val in cams.items()}
            for name, uv in (("net", uvh + disp[v]), ("oracle", uvt)):
                for pr, ppx in ((None, 2.0), (POSE_PRIOR, 2.0), (POSE_PRIOR, 0.15), (POSE_PRIOR, 0.05)):
                    Tj, _ = solve_pose(Tn[:1], X.reshape(1, -1, 3), uv.reshape(1, -1, 2), w.reshape(1, -1), cam1,
                                       iters=20, prior_sigma=pr, prior_px=ppx)
                    add(grp, f"joint{len(idx)}/{name}/{'prior_px' + str(ppx) if pr else 'noprior'}", *err(Tj, Tg[:1]))
            add(grp, f"joint{len(idx)}/before", *err(Tn[:1], Tg[:1]))
    out = {g: {k: dict(rot=float(np.median(r)), trans=float(np.median(tr)), n=len(r)) for k, (r, tr) in d.items()}
           for g, d in res.items()}
    for g in ("all", "real", "procgen"):
        if g in out:
            print(f"\n== {g}")
            for k, m in out[g].items():
                print(f"  {k:28s} rot {m['rot']:5.2f} deg  trans {m['trans']:5.2f} cm  (n={m['n']})")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
