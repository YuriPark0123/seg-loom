"""Where does the camera-pose error of stage 1 come from? Runs the network once per eval batch and solves the pose
with several variants on the same matches:

  oracle        uv = true pixel of the matched points (upper bound of the solve itself)
  oracle+n<k>   oracle + Gaussian pixel noise sigma k px (conditioning: how much pose error does k px cost)
  net           network displacement (as in eval), with prior / Huber / iteration variants

Metrics per variant: median rot deg, trans cm, reproj px, per camera, plus the error split along the camera axes
(depth = optical axis vs lateral), which tells conditioning (depth / rot-trans trade-off) from bad matches.

    python scripts/diag_pose.py out/stage1_main/stages/stage3.5_final_step209628.pt --stride 60 --out diag.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from shadow.learn.data import SegLoomFrames, collate  # noqa: E402
from shadow.learn.geometry import rot_angle  # noqa: E402
from shadow.learn.register import solve_pose  # noqa: E402
from train_stage1 import CAMS, POSE_PRIOR, REAL, Stage1Net, prepare, reproj_px, sample_flow  # noqa: E402

VARIANTS = {
    # name: (source, noise_px, solve kwargs)
    "oracle_noprior": ("oracle", 0, dict(prior_sigma=None)),
    "oracle_prior": ("oracle", 0, dict(prior_sigma=POSE_PRIOR)),
    "oracle+n1_noprior": ("oracle", 1.0, dict(prior_sigma=None)),
    "oracle+n2_noprior": ("oracle", 2.0, dict(prior_sigma=None)),
    "oracle+n2_prior": ("oracle", 2.0, dict(prior_sigma=POSE_PRIOR)),
    "net_prior(eval)": ("net", 0, dict(prior_sigma=POSE_PRIOR)),
    "net_noprior": ("net", 0, dict(prior_sigma=None)),
    "net_prior_weak4x": ("net", 0, dict(prior_sigma=(0.1, 0.06))),
    "net_prior_strong2x": ("net", 0, dict(prior_sigma=(0.025, 0.015))),
    "net_prior_px0.15": ("net", 0, dict(prior_sigma=POSE_PRIOR, prior_px=0.15)),
    "net_prior_px0.3": ("net", 0, dict(prior_sigma=POSE_PRIOR, prior_px=0.3)),
    "net_prior_px0.6": ("net", 0, dict(prior_sigma=POSE_PRIOR, prior_px=0.6)),
    "net_prior_px1": ("net", 0, dict(prior_sigma=POSE_PRIOR, prior_px=1.0)),
    "net_prior_iters30": ("net", 0, dict(prior_sigma=POSE_PRIOR, iters=30)),
    "net_prior_huber1": ("net", 0, dict(prior_sigma=POSE_PRIOR, huber_px=1.0)),
    "net_prior_huber4": ("net", 0, dict(prior_sigma=POSE_PRIOR, huber_px=4.0)),
}


def axis_err(T, T_gt):
    """translation error in the TRUE camera frame: (|lateral| cm, |depth| cm); rotation error split into
    roll about the optical axis vs tilt (deg)."""
    dt = (T[:, :3, 3] - T_gt[:, :3, 3])
    dc = (T_gt[:, :3, :3].transpose(1, 2) @ dt[..., None])[..., 0] * 100
    dR = T_gt[:, :3, :3].transpose(1, 2) @ T[:, :3, :3]
    w = torch.stack([dR[:, 2, 1] - dR[:, 1, 2], dR[:, 0, 2] - dR[:, 2, 0], dR[:, 1, 0] - dR[:, 0, 1]], -1) / 2
    w = torch.rad2deg(torch.asin(w.norm(dim=-1).clamp(max=1)))[:, None] * w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return dc[:, :2].norm(dim=-1), dc[:, 2].abs(), w[:, 2].abs(), w[:, :2].norm(dim=-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default="data/eval")
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--corr", type=int, default=None, help="matches per view (default: as trained)")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    s = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta = s["args"]
    net = Stage1Net(pretrained=False).to(dev).eval()
    net.load_state_dict(s["net"])
    hyp = dict(rot_deg=ta["rot_deg"], trans_m=ta["trans_m"], spacing=ta["spacing"], r_max=ta["r_max"],
               corr=a.corr or ta["corr"])
    ds = SegLoomFrames(a.data, scale=ta["scale"], hypothesis=hyp, seed=20261126, deterministic=True)
    sub = Subset(ds, list(range(0, len(ds), a.stride)))
    dl = DataLoader(sub, a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    gen = torch.Generator(device=dev).manual_seed(0)
    acc = {}
    for b in dl:
        x, t = prepare(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, flow = net(x)
        disp = sample_flow(flow.float(), t["uvh"], *x.shape[-2:])
        groups = ["real" if r in REAL else "procgen" for r in b["robot"] for _ in CAMS]
        cams = [c for _ in b["robot"] for c in CAMS]
        has = (t["w"].sum(1) >= 6).cpu()
        T_gt = t["T_gt"]
        rows = {"before": t["T_noisy"]}
        for name, (src, npx, kw) in VARIANTS.items():
            uv = t["uvt"] if src == "oracle" else t["uvh"] + disp
            if npx:
                uv = uv + torch.randn(uv.shape, device=dev, generator=gen) * npx
            rows[name], _ = solve_pose(t["T_noisy"], t["X"], uv, t["w"], t["cams"], **kw)
        match_err = ((t["uvh"] + disp - t["uvt"]).norm(dim=-1) * t["w"]).sum(1) / t["w"].sum(1).clamp_min(1)
        for name, T in rows.items():
            rot = torch.rad2deg(rot_angle(T[:, :3, :3], T_gt[:, :3, :3])).cpu()
            tr = ((T[:, :3, 3] - T_gt[:, :3, 3]).norm(dim=-1) * 100).cpu()
            lat, dep, roll, tilt = (v.cpu() for v in axis_err(T, T_gt))
            rp = reproj_px(T, t).cpu()
            for v in range(len(groups)):
                if not has[v]:
                    continue
                for key in ("all", groups[v], f"{groups[v]}/{cams[v]}", f"all/{cams[v]}"):
                    d = acc.setdefault(key, {}).setdefault(name, {})
                    for k_, val in (("rot", rot), ("trans", tr), ("lat", lat), ("depth", dep), ("roll", roll),
                                    ("tilt", tilt), ("reproj", rp)):
                        d.setdefault(k_, []).append(val[v].item())
                    if name == "before":
                        d.setdefault("match_px", []).append(match_err[v].item())
                        d.setdefault("n_match", []).append(t["w"][v].sum().item())
    out = {g: {n: {k: float(np.median(v)) for k, v in d.items()} | {"n": len(d["rot"])} for n, d in vs.items()}
           for g, vs in acc.items()}
    for g in ("all", "real", "procgen", "all/top", "all/front", "all/wrist"):
        if g not in out:
            continue
        print(f"\n== {g} (views {out[g]['before']['n']}, matches/view {out[g]['before']['n_match']:.0f}, "
              f"net match err {out[g]['before']['match_px']:.2f} px median)")
        print(f"  {'variant':22s} {'rot':>6s} {'trans':>6s} | {'lat':>5s} {'depth':>5s} cm | {'roll':>5s} {'tilt':>5s} deg | reproj px")
        for n, m in out[g].items():
            print(f"  {n:22s} {m['rot']:6.2f} {m['trans']:6.2f} | {m['lat']:5.2f} {m['depth']:5.2f}    | "
                  f"{m['roll']:5.2f} {m['tilt']:5.2f}     | {m['reproj']:.2f}")
    print(f"\n{(time.time() - t0) / 60:.1f} min")
    if a.out:
        json.dump(dict(ckpt=a.ckpt, stride=a.stride, corr=hyp["corr"], metrics=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
