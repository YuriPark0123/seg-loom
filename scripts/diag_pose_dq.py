"""Is the stage-1 pose floor (~3 deg) the approx-URDF joint zero offsets? (3090 finding, 2026-11-28)

Static top/front cameras of eval episodes, K frames spread over the episode. Correspondences: approx-URDF surface
points P (link-local) that are visible in the TRUE camera. Two kinds of target pixel:

  uvt      the current training target: approx point (approx FK at q_enc) projected with the true camera.
           Self-consistent with the approx model, so a camera-only solve recovers the camera exactly.
  surface  where that link point really is: TRUE FK at q_true, same link-local point, true camera (what an ideal
           matcher that looks at the image would report).

Solves (no camera prior unless noted, init = perturbed camera of frame 0):
  cam1     camera only, one frame (per frame, then median)
  camK     camera only, K frames stacked
  camK+dq_gt   camera only, K frames, approx FK at q_enc + dq_gt (dq_gt = -zero_deg of perturb.json)
  camK+dq  camera + per-joint zero offset dq (constant over the episode), K frames, Gaussian prior on dq
Reported: median rot deg / trans cm vs the true camera, and |dq - dq_gt| (deg) for the joint solve.

    python scripts/diag_pose_dq.py --episodes 40 --frames 10 --out results/pose_diag/dq.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from shadow.learn.camera import project  # noqa: E402
from shadow.learn.data import RobotBank, SegLoomFrames, q_names  # noqa: E402
from shadow.learn.geometry import inv_T, make_T, rot_angle, so3_exp, transform  # noqa: E402
from shadow.learn.points import splat_labels, world_points  # noqa: E402

CAMS = ("top", "front", "wrist")
dt = torch.float64


def err(T, T_gt):
    return (float(torch.rad2deg(rot_angle(T[None, :3, :3], T_gt[None, :3, :3]))[0]),
            float((T[:3, 3] - T_gt[:3, 3]).norm() * 100))


def residuals(th, T0, kin, names, q, dqmask, P, L, bidx, uv, w, cams):
    """th (S, 6 + J): camera correction xi and dq. q (K, J). P (K, M, 3) link-local, L (K, M) link (in bidx order).
    Returns r (S, K, M, 2), ok (S, K, M)."""
    S, (K, M) = th.shape[0], L.shape
    xi, dq = th[:, :6], th[:, 6:] * dqmask
    qq = (q[None] + dq[:, None]).reshape(S * K, -1)
    Tl = kin.fk(qq, names)[:, bidx].reshape(S, K, -1, 4, 4)
    T = torch.gather(Tl, 2, L[None, :, :, None, None].expand(S, K, M, 4, 4))  # (S, K, M, 4, 4)
    X = (T[..., :3, :3] @ P[None, ..., None])[..., 0] + T[..., :3, 3]
    Tc = T0[None] @ make_T(so3_exp(xi[:, :3]), xi[:, 3:])                       # (S, 4, 4) base<-cam
    pc = transform(inv_T(Tc), X.reshape(S, K * M, 3))
    c = {k: v[:1].expand(S, *v.shape[1:]) for k, v in cams.items()}
    p, ok = project(pc, c)
    return p.reshape(S, K, M, 2) - uv[None], ok.reshape(S, K, M) & (w[None] > 0)


def lm(T0, kin, names, q, dqmask, P, L, bidx, uv, w, cams, solve_dq, dq0=None, dq_sigma=np.deg2rad(3.0),
       iters=15, eps=1e-6):
    J = q.shape[1]
    th = torch.zeros(6 + J, dtype=dt)
    if dq0 is not None:
        th[6:] = dq0
    free = torch.cat([torch.ones(6, dtype=torch.bool), dqmask.bool() if solve_dq else torch.zeros(J, dtype=torch.bool)])
    fi = torch.nonzero(free)[:, 0]
    pri = torch.zeros(6 + J, dtype=dt)
    if solve_dq:
        pri[6:] = dqmask / dq_sigma ** 2
    lam = 1e-3

    def cost(r, ok, th_):
        return float((ok[..., None] * r ** 2).sum() + (pri * th_ ** 2).sum())

    for _ in range(iters):
        E = torch.zeros(len(fi), 6 + J, dtype=dt)
        E[torch.arange(len(fi)), fi] = eps
        ths = torch.cat([th[None], th[None] + E], 0)
        r, ok = residuals(ths, T0, kin, names, q, dqmask, P, L, bidx, uv, w, cams)
        okall = ok.all(0)
        r0 = (r[0] * okall[..., None]).reshape(-1)
        Jm = ((r[1:] - r[:1]) / eps * okall[None, ..., None]).reshape(len(fi), -1).T     # (N, F)
        H = Jm.T @ Jm + torch.diag(pri[fi])
        g = Jm.T @ r0 + pri[fi] * th[fi]
        step = -torch.linalg.solve(H + lam * torch.diag(H.diagonal().clamp_min(1e-9)), g)
        tn = th.clone()
        tn[fi] += step
        rn, okn = residuals(tn[None], T0, kin, names, q, dqmask, P, L, bidx, uv, w, cams)
        if cost(rn[0], okn[0] & okall, tn) < cost(r[0], okall, th):
            th, lam = tn, lam * 0.3
        else:
            lam *= 10
    return T0 @ make_T(so3_exp(th[None, :3]), th[None, 3:6])[0], th[6:]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--points", type=int, default=200)
    ap.add_argument("--rot-deg", type=float, default=3.0)
    ap.add_argument("--trans-m", type=float, default=0.03)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    ds = SegLoomFrames(a.data, scale=0.5, hypothesis=None, with_rgb=False)
    bank = RobotBank(0.003, "cpu")
    cands = []
    for i, (root, ep, robot, T) in enumerate(ds.episodes):
        meta = json.load(open(os.path.join(root, "episodes", ep, "meta.json")))
        st = [c for c in ("top", "front") if meta["cameras"][c].get("motion") == "static"]
        if st and T >= a.frames:
            cands.append((i, st[0]))
    random.Random(0).shuffle(cands)
    real = [c for c in cands if not ds.episodes[c[0]][2].startswith("pg")]
    cands = real[: a.episodes // 4] + [c for c in cands if ds.episodes[c[0]][2].startswith("pg")][: a.episodes - min(len(real), a.episodes // 4)]
    rng = np.random.default_rng(0)
    res = {}

    def add(grp, key, v):
        for g in ("all", grp):
            res.setdefault(g, {}).setdefault(key, []).append(v)

    for n_ep, (i, cam) in enumerate(cands):
        root, ep, robot, T = ds.episodes[i]
        meta = json.load(open(os.path.join(root, "episodes", ep, "meta.json")))
        A = dict(np.load(os.path.join(root, "episodes", ep, "arrays.npz")))
        R = bank.get(root, robot, meta["robot_bodies"])
        pert = json.load(open(os.path.join(root, "robots", robot, "perturb.json")))
        nq = A["q_enc"].shape[1]
        names = q_names(meta, nq)
        dqmask = torch.tensor([float(n in meta["arm_joints"]) for n in names], dtype=dt)
        dq_gt = torch.tensor([-np.deg2rad(pert["zero_deg"].get(n, 0.0)) for n in names], dtype=dt)
        ts = [int(t) for t in np.linspace(0, T - 1, a.frames).round()]
        item = ds[ds.items.index((i, ts[0]))]
        ci = CAMS.index(cam)
        T_gt = item["T_base_cam"][ci].to(dt)
        cams = dict(K=item["cam_K"][ci:ci + 1].to(dt), dist=item["cam_dist"][ci:ci + 1].to(dt),
                    kind=item["cam_kind"][ci:ci + 1], size=item["cam_size"][ci:ci + 1])
        W_, H_ = [int(x) for x in cams["size"][0]]
        w0 = rng.normal(0, np.deg2rad(a.rot_deg), 3)
        t0 = rng.normal(0, a.trans_m, 3)
        T0 = T_gt @ make_T(so3_exp(torch.tensor(w0)[None]), torch.tensor(t0)[None])[0]
        Pa, La = R["pts_approx"].to(dt), R["link_approx"]
        q_enc = torch.from_numpy(A["q_enc"][ts]).to(dt)
        q_true = torch.from_numpy(A["q_true"][ts]).to(dt)
        Tla = R["kin_approx"].fk(q_enc, names)[:, R["body_idx_approx"]]
        Tlt = R["kin_true"].fk(q_true, names)[:, R["body_idx_true"]]
        Xs = world_points(Tlt, Pa, La)                                  # (K, N, 3) approx pts on TRUE links
        Xa = world_points(Tla, Pa, La)
        Tcb = inv_T(T_gt)
        # visibility of the true-placed points in the true camera: z-buffer of the TRUE render points
        Pt, Lt = R["pts_true"].to(dt), R["link_true"]
        pct = transform(Tcb[None].expand(len(ts), 4, 4), world_points(Tlt, Pt, Lt))
        c3 = {k: v.expand(len(ts), *v.shape[1:]) for k, v in cams.items()}
        _, dep = splat_labels(pct.float(), Lt + 1, {k: (v.float() if v.is_floating_point() else v) for k, v in c3.items()},
                              H_, W_, 0.003, r_max=3)
        pcs = transform(Tcb[None].expand(len(ts), 4, 4), Xs)
        uvs, oks = project(pcs, c3)
        uvt, okt = project(transform(Tcb[None].expand(len(ts), 4, 4), Xa), c3)
        P = torch.zeros(len(ts), a.points, 3, dtype=dt); L = torch.zeros(len(ts), a.points, dtype=torch.long)
        US = torch.zeros(len(ts), a.points, 2, dtype=dt); UT = torch.zeros_like(US)
        Wt = torch.zeros(len(ts), a.points, dtype=dt)
        for k in range(len(ts)):
            u, v = uvs[k, :, 0].round().long(), uvs[k, :, 1].round().long()
            ins = oks[k] & okt[k] & (u >= 0) & (v >= 0) & (u < W_) & (v < H_)
            z = dep[k, v.clamp(0, H_ - 1), u.clamp(0, W_ - 1)].to(dt)
            vis = ins & (z > 0) & ((pcs[k, :, 2] - z).abs() < 0.01 + 0.02 * z)
            idx = torch.nonzero(vis)[:, 0]
            if len(idx) < 12:
                continue
            sel = idx[torch.from_numpy(rng.choice(len(idx), min(a.points, len(idx)), replace=False))]
            n = len(sel)
            P[k, :n], L[k, :n], US[k, :n], UT[k, :n], Wt[k, :n] = Pa[sel], La[sel], uvs[k, sel], uvt[k, sel], 1
        if (Wt.sum(1) > 0).sum() < 2:
            continue
        grp = "real" if not robot.startswith("pg") else "procgen"
        add(grp, "before", err(T0, T_gt))
        kin, bidx = R["kin_approx"], R["body_idx_approx"]
        for tgt, UV in (("uvt", UT), ("surface", US)):
            # one frame (first usable)
            k0 = int(torch.nonzero(Wt.sum(1) > 0)[0, 0])
            T1, _ = lm(T0, kin, names, q_enc[k0:k0 + 1], dqmask, P[k0:k0 + 1], L[k0:k0 + 1], bidx, UV[k0:k0 + 1],
                       Wt[k0:k0 + 1], cams, solve_dq=False)
            add(grp, f"{tgt}/cam1", err(T1, T_gt))
            TK, _ = lm(T0, kin, names, q_enc, dqmask, P, L, bidx, UV, Wt, cams, solve_dq=False)
            add(grp, f"{tgt}/camK", err(TK, T_gt))
            if tgt == "surface":
                Tg, _ = lm(T0, kin, names, q_enc, dqmask, P, L, bidx, UV, Wt, cams, solve_dq=False, dq0=dq_gt)
                add(grp, f"{tgt}/camK+dq_gt", err(Tg, T_gt))
                Tj, dq = lm(T0, kin, names, q_enc, dqmask, P, L, bidx, UV, Wt, cams, solve_dq=True)
                add(grp, f"{tgt}/camK+dq", err(Tj, T_gt))
                d = (dq - dq_gt)[dqmask > 0]
                add(grp, "dq_err_deg(joint solve)", (float(np.rad2deg(d.abs().median())), 0.0))
                add(grp, "dq_gt_deg", (float(np.rad2deg(dq_gt[dqmask > 0].abs().median())), 0.0))
        print(f"[{n_ep + 1}/{len(cands)}] {robot} {cam}: " + "  ".join(
            f"{k} {v[-1][0]:.2f}/{v[-1][1]:.2f}" for k, v in res[grp].items()), flush=True)
    out = {g: {k: dict(rot=float(np.median([x[0] for x in v])), trans=float(np.median([x[1] for x in v])), n=len(v))
               for k, v in d.items()} for g, d in res.items()}
    for g in ("all", "real", "procgen"):
        if g in out:
            print(f"\n== {g}")
            for k, m in out[g].items():
                print(f"  {k:26s} rot {m['rot']:5.2f} deg  trans {m['trans']:5.2f} cm  (n={m['n']})")
    if a.out:
        json.dump(dict(args=vars(a), metrics=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
