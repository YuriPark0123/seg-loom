"""Sequential (video-order) evaluation of a stage-1 checkpoint.

Every episode is played from frame 0 in order; a method may use only the past (frames <= t) and starts over at a
new episode. Methods:

  independent   the current single-frame evaluation: every frame gets its own fresh perturbation of the TRUE
                camera as hypothesis (baseline; nothing carries over, so its curves should be flat)
  track         frame 0 as above; frame t's hypothesis camera = the estimate of frame t-1 (top/front), or for the
                wrist camera the estimate moved with the end effector (approx URDF FK at q_enc, t-1 -> t). Errors
                carry over: after a camera jump the hypothesis is off by the jump.
  track_mem     track + memory for the pose: the camera is solved jointly from the matches of the last --window
                frames (in the anchor frame: base for top/front, end effector for the wrist camera, so that
                rigidly mounted cameras are constant), reset when the current frame disagrees (jump detection).
  track_dq      track_mem for all cameras at once + the approx URDF's joint zero offsets dq (one per arm joint, the
                same for the whole episode; register.solve_pose_dq): the wrist anchor moves with q + dq too. Meant
                for --flow-target surface models (their matches show where the links really are).

Segmentation comes from the single-frame network in all three (no recurrent model yet): its curve only moves through
the hypothesis it is given. Reported per eval set, method and camera-motion type (meta cameras.<cam>.motion:
static / drift / shake / jump, wrist: mounted):
  IoU@0, @5, @10, @20, @last (amodal and visible), gain = last - first, frames to 95 % of the final value,
  mean IoU and pose-error curves over t, curves over the cumulative joint motion sum_t |dq_true| (arm joints),
  "still prefix" frames (robot not moved yet, t >= 1) vs frame 0, and for jump cameras the error just before,
  at and after the jump frame (discontinuity of T_base_cam) and the frames until it recovers.

    python scripts/eval_sequential.py out/stage1_main/stages/stage3.5_final_step209628.pt \\
        --set eval=data/eval --set eval_real=data/hf/eval_real/i000 --out out/seq/final
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from shadow.learn.data import SegLoomFrames, collate, q_names  # noqa: E402
from shadow.learn.fk import URDFKinematics  # noqa: E402
from shadow.learn.geometry import inv_T, make_T, rot_angle, so3_exp, transform  # noqa: E402
from shadow.learn.camera import project  # noqa: E402
from shadow.learn.register import solve_pose, solve_pose_dq  # noqa: E402
from train_stage1 import CAMS, POSE_PRIOR, PRIOR_PX, Stage1Net, is_real, prepare, sample_flow  # noqa: E402

METHODS = ("independent", "track", "track_mem", "track_dq")
_DS = None
RESETS = [0, 0]                                       # resets, window solves (diagnostics)


def _work(job):
    idx, T_init = job
    return _DS.item_with_init(idx, None if T_init is None else torch.from_numpy(T_init))


class Episode:
    """Per-episode state: frame indices, camera motion types, anchors (approx FK of the wrist-camera body)."""

    def __init__(self, ds, i, start):
        root, ep, robot, T = ds.episodes[i]
        self.root, self.ep, self.robot, self.T, self.start = root, ep, robot, T, start
        meta = json.load(open(os.path.join(root, "episodes", ep, "meta.json")))
        A = np.load(os.path.join(root, "episodes", ep, "arrays.npz"))
        self.mode = meta["mode"]
        self.motion = [meta["cameras"][c].get("motion", "?") for c in CAMS]
        self.T_gt = torch.from_numpy(np.stack([A[f"T_base_cam_{c}"] for c in CAMS], 1)).double()   # (T, C, 4, 4)
        na = len(meta["arm_joints"])
        dq = np.abs(np.diff(A["q_true"][:, :na], axis=0)).sum(1)
        self.cum_motion = np.r_[0.0, np.cumsum(dq)]                                                  # rad
        spec = yaml.safe_load(open(os.path.join(root, "robots", robot, "spec.yaml")))
        kin = URDFKinematics.from_file(os.path.join(root, "robots", robot, "approx.urdf"))
        ee = spec.get("ee_body")
        li = kin.links.index(ee) if ee in kin.links else len(kin.links) - 1
        q = torch.from_numpy(A["q_enc"]).double()
        F = kin.fk(q, q_names(meta, q.shape[1]))[:, li]                                             # (T, 4, 4)
        eye = torch.eye(4, dtype=torch.float64).expand(self.T, 4, 4)
        self.anchor = torch.stack([eye if c != "wrist" else F for c in CAMS], 1)                     # (T, C, 4, 4)
        self.kin, self.names, self.ee = kin, q_names(meta, q.shape[1]), li
        self.bidx = [kin.links.index(b) for b in meta["robot_bodies"]]
        # dq of the FIRST arm joint (vertical base axis) is fixed at 0: from images it is indistinguishable from all
        # top/front cameras turning about that axis (3090 note 2026-11-28); control only needs the robot's own frame
        self.dqmask = torch.tensor([float(n in meta["arm_joints"][1:]) for n in self.names], dtype=torch.float64)
        pj = os.path.join(root, "robots", robot, "perturb.json")
        zd = json.load(open(pj)).get("zero_deg", {}) if os.path.exists(pj) else {}
        self.dq_gt = torch.tensor([-np.deg2rad(zd.get(n, 0.0)) for n in self.names], dtype=torch.float64)
        # the robot's believed base frame: the true one turned about joint 1's axis by its zero offset (approx(q) =
        # true(q + zero)); errors vs T_bel @ T_gt leave out that unobservable component (checked with oracle matches)
        from shadow.learn.fk import rpy_to_R
        j = kin.joints[kin.joint_index[meta["arm_joints"][0]]]
        Tp = kin.fk(torch.zeros(1, q.shape[1], dtype=torch.float64), self.names)[0, kin.links.index(j.parent)]
        Tj = Tp @ make_T(rpy_to_R(torch.tensor(j.rpy, dtype=torch.float64))[None], torch.tensor(j.xyz, dtype=torch.float64)[None])[0]
        ax = Tj[:3, :3] @ torch.tensor(j.axis / np.linalg.norm(j.axis), dtype=torch.float64)
        th = -float(self.dq_gt[self.names.index(j.name)]) if j.name in self.names else 0.0
        Rb = so3_exp((ax * th)[None])[0]
        self.T_bel = torch.eye(4, dtype=torch.float64)
        self.T_bel[:3, :3], self.T_bel[:3, 3] = Rb, Tj[:3, 3] - Rb @ Tj[:3, 3]
        self.kin_true = URDFKinematics.from_file(os.path.join(root, "robots", robot, "true.urdf"))
        self.bidx_true = [self.kin_true.links.index(b) for b in meta["robot_bodies"]]
        # jump frame per camera: the largest consecutive pose change, if clearly a discontinuity
        self.jump = []
        for ci in range(len(CAMS)):
            if self.motion[ci] != "jump":
                self.jump.append(None); continue
            Tg = self.T_gt[:, ci]
            dtr = (Tg[1:, :3, 3] - Tg[:-1, :3, 3]).norm(dim=-1)
            drt = torch.rad2deg(rot_angle(Tg[1:, :3, :3], Tg[:-1, :3, :3]))
            score = dtr * 100 + drt                                                                  # cm + deg
            j = int(score.argmax())
            ok = score[j] > 3.0 and score[j] > 5 * score.median()
            self.jump.append(j + 1 if ok else None)


def solve_window(mem, X, uv, w, cams, T_init, A_t, window, reset_px):
    """Joint pose from the window's matches in the anchor frame. mem: list of (X_anchor, uv, w) of past frames."""
    RESETS[1] += 1
    Xa = transform(inv_T(A_t)[None], X[None])[0]
    mem.append((Xa, uv, w))
    del mem[:-window]
    T_a_init = inv_T(A_t) @ T_init
    one = lambda Xs, uvs, ws, Ti: solve_pose(Ti[None], Xs[None], uvs[None], ws[None], {k: v[None] for k, v in cams.items()},
                                             iters=15, prior_sigma=POSE_PRIOR, prior_px=PRIOR_PX)[0][0]
    Ta = one(torch.cat([m[0] for m in mem]), torch.cat([m[1] for m in mem]), torch.cat([m[2] for m in mem]), T_a_init)
    if len(mem) > 1:                                  # does the current frame agree with the window solution?
        p, ok = project(transform(inv_T(Ta)[None], Xa[None]), {k: v[None] for k, v in cams.items()})
        r = ((p[0] - uv).norm(dim=-1))[(w > 0) & ok[0]]
        if r.numel() >= 6 and r.median() > reset_px:
            RESETS[0] += 1
            del mem[:-1]
            Ta = one(Xa, uv, w, T_a_init)
    return A_t @ Ta


def reproj_surface(grp, act, tg, b, T_out, dqs, dev):
    """Median px between where the robot's own model (approx URDF at q_enc + dq estimate) puts each match point through
    the estimated camera and where that link point really is in the image (true FK, true camera, true lens). The
    control-relevant alignment; blind to the joint-1 / camera ambiguity. NaN for views without matches."""
    C = len(CAMS)
    out = torch.full((T_out.shape[0],), float("nan"))
    for n, k in enumerate(act):
        e = grp[k]
        nq = len(e.names)
        dq = dqs[k].dq.cpu() if dqs is not None else torch.zeros(nq, dtype=torch.float64)
        qe = b["q_enc"][n, :nq].double()[None]
        qt = b["q_true"][n, :nq].double()[None]
        Ta = e.kin.fk(qe + dq[None], e.names)[0][e.bidx]
        Tt = e.kin_true.fk(qt, e.names)[0][e.bidx_true]
        for ci in range(C):
            v = n * C + ci
            w = tg["w"][v].cpu() > 0
            if w.sum() < 6:
                continue
            P, L = tg["P"][v].cpu().double(), tg["L"][v].cpu()
            Xa = (Ta[L][:, :3, :3] @ P[..., None])[..., 0] + Ta[L][:, :3, 3]
            Xt = (Tt[L][:, :3, :3] @ P[..., None])[..., 0] + Tt[L][:, :3, 3]
            cn = {kk: vv[v:v + 1].cpu().double() if vv.is_floating_point() else vv[v:v + 1].cpu() for kk, vv in tg["cams"].items()}
            ct = dict(cn, K=b["cam_K_true"][n, ci:ci + 1].double(), dist=b["cam_dist_true"][n, ci:ci + 1].double())
            pa, oka = project(transform(inv_T(T_out[v:v + 1].cpu()), Xa[None]), cn)
            pt, okt = project(transform(inv_T(tg["T_gt"][v:v + 1].cpu().double()), Xt[None]), ct)
            m = w & oka[0] & okt[0]
            if m.sum() >= 6:
                out[v] = (pa[0] - pt[0]).norm(dim=-1)[m].median()
    return out


class DqState:
    """track_dq memory of one episode: window entries per view, current anchor<-camera estimates and dq."""

    def __init__(self, e, dev):
        self.e, self.dev = e, dev
        self.mem = []                                 # (t, cam, P, L, q, uv, w, cams)
        self.Ta = [None] * len(CAMS)
        self.dq = torch.zeros(len(e.names), dtype=torch.float64, device=dev)

    def anchor(self, ci, q):
        if CAMS[ci] != "wrist":
            return torch.eye(4, dtype=torch.float64, device=self.dev)
        return self.e.kin.fk((q + self.dq)[None], self.e.names)[0, self.e.ee].to(self.dev)

    def step(self, t, views, window, reset_px):
        """views: list of (ci, P, L, q, uv, w, cams, T_hyp base<-cam). Returns {ci: base<-cam estimate}."""
        RESETS[1] += 1
        for ci, P, L, q, uv, w, cams, T_hyp in views:
            if self.Ta[ci] is None:
                self.Ta[ci] = inv_T(self.anchor(ci, q)[None])[0] @ T_hyp
            self.mem.append((t, ci, P, L, q, uv, w, cams))
        self.mem = [m for m in self.mem if m[0] > t - window]
        out = self._solve()
        # jump detection per camera: does the current view agree with the window solution?
        for ci, P, L, q, uv, w, cams, T_hyp in views:
            n = [k for k, m in enumerate(self.mem) if m[0] == t and m[1] == ci][0]
            if sum(1 for m in self.mem if m[1] == ci) < 2:
                continue
            r = self._resid(n, out[n])
            if r is not None and r > reset_px:
                RESETS[0] += 1
                self.mem = [m for m in self.mem if m[1] != ci or m[0] == t]
                self.Ta[ci] = inv_T(self.anchor(ci, q)[None])[0] @ T_hyp
                out = self._solve()
        res = {}
        for k, m in enumerate(self.mem):
            if m[0] == t:
                res[m[1]] = out[k]
        return res

    def _solve(self):
        cs = sorted({m[1] for m in self.mem})
        idx = {c: k for k, c in enumerate(cs)}
        st = lambda j: torch.stack([m[j] for m in self.mem])
        cams = {k: torch.stack([m[7][k] for m in self.mem]) for k in self.mem[0][7]}
        Ta, dq, Tv = solve_pose_dq(torch.stack([self.Ta[c] for c in cs]), st(2), st(3), st(4), self.e.kin, self.e.names,
                                   self.e.bidx, st(5), st(6), cams, torch.tensor([idx[m[1]] for m in self.mem]),
                                   [self.e.ee if CAMS[c] == "wrist" else None for c in cs], self.e.dqmask.to(self.dev),
                                   dq0=self.dq, iters=8, prior_sigma=POSE_PRIOR, prior_px=PRIOR_PX)
        if torch.isfinite(Ta).all() and torch.isfinite(dq).all():
            for k, c in enumerate(cs):
                self.Ta[c] = Ta[k]
            self.dq = dq
        return Tv

    def _resid(self, n, T):
        t, ci, P, L, q, uv, w, cams = self.mem[n]
        Tl = self.e.kin.fk((q + self.dq)[None], self.e.names)[0][self.e.bidx].to(self.dev)
        X = (Tl[L][:, :3, :3] @ P[..., None])[..., 0] + Tl[L][:, :3, 3]
        p, ok = project(transform(inv_T(T)[None], X[None]), {k: v[None] for k, v in cams.items()})
        r = (p[0] - uv).norm(dim=-1)[(w > 0) & ok[0]]
        return float(r.median()) if r.numel() >= 6 else None


@torch.no_grad()
def run_set(net, ds, eps, method, pool, dev, a):
    """Returns records: per (episode, cam, t) dict."""
    recs = []
    for g0 in range(0, len(eps), a.group):
        grp = eps[g0:g0 + a.group]
        est = [None] * len(grp)                       # (C, 4, 4) last estimate
        mems = [[[] for _ in CAMS] for _ in grp]
        dqs = [DqState(e, dev) for e in grp] if method == "track_dq" else None
        for t in range(max(e.T for e in grp)):
            act = [k for k, e in enumerate(grp) if t < e.T]
            jobs, base = [], {}
            for k in act:
                e = grp[k]
                if method == "independent" or t == 0:
                    jobs.append((e.start + t, None))
                else:
                    Ti = e.anchor[t] @ inv_T(e.anchor[t - 1]) @ est[k]                 # moves the wrist cam with the EE
                    base[k] = Ti.clone()
                    if a.dither > 0:                  # random perturbation around our own estimate (bias averaging)
                        g = np.random.default_rng([e.start, t, 7])
                        w_ = torch.from_numpy(g.normal(0, np.deg2rad(3.0) * a.dither, (len(CAMS), 3)))
                        sc = np.array([0.25 if c == "wrist" else 1.0 for c in CAMS])[:, None]
                        tr_ = torch.from_numpy(g.normal(0, 0.03 * a.dither, (len(CAMS), 3)) * sc)
                        Ti = Ti @ make_T(so3_exp(w_), tr_)
                    jobs.append((e.start + t, Ti.float().numpy()))
            items = pool.map(_work, jobs, chunksize=1)
            b = collate(items)
            x, tg = prepare(b, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                seg, flow = net(x)
            seg = seg.float()
            disp = sample_flow(flow.float(), tg["uvh"], *x.shape[-2:])
            uv = tg["uvh"] + disp
            T_single, _ = solve_pose(tg["T_noisy"], tg["X"], uv, tg["w"], tg["cams"], prior_sigma=POSE_PRIOR,
                                     prior_px=PRIOR_PX)
            T_out = T_single.double().clone()
            C = len(CAMS)
            if method == "track_mem":
                for n, k in enumerate(act):
                    e = grp[k]
                    for ci in range(C):
                        v = n * C + ci
                        if tg["w"][v].sum() < 6:
                            continue
                        cams = {kk: vv[v].double() for kk, vv in tg["cams"].items()}
                        T_out[v] = solve_window(mems[k][ci], tg["X"][v].double(), uv[v].double(), tg["w"][v].double(),
                                                cams, base[k][ci].to(dev) if k in base else tg["T_noisy"][v].double(),
                                                e.anchor[t, ci].to(dev), a.window,
                                                a.reset_px)
            if method == "track_dq":
                for n, k in enumerate(act):
                    e = grp[k]
                    vs = []
                    for ci in range(C):
                        v = n * C + ci
                        if tg["w"][v].sum() < 6:
                            continue
                        cams = {kk: vv[v].double() for kk, vv in tg["cams"].items()}
                        nq = len(e.names)
                        vs.append((ci, tg["P"][v].double(), tg["L"][v], tg["q_enc"][v, :nq].double(), uv[v].double(),
                                   tg["w"][v].double(), cams,
                                   base[k][ci].to(dev) if k in base else tg["T_noisy"][v].double()))
                    if vs:
                        for ci, T in dqs[k].step(t, vs, a.dq_window, a.reset_px).items():
                            T_out[n * C + ci] = T
            bad = ~torch.isfinite(T_out).all(-1).all(-1)
            T_out[bad] = tg["T_noisy"][bad].double()
            vb = tg["valid"].bool()
            am_p, vs_p = (seg[:, 0] > 0) & vb, (seg[:, 1] > 0) & vb
            am_y, vs_y = tg["am"].bool() & vb, tg["vis"].bool() & vb
            ia, ua = (am_p & am_y).sum((1, 2)), (am_p | am_y).sum((1, 2))
            iv, uvv = (vs_p & vs_y).sum((1, 2)), (vs_p | vs_y).sum((1, 2))
            Tgt = tg["T_gt"].double()
            rot = torch.rad2deg(rot_angle(T_out[:, :3, :3], Tgt[:, :3, :3])).cpu()
            tr = ((T_out[:, :3, 3] - Tgt[:, :3, 3]).norm(dim=-1) * 100).cpu()
            Tb = torch.stack([grp[k].T_bel for k in act]).to(dev).repeat_interleave(C, 0) @ Tgt
            rot_b = torch.rad2deg(rot_angle(T_out[:, :3, :3], Tb[:, :3, :3])).cpu()
            tr_b = ((T_out[:, :3, 3] - Tb[:, :3, 3]).norm(dim=-1) * 100).cpu()
            rp = reproj_surface(grp, act, tg, b, T_out, dqs, dev)
            rot0 = torch.rad2deg(rot_angle(tg["T_noisy"][:, :3, :3].double(), Tgt[:, :3, :3])).cpu()
            for n, k in enumerate(act):
                e = grp[k]
                est[k] = T_out[n * C:(n + 1) * C].cpu()
                for ci in range(C):
                    v = n * C + ci
                    recs.append(dict(ep=e.ep, robot=e.robot, real=is_real(e.robot), mode=e.mode, cam=CAMS[ci],
                                     motion=e.motion[ci], t=t, T=e.T, cum=float(e.cum_motion[t]), jump=e.jump[ci],
                                     ia=int(ia[v]), ua=int(ua[v]), iv=int(iv[v]), uv=int(uvv[v]),
                                     rot=float(rot[v]), tr=float(tr[v]), rot_init=float(rot0[v]),
                                     rot_b=float(rot_b[v]), tr_b=float(tr_b[v]), rp=float(rp[v]),
                                     has=bool(tg["w"][v].sum() >= 6)))
                    if dqs is not None:
                        d = (dqs[k].dq.cpu() - e.dq_gt)[e.dqmask > 0]
                        recs[-1]["dq_err"] = float(torch.rad2deg(d.abs()).median())
        print(f"    {method}: {min(g0 + a.group, len(eps))}/{len(eps)} episodes"
              + (f", window resets {RESETS[0]}/{RESETS[1]}" if method in ("track_mem", "track_dq") else ""), flush=True)
    return recs


# --------------------------------------------------------------------------------------------- aggregation
CUM_BINS = [0, 0.01, 0.1, 0.25, 0.5, 1, 2, 4, 8, 1e9]


def pooled(rs, i="ia", u="ua"):
    s_u = sum(r[u] for r in rs)
    return sum(r[i] for r in rs) / s_u if s_u else float("nan")


def med(rs, k):
    v = [r[k] for r in rs if r["has"] and k in r and r[k] == r[k]]
    return float(np.median(v)) if v else float("nan")


def summarize(recs):
    by_t = {}
    for r in recs:
        by_t.setdefault(r["t"], []).append(r)
    ts = sorted(by_t)
    last = [r for r in recs if r["t"] == r["T"] - 1]
    curve = {k: [pooled(by_t[t], *k2) for t in ts] for k, k2 in (("iou_amodal", ("ia", "ua")), ("iou_visible", ("iv", "uv")))}
    curve |= {k: [med(by_t[t], k) for t in ts] for k in ("rot", "tr", "rot_b", "tr_b", "rp")}
    out = dict(n_views=len({(r["ep"], r["cam"]) for r in recs}), t=ts, curve=curve)
    for k, (i, u) in (("iou_amodal", ("ia", "ua")), ("iou_visible", ("iv", "uv"))):
        at = {f"@{t}": pooled(by_t[t], i, u) for t in (0, 5, 10, 20) if t in by_t}
        at["@last"] = pooled(last, i, u)
        at["gain"] = at["@last"] - at["@0"]
        c = curve[k]
        fin = at["@last"]
        at["t95"] = next((t for t, v in zip(ts, c) if v >= 0.95 * fin), None) if fin == fin else None
        out[k] = at
    for k in ("rot", "tr", "rot_b", "tr_b", "rp"):
        at = {f"@{t}": med(by_t[t], k) for t in (0, 5, 10, 20) if t in by_t}
        at["@last"] = med(last, k)
        at["gain"] = at["@last"] - at["@0"]
        out[k] = at
    # cumulative joint motion
    cb = []
    for lo, hi in zip(CUM_BINS[:-1], CUM_BINS[1:]):
        rs = [r for r in recs if lo <= r["cum"] < hi]
        if rs:
            cb.append(dict(lo=lo, hi=hi, n=len(rs), iou_amodal=pooled(rs), iou_visible=pooled(rs, "iv", "uv"),
                           rot=med(rs, "rot"), tr=med(rs, "tr")))
    out["by_cum_motion"] = cb
    dqe = [r["dq_err"] for r in last if "dq_err" in r]
    if dqe:
        out["dq_err_deg_last"] = float(np.median(dqe))                # track_dq: |dq - dq_gt| at the last frame
    still = [r for r in recs if r["t"] >= 1 and r["cum"] < 0.01]
    eps_still = {(r["ep"], r["cam"]) for r in still}
    first = [r for r in recs if r["t"] == 0 and (r["ep"], r["cam"]) in eps_still]
    out["still_prefix"] = dict(n=len(still), iou_amodal_t0=pooled(first), iou_amodal_still=pooled(still),
                               rot_t0=med(first, "rot"), rot_still=med(still, "rot"),
                               tr_t0=med(first, "tr"), tr_still=med(still, "tr"))
    # camera jumps
    jumps = {}
    for r in recs:
        if r["jump"] is not None:
            jumps.setdefault((r["ep"], r["cam"]), {})[r["t"]] = r
    J = []
    for key, fr in jumps.items():
        j = next(iter(fr.values()))["jump"]
        if j - 1 not in fr or j not in fr:
            continue
        pre = fr[j - 1]
        rec_t = None
        for t in sorted(fr):
            if t >= j and fr[t]["rot"] <= max(1.25 * pre["rot"], pre["rot"] + 0.5) and \
                    fr[t]["tr"] <= max(1.25 * pre["tr"], pre["tr"] + 0.5):
                rec_t = t - j; break
        J.append(dict(pre_rot=pre["rot"], at_rot=fr[j]["rot"], pre_tr=pre["tr"], at_tr=fr[j]["tr"],
                      pre_iou=pre["ia"] / max(pre["ua"], 1), at_iou=fr[j]["ia"] / max(fr[j]["ua"], 1),
                      post5_rot=fr[j + 5]["rot"] if j + 5 in fr else None, recover=rec_t))
    if J:
        rec = [x["recover"] for x in J]
        out["jump"] = dict(n=len(J), **{k: float(np.median([x[k] for x in J if x[k] is not None]))
                                        for k in ("pre_rot", "at_rot", "post5_rot", "pre_tr", "at_tr", "pre_iou", "at_iou")},
                           recovered_frac=float(np.mean([x is not None for x in rec])),
                           recover_frames_median=float(np.median([x for x in rec if x is not None])) if any(
                               x is not None for x in rec) else None)
    return out


def groups(recs):
    g = {"all": recs}
    for m in ("static", "drift", "shake", "jump", "mounted"):
        rs = [r for r in recs if r["motion"] == m]
        if rs:
            g[m] = rs
    for kind in ("real", "procgen"):
        rs = [r for r in recs if r["real"] == (kind == "real")]
        if rs and len(rs) != len(recs):
            g[kind] = rs
    for md in ("task", "motion"):
        rs = [r for r in recs if r["mode"] == md]
        if rs:
            g[f"mode_{md}"] = rs
    return g


def plot(summ, path, set_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    cols = {"independent": "#4C72B0", "track": "#DD8452", "track_mem": "#55A868", "track_dq": "#C44E52"}
    for m, S in summ.items():
        for gi, (g, ls) in enumerate((("all", "-"), ("jump", "--"))):
            if g not in S:
                continue
            c = S[g]["curve"]
            ax[0, 0].plot(S[g]["t"], c["iou_amodal"], ls, color=cols[m], label=f"{m} / {g}")
            ax[0, 1].plot(S[g]["t"], c["rot"], ls, color=cols[m], label=f"{m} / {g}")
            ax[0, 2].plot(S[g]["t"], c["tr"], ls, color=cols[m], label=f"{m} / {g}")
        cb = S["all"]["by_cum_motion"]
        xs = [np.sqrt(max(b["lo"], 1e-3) * min(b["hi"], 16)) for b in cb]
        ax[1, 0].plot(xs, [b["iou_amodal"] for b in cb], "o-", color=cols[m], label=m)
        ax[1, 1].plot(xs, [b["rot"] for b in cb], "o-", color=cols[m], label=m)
        ax[1, 2].plot(xs, [b["tr"] for b in cb], "o-", color=cols[m], label=m)
    for a_, yl in zip(ax[0], ("IoU amodal", "rot err (deg, median)", "trans err (cm, median)")):
        a_.set_xlabel("frame t"); a_.set_ylabel(yl); a_.grid(alpha=0.3)
    for a_, yl in zip(ax[1], ("IoU amodal", "rot err (deg, median)", "trans err (cm, median)")):
        a_.set_xscale("log"); a_.set_xlabel("cumulative joint motion (rad)"); a_.set_ylabel(yl); a_.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
    fig.suptitle(f"{set_name}: sequential evaluation (solid: all cameras, dashed: jump cameras)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fmt(x, p=3):
    return "-" if x is None or x != x else f"{x:.{p}f}"


def main():
    global _DS
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--set", action="append", default=None, help="name=root[,root...] (repeatable)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--episodes", type=int, default=0, help="episodes per set (0 = all), evenly spread")
    ap.add_argument("--group", type=int, default=32, help="episodes played in lockstep")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--dq-window", type=int, default=20, help="track_dq window (frames, all cameras)")
    ap.add_argument("--reset-px", type=float, default=6.0)
    ap.add_argument("--dither", type=float, default=0.0, help="track*: hypothesis = estimate + this x the training "
                    "perturbation (3 deg, 3 cm), fresh every frame")
    ap.add_argument("--lens-noise", type=float, default=None, help="lens error level (default: as trained)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True, help="output prefix (writes <out>_<set>.json/.png, <out>.md)")
    a = ap.parse_args()
    sets = a.set or ["eval=data/eval", "eval_real=data/hf/eval_real/i000"]
    dev = "cuda"
    s = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta = s["args"]
    net = Stage1Net(pretrained=False).to(dev).eval()
    net.load_state_dict(s["net"])
    hyp = dict(rot_deg=ta["rot_deg"], trans_m=ta["trans_m"], spacing=ta["spacing"], r_max=ta["r_max"], corr=ta["corr"],
               target=ta.get("flow_target", "approx"), lens_noise=ta.get("lens_noise", 0.0) if a.lens_noise is None else a.lens_noise)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    md = [f"# Sequential evaluation: `{os.path.basename(a.ckpt)}` (step {s['step']})", "",
          f"methods: {a.methods}; window {a.window}, reset {a.reset_px} px; prior {POSE_PRIOR} x {PRIOR_PX} px", ""]
    for spec in sets:
        name, roots = spec.split("=", 1)
        t0 = time.time()
        _DS = SegLoomFrames(roots.split(","), scale=ta["scale"], hypothesis=hyp, seed=20261126, deterministic=True)
        starts = np.r_[0, np.cumsum([e[3] for e in _DS.episodes])[:-1]]
        sel = range(len(_DS.episodes))
        if a.episodes and a.episodes < len(_DS.episodes):
            sel = np.linspace(0, len(_DS.episodes) - 1, a.episodes).round().astype(int)
        eps = [Episode(_DS, int(i), int(starts[i])) for i in sel]
        print(f"[{name}] {len(eps)} episodes", flush=True)
        summ, raw = {}, {}
        with mp.get_context("fork").Pool(a.workers) as pool:
            for m in a.methods.split(","):
                recs = run_set(net, _DS, eps, m, pool, dev, a)
                raw[m] = recs
                summ[m] = {g: summarize(rs) for g, rs in groups(recs).items()}
        json.dump(dict(ckpt=a.ckpt, step=s["step"], set=name, roots=roots, episodes=len(eps), summary=summ),
                  open(f"{a.out}_{name}.json", "w"))
        plot(summ, f"{a.out}_{name}.png", name)
        md += [f"## {name} ({roots}, {len(eps)} episodes, {(time.time() - t0) / 60:.1f} min)", "",
               "| method | group | views | IoU@0 | @5 | @10 | @20 | @last | gain | t95 | rot@0 | rot@last | trans@0 | trans@last "
               "| rot*@last | trans*@last | reproj px @0 | @last |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, S in summ.items():
            for g, r in S.items():
                i_ = r["iou_amodal"]
                md.append(f"| {m} | {g} | {r['n_views']} | {fmt(i_['@0'])} | {fmt(i_.get('@5'))} | {fmt(i_.get('@10'))} | "
                          f"{fmt(i_.get('@20'))} | {fmt(i_['@last'])} | {fmt(i_['gain'])} | {i_['t95']} | "
                          f"{fmt(r['rot']['@0'], 2)} | {fmt(r['rot']['@last'], 2)} | {fmt(r['tr']['@0'], 2)} | {fmt(r['tr']['@last'], 2)} | "
                          f"{fmt(r['rot_b']['@last'], 2)} | {fmt(r['tr_b']['@last'], 2)} | {fmt(r['rp']['@0'], 2)} | {fmt(r['rp']['@last'], 2)} |")
        md += ["", "rot* / trans*: vs the robot's believed base frame (true base turned by joint 1's zero offset, which "
               "images cannot tell from the cameras turning); reproj px: robot model (approx URDF at q_enc + dq) through "
               "the estimated camera vs where the link points really are (true FK / camera / lens)."]
        for m, S in summ.items():
            if "dq_err_deg_last" in S.get("all", {}):
                md.append(f"\n{m}: joint zero offset error at the last frame |dq - dq_gt| median "
                          f"{S['all']['dq_err_deg_last']:.2f} deg (per arm joint)")
        md += ["", "Still prefix (robot not moved yet, t >= 1) vs frame 0 of the same views:", "",
               "| method | n | IoU t0 | IoU still | rot t0 | rot still | trans t0 | trans still |", "|---|---|---|---|---|---|---|---|"]
        for m, S in summ.items():
            p = S["all"]["still_prefix"]
            md.append(f"| {m} | {p['n']} | {fmt(p['iou_amodal_t0'])} | {fmt(p['iou_amodal_still'])} | {fmt(p['rot_t0'], 2)} | "
                      f"{fmt(p['rot_still'], 2)} | {fmt(p['tr_t0'], 2)} | {fmt(p['tr_still'], 2)} |")
        md += ["", "Camera jumps (median over jump cameras; recovery = rot and trans back within 1.25x of the frame before):", "",
               "| method | n | rot before | rot at | rot +5 | trans before | trans at | IoU before | IoU at | recovered | frames to recover |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, S in summ.items():
            j = S["all"].get("jump")
            if j:
                md.append(f"| {m} | {j['n']} | {fmt(j['pre_rot'], 2)} | {fmt(j['at_rot'], 2)} | {fmt(j['post5_rot'], 2)} | "
                          f"{fmt(j['pre_tr'], 2)} | {fmt(j['at_tr'], 2)} | {fmt(j['pre_iou'])} | {fmt(j['at_iou'])} | "
                          f"{j['recovered_frac']:.0%} | {fmt(j['recover_frames_median'], 0)} |")
        md += ["", "By cumulative joint motion (all cameras):", "", "| method | motion (rad) | n | IoU amodal | rot | trans |",
               "|---|---|---|---|---|---|"]
        for m, S in summ.items():
            for b in S["all"]["by_cum_motion"]:
                md.append(f"| {m} | {b['lo']:g}-{b['hi'] if b['hi'] < 1e8 else 'inf'} | {b['n']} | {fmt(b['iou_amodal'])} | "
                          f"{fmt(b['rot'], 2)} | {fmt(b['tr'], 2)} |")
        md.append("")
        open(f"{a.out}.md", "w").write("\n".join(md))
        print("\n".join(md[-60:]), flush=True)


if __name__ == "__main__":
    main()
