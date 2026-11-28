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
from shadow.learn.register import solve_pose  # noqa: E402
from train_stage1 import CAMS, POSE_PRIOR, PRIOR_PX, Stage1Net, is_real, prepare, sample_flow  # noqa: E402

METHODS = ("independent", "track", "track_mem")
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


@torch.no_grad()
def run_set(net, ds, eps, method, pool, dev, a):
    """Returns records: per (episode, cam, t) dict."""
    recs = []
    for g0 in range(0, len(eps), a.group):
        grp = eps[g0:g0 + a.group]
        est = [None] * len(grp)                       # (C, 4, 4) last estimate
        mems = [[[] for _ in CAMS] for _ in grp]
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
                                     has=bool(tg["w"][v].sum() >= 6)))
        print(f"    {method}: {min(g0 + a.group, len(eps))}/{len(eps)} episodes"
              + (f", window resets {RESETS[0]}/{RESETS[1]}" if method == "track_mem" else ""), flush=True)
    return recs


# --------------------------------------------------------------------------------------------- aggregation
CUM_BINS = [0, 0.01, 0.1, 0.25, 0.5, 1, 2, 4, 8, 1e9]


def pooled(rs, i="ia", u="ua"):
    s_u = sum(r[u] for r in rs)
    return sum(r[i] for r in rs) / s_u if s_u else float("nan")


def med(rs, k):
    v = [r[k] for r in rs if r["has"]]
    return float(np.median(v)) if v else float("nan")


def summarize(recs):
    by_t = {}
    for r in recs:
        by_t.setdefault(r["t"], []).append(r)
    ts = sorted(by_t)
    last = [r for r in recs if r["t"] == r["T"] - 1]
    curve = {k: [pooled(by_t[t], *k2) for t in ts] for k, k2 in (("iou_amodal", ("ia", "ua")), ("iou_visible", ("iv", "uv")))}
    curve |= {k: [med(by_t[t], k) for t in ts] for k in ("rot", "tr")}
    out = dict(n_views=len({(r["ep"], r["cam"]) for r in recs}), t=ts, curve=curve)
    for k, (i, u) in (("iou_amodal", ("ia", "ua")), ("iou_visible", ("iv", "uv"))):
        at = {f"@{t}": pooled(by_t[t], i, u) for t in (0, 5, 10, 20) if t in by_t}
        at["@last"] = pooled(last, i, u)
        at["gain"] = at["@last"] - at["@0"]
        c = curve[k]
        fin = at["@last"]
        at["t95"] = next((t for t, v in zip(ts, c) if v >= 0.95 * fin), None) if fin == fin else None
        out[k] = at
    for k in ("rot", "tr"):
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
    cols = {"independent": "#4C72B0", "track": "#DD8452", "track_mem": "#55A868"}
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
    ap.add_argument("--reset-px", type=float, default=6.0)
    ap.add_argument("--dither", type=float, default=0.0, help="track*: hypothesis = estimate + this x the training "
                    "perturbation (3 deg, 3 cm), fresh every frame")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True, help="output prefix (writes <out>_<set>.json/.png, <out>.md)")
    a = ap.parse_args()
    sets = a.set or ["eval=data/eval", "eval_real=data/hf/eval_real/i000"]
    dev = "cuda"
    s = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ta = s["args"]
    net = Stage1Net(pretrained=False).to(dev).eval()
    net.load_state_dict(s["net"])
    hyp = dict(rot_deg=ta["rot_deg"], trans_m=ta["trans_m"], spacing=ta["spacing"], r_max=ta["r_max"], corr=ta["corr"])
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
               "| method | group | views | IoU@0 | @5 | @10 | @20 | @last | gain | t95 | rot@0 | rot@last | trans@0 | trans@last |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, S in summ.items():
            for g, r in S.items():
                i_ = r["iou_amodal"]
                md.append(f"| {m} | {g} | {r['n_views']} | {fmt(i_['@0'])} | {fmt(i_.get('@5'))} | {fmt(i_.get('@10'))} | "
                          f"{fmt(i_.get('@20'))} | {fmt(i_['@last'])} | {fmt(i_['gain'])} | {i_['t95']} | "
                          f"{fmt(r['rot']['@0'], 2)} | {fmt(r['rot']['@last'], 2)} | {fmt(r['tr']['@0'], 2)} | {fmt(r['tr']['@last'], 2)} |")
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
