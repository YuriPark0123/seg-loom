"""Stage-1 training: render-conditioned robot segmentation (L1 / M1) + correspondence-based camera registration (L3 / M3).

Per camera view the network sees
  rgb (3) | hypothesis render of the APPROX URDF at q_enc through a PERTURBED camera: mask, link id, inverse depth (3)
  | ray map (3)                                                                                    = 9 channels
and predicts
  * amodal + visible robot masks
  * a dense 2D displacement field: where each hypothesis pixel's robot surface really is in the image
The camera pose is NOT regressed: hypothesis points + predicted displacements are 2D-3D matches, and
shadow.learn.register.solve_pose (LM on the lens reprojection error) turns them into T_base_cam. Direct 6-DoF
regression did not learn here even with an oracle URDF; a local displacement is what a CNN learns well, and the
geometry stays structural (docs/ARCHITECTURE.md A: FK + projection fixed, the discrepancy learned).

Robots are split so validation robots are never trained on; the four real robots are always held out.

    python scripts/train_stage1.py --data data/eval --steps 3000 --out out/stage1_smoke                 # smoke
    python scripts/train_stage1.py --data data/train_000 --val-data data/eval --steps 60000 --out out/stage1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadow.learn.data import CAMS, EpisodeChunkSampler, MixedChunkSampler, SegLoomFrames, collate  # noqa: E402
from shadow.learn.camera import project  # noqa: E402
from shadow.learn.geometry import inv_T, rot_angle, transform  # noqa: E402
from shadow.learn.register import solve_pose  # noqa: E402

REAL = ("so101", "ur7e", "piper", "panda")
FLOW_UNIT = 8.0          # network outputs displacement / FLOW_UNIT pixels
POSE_PRIOR = (0.05, 0.03)  # rad, m: Gaussian prior of the pose solve around the initial camera (see register.py)


# ------------------------------------------------------------------------------------------------ model
class Stage1Net(nn.Module):
    def __init__(self, in_ch=9, pretrained=True, width=128):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        w = r.conv1.weight.data
        conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        conv1.weight.data.zero_()
        conv1.weight.data[:, :3] = w
        conv1.weight.data[:, 3:] = w.mean(1, keepdim=True) * 0.5
        self.stem = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512)])
        head = lambda o: nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, o, 1))
        self.seg = head(2)
        self.flow = head(2)
        nn.init.zeros_(self.flow[-1].weight); nn.init.zeros_(self.flow[-1].bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = Fn.interpolate(p, size=c.shape[-2:], mode="bilinear", align_corners=False) + lat(c)
        seg = Fn.interpolate(self.seg(p), size=(H, W), mode="bilinear", align_corners=False)
        return seg, self.flow(p)                     # flow at 1/4 resolution, in FLOW_UNIT pixels


def sample_flow(flow, uv, Hp, Wp):
    """flow (V, 2, h, w) over the padded image, uv (V, M, 2) pixel coords -> displacement (V, M, 2) in pixels."""
    g = torch.stack([(uv[..., 0] + 0.5) / Wp * 2 - 1, (uv[..., 1] + 0.5) / Hp * 2 - 1], -1)[:, :, None]
    s = Fn.grid_sample(flow.float(), g.float(), mode="bilinear", align_corners=False)      # (V, 2, M, 1)
    return s[..., 0].permute(0, 2, 1) * FLOW_UNIT


# ------------------------------------------------------------------------------------------------ batch prep
@torch.no_grad()
def prepare(b, dev):
    C = len(CAMS)
    f = lambda k: b[k].to(dev).flatten(0, 1)
    hyp_lab, hyp_dep = f("hyp_lab"), f("hyp_dep")
    nb = b["n_bodies"].to(dev).repeat_interleave(C).float()
    hm = (hyp_lab > 0).float()
    hl = hyp_lab.float() / nb[:, None, None]
    hd = torch.where(hyp_dep > 0, 1.0 / hyp_dep.clamp_min(0.05), torch.zeros_like(hyp_dep)) * 0.2
    rgb = f("rgb").float() / 255.0
    rgb = (rgb - torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]) / torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    x = torch.cat([rgb, hm[:, None], hl[:, None], hd[:, None], f("ray").float()], 1)
    nbv = nb.long()[:, None, None]
    am, vs = f("am_r").long(), f("vis").long()
    tgt = dict(am=((am >= 1) & (am <= nbv)).float(), vis=((vs >= 1) & (vs <= nbv)).float(), valid=f("valid").float(),
               hm=hm, T_gt=f("T_base_cam"), T_noisy=f("T_noisy"),
               X=f("corr_X"), uvh=f("corr_uvh"), uvt=f("corr_uvt"), w=f("corr_w"),
               cams=dict(K=f("cam_K"), dist=f("cam_dist"), kind=f("cam_kind"), size=f("cam_size")))
    return x, tgt


def seg_loss(logit, y, valid):
    bce = Fn.binary_cross_entropy_with_logits(logit, y, reduction="none")
    bce = (bce * valid).sum() / valid.sum()
    p = torch.sigmoid(logit) * valid
    dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + (y * valid).sum((1, 2)) + 1)
    return bce + dice.mean()


MAX_DISP = 48.0          # px: larger targets are lens-edge / near-plane outliers, left out of the loss


def flow_loss(disp, tgt):
    d = tgt["uvt"] - tgt["uvh"]
    w = tgt["w"] * (d.norm(dim=-1) < MAX_DISP)
    e = (disp - d).abs().sum(-1)                                              # L1 px
    return (e * w).sum() / w.sum().clamp_min(1)


def iou_sum(pred, y, valid):
    p, y = (pred & valid), (y & valid)
    return (p & y).sum((1, 2)).float(), (p | y).sum((1, 2)).float()


# ------------------------------------------------------------------------------------------------ eval
def reproj_px(T, t):
    """Mean pixel distance of the matched robot points seen through T vs the true camera (what registration is for)."""
    p, ok = project(transform(inv_T(T), t["X"]), t["cams"])
    w = t["w"] * ok
    return ((p - t["uvt"]).norm(dim=-1) * w).sum(1) / w.sum(1).clamp_min(1)


@torch.no_grad()
def evaluate(net, dl, dev, max_batches=None, prior=POSE_PRIOR):
    """IoU (pixel-pooled) and per-view medians/means of the camera-pose errors after the solve. Groups: all, real,
    procgen, and <group>/<cam>."""
    was = net.training
    net.eval()
    acc = {}
    for k, b in enumerate(dl):
        if max_batches is not None and k >= max_batches:
            break
        x, t = prepare(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            seg, flow = net(x)
        seg = seg.float()
        Hp, Wp = x.shape[-2:]
        disp = sample_flow(flow, t["uvh"], Hp, Wp)
        T_pred, _ = solve_pose(t["T_noisy"], t["X"], t["uvh"] + disp, t["w"], t["cams"], prior_sigma=prior)
        vb = t["valid"].bool()
        groups = ["real" if r in REAL else "procgen" for r in b["robot"] for _ in CAMS]
        cams = [c for _ in b["robot"] for c in CAMS]
        ious = {"iou_amodal": iou_sum(seg[:, 0] > 0, t["am"].bool(), vb),
                "iou_visible": iou_sum(seg[:, 1] > 0, t["vis"].bool(), vb),
                "iou_hypothesis": iou_sum(t["hm"].bool(), t["am"].bool(), vb)}
        T_gt = t["T_gt"]
        err = lambda T: (torch.rad2deg(rot_angle(T[:, :3, :3], T_gt[:, :3, :3])), (T[:, :3, 3] - T_gt[:, :3, 3]).norm(dim=-1) * 100)
        (rb, tb), (ra, ta) = err(t["T_noisy"]), err(T_pred)
        rp_b, rp_a = reproj_px(t["T_noisy"], t), reproj_px(T_pred, t)
        has = t["w"].sum(1) >= 6
        for v in range(len(groups)):
            for key in ("all", groups[v], f"{groups[v]}/{cams[v]}"):
                a = acc.setdefault(key, {})
                for s_, (inter, union) in ious.items():
                    a.setdefault(s_, [0.0, 0.0])
                    a[s_][0] += inter[v].item(); a[s_][1] += union[v].item()
                if has[v]:
                    for s_, val in (("rot_deg_before", rb), ("rot_deg_after", ra), ("trans_cm_before", tb),
                                    ("trans_cm_after", ta), ("reproj_px_before", rp_b), ("reproj_px_after", rp_a)):
                        a.setdefault("m_" + s_, []).append(val[v].item())
    net.train(was)
    out = {}
    for g, a in sorted(acc.items()):
        r = {}
        for s_, v in a.items():
            if s_.startswith("m_"):
                r[s_[2:]] = float(np.median(v)); r[s_[2:] + "_mean"] = float(np.mean(v))
            else:
                r[s_] = v[0] / max(v[1], 1)
        r["n_views"] = len(a.get("m_rot_deg_after", []))
        out[g] = r
    return out


def short(res, keys=("iou_amodal", "iou_visible", "rot_deg_before", "rot_deg_after", "trans_cm_before", "trans_cm_after")):
    return json.dumps({g: {k: round(s[k], 3) for k in keys if k in s} for g, s in res.items() if "/" not in g})


# ------------------------------------------------------------------------------------------------ lr schedule
def lr_at(step, sc):
    """Linear warm-up to `peak`, then cosine from (s0, lr0) down to lr_min at `total`. Extending the run re-anchors
    the cosine at the current step and lr (re-plan the remaining steps, no jump)."""
    if step < sc["warmup"]:
        return sc["peak"] * (step + 1) / sc["warmup"]
    if step >= sc["total"]:
        return sc["lr_min"]
    f = (step - sc["s0"]) / max(1, sc["total"] - sc["s0"])
    return sc["lr_min"] + (sc["lr0"] - sc["lr_min"]) * 0.5 * (1 + math.cos(math.pi * f))


# ------------------------------------------------------------------------------------------------ main
def robots_of(roots):
    out = set()
    for r in roots:
        out |= {json.loads(l)["robot"] for l in open(os.path.join(r, "manifest.jsonl"))}
    return out


def save_ckpt(path, net, opt, step, sched, a, data):
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), step=step, sched=sched, args=vars(a), data=data,
                    time=time.strftime("%Y-%m-%d %H:%M:%S")), path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/eval"], help="training split roots (simple scenes)")
    ap.add_argument("--clutter-data", nargs="*", default=[], help="cluttered-scene roots, mixed in at --clutter-frac")
    ap.add_argument("--clutter-frac", type=float, default=0.5, help="share of each batch drawn from --clutter-data")
    ap.add_argument("--val-data", default=None, help="validation split (default: held-out robots of --data[0])")
    ap.add_argument("--out", default="out/stage1_smoke")
    ap.add_argument("--steps", type=int, default=3000, help="planned total steps (re-planned on --resume)")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=12, help="time steps per batch (x3 camera views)")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--rot-deg", type=float, default=3.0)
    ap.add_argument("--trans-m", type=float, default=0.03)
    ap.add_argument("--spacing", type=float, default=0.003, help="approx-URDF point spacing for the hypothesis render")
    ap.add_argument("--r-max", type=int, default=3)
    ap.add_argument("--corr", type=int, default=192, help="correspondences per view")
    ap.add_argument("--hyp-true", action="store_true", help="hypothesis from the TRUE urdf + q_true (oracle check)")
    ap.add_argument("--seg-w", type=float, default=1.0)
    ap.add_argument("--flow-w", type=float, default=0.1, help="weight of the L1 displacement loss (pixels)")
    ap.add_argument("--mag-lo", type=float, default=0.1, help="training noise magnitude ~ U(mag_lo, 1.5) x sigma")
    ap.add_argument("--val-procgen", type=int, default=10, help="procgen robots held out (when no --val-data)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--mem-frac", type=float, default=0.0, help="cap this process's share of GPU memory (shared GPU)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt (data and --steps may change)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    if dev.startswith("cuda") and a.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    train_robots = sorted(robots_of(a.data + a.clutter_data) - set(REAL))
    if a.val_data:
        val_robots, val_root = sorted(robots_of([a.val_data])), a.val_data
        assert not set(val_robots) & set(train_robots), "validation robots overlap the training robots"
    else:
        rng = np.random.default_rng(a.seed)
        val_pg = sorted(rng.choice(train_robots, a.val_procgen, replace=False).tolist())
        train_robots = [r for r in train_robots if r not in val_pg]
        val_robots, val_root = val_pg + list(REAL), a.data[0]

    hyp = dict(rot_deg=a.rot_deg, trans_m=a.trans_m, spacing=a.spacing, r_max=a.r_max, use_true=a.hyp_true, corr=a.corr)
    hyp_tr = dict(hyp, mag_range=(a.mag_lo, 1.5))
    srcs = [SegLoomFrames(a.data, robots=train_robots, scale=a.scale, seed=a.seed, hypothesis=hyp_tr)]
    weights = [1.0]
    if a.clutter_data:
        srcs.append(SegLoomFrames(a.clutter_data, robots=train_robots, scale=a.scale, seed=a.seed + 1, hypothesis=hyp_tr))
        weights = [1 - a.clutter_frac, a.clutter_frac]
    ds_tr = ConcatDataset(srcs)
    ds_va = SegLoomFrames(val_root, robots=val_robots, scale=a.scale, hypothesis=hyp, seed=1234, deterministic=True)
    data = dict(simple=a.data, clutter=a.clutter_data, clutter_frac=a.clutter_frac if a.clutter_data else 0.0,
                frames=[len(d) for d in srcs], episodes=[len(d.episodes) for d in srcs], robots=len(train_robots))
    print(f"train: {data} | val: {len(val_robots)} robots, {len(ds_va)} frames", flush=True)

    net = Stage1Net().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    step = 0
    sched = dict(warmup=a.warmup, peak=a.lr, s0=a.warmup, lr0=a.lr, total=a.steps, lr_min=a.lr_min)
    ck = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"])
        step, sched = s["step"], s["sched"]
        if a.steps != sched["total"]:                       # re-plan the remaining steps from here, no lr jump
            cur = lr_at(step, sched)
            sched = dict(sched, s0=max(step, sched["warmup"]), lr0=cur if step >= sched["warmup"] else sched["peak"],
                         total=a.steps)
        print(f"resumed from step {step} ({s.get('time')}), lr {lr_at(step, sched):.2e}, plan total {sched['total']}",
              flush=True)
    log.write(json.dumps(dict(event="start", step=step, time=time.strftime("%Y-%m-%d %H:%M:%S"), data=data,
                              sched=sched, args=vars(a))) + "\n"); log.flush()

    dl_tr = DataLoader(ds_tr, a.batch, sampler=MixedChunkSampler(srcs, weights, 4, a.seed + step), num_workers=a.workers,
                       collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, a.batch, sampler=EpisodeChunkSampler(ds_va, 4, 999), num_workers=max(1, a.workers // 4),
                       collate_fn=collate)
    switch = os.path.join(a.out, "SWITCH")
    t0, tl, n_log, step0 = time.time(), {}, 0, step
    for b in dl_tr:
        if step >= sched["total"]:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, sched)
        x, t = prepare(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            seg, flow = net(x.contiguous(memory_format=torch.channels_last))
        seg = seg.float()
        l_am = seg_loss(seg[:, 0], t["am"], t["valid"])
        l_vis = seg_loss(seg[:, 1], t["vis"], t["valid"])
        l_flow = flow_loss(sample_flow(flow, t["uvh"], *x.shape[-2:]), t)
        loss = a.seg_w * (l_am + l_vis) + a.flow_w * l_flow
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        n_log += 1
        for k, v in (("loss", loss), ("am", l_am), ("vis", l_vis), ("flow_px", l_flow)):
            tl[k] = tl.get(k, 0) + v.item()
        if step % 100 == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.startswith("cuda") else 0
            print(f"step {step} " + " ".join(f"{k} {v / n_log:.4f}" for k, v in tl.items()) +
                  f" | lr {lr_at(step, sched):.2e} | {(step - step0) / el:.2f} it/s | mem {mem:.1f}G", flush=True)
            log.write(json.dumps(dict(step=step, **{k: v / n_log for k, v in tl.items()}, lr=lr_at(step, sched),
                                      it_s=(step - step0) / el, mem_gb=mem)) + "\n"); log.flush()
            tl, n_log = {}, 0
        if step % a.eval_every == 0:
            res = evaluate(net, dl_va, dev, a.eval_batches)
            print(f"step {step} eval", short(res), flush=True)
            log.write(json.dumps(dict(step=step, eval=res)) + "\n"); log.flush()
        if step % a.ckpt_every == 0 or step >= sched["total"]:
            save_ckpt(ck, net, opt, step, sched, a, data)
            if os.path.exists(switch):                      # controller asked to restart with more data
                print(f"step {step}: checkpoint saved, SWITCH requested -> exit 3", flush=True)
                log.write(json.dumps(dict(event="switch_exit", step=step, time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
                log.flush()
                sys.exit(3)
    save_ckpt(ck, net, opt, step, sched, a, data)
    print(f"done: step {step} (planned {sched['total']}) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
