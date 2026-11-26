"""Stage-1 smoke training: render-conditioned robot segmentation (L1 / M1) + one-step camera registration (L3 / M3).

Per camera view the network sees
  rgb (3) | hypothesis render of the APPROX URDF at q_enc through a PERTURBED camera pose: mask (1), link id (1),
  inverse depth (1) | ray map (3)                                                         = 9 channels
and predicts
  amodal + visible robot masks, and a camera pose correction dT (rotation vector, translation / hypothesis depth),
  applied as T_pred = T_noisy @ exp(dT).
The ground truth comes from the dataset (am_r / vis labels, T_base_cam). This is a pipeline check on data/eval,
not a result: robots are split so validation robots are never trained on (real robots always held out).

    python scripts/train_stage1.py --data data/eval --steps 3000 --out out/stage1_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadow.learn.data import CAMS, EpisodeChunkSampler, SegLoomFrames, collate  # noqa: E402
from shadow.learn.geometry import inv_T, make_T, rot_angle, so3_exp, transform  # noqa: E402

REAL = ("so101", "ur7e", "piper", "panda")


# ------------------------------------------------------------------------------------------------ model
class Stage1Net(nn.Module):
    def __init__(self, in_ch=9, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        w = r.conv1.weight.data
        conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        conv1.weight.data.zero_()
        conv1.weight.data[:, :3] = w
        conv1.weight.data[:, 3:] = w.mean(1, keepdim=True) * 0.5
        self.stem = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, 96, 1) for c in (64, 128, 256, 512)])
        self.seg = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv2d(96, 2, 1))
        self.pose = nn.Sequential(nn.Linear(512 + 96, 256), nn.ReLU(), nn.Linear(256, 6))
        nn.init.zeros_(self.pose[-1].weight); nn.init.zeros_(self.pose[-1].bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = Fn.interpolate(p, size=c.shape[-2:], mode="bilinear", align_corners=False) + lat(c)
        seg = Fn.interpolate(self.seg(p), size=(H, W), mode="bilinear", align_corners=False)
        g = torch.cat([c4.mean((2, 3)), p.mean((2, 3))], 1)
        return seg, self.pose(g)


# ------------------------------------------------------------------------------------------------ batch prep
@torch.no_grad()
def prepare(b, dev):
    """-> network input (V, 9, H, W), targets, noisy/gt camera poses, per-view skeleton points (list).
    The hypothesis render (approx URDF at q_enc, perturbed camera) comes from the loader workers (CPU)."""
    C = len(CAMS)
    T_gt = b["T_base_cam"].to(dev).flatten(0, 1)                             # (V, 4, 4), V = B*C
    T_noisy = b["T_noisy"].to(dev).flatten(0, 1)
    hyp_lab = b["hyp_lab"].to(dev).flatten(0, 1)
    hyp_dep = b["hyp_dep"].to(dev).flatten(0, 1)
    skel = [s.to(dev) for s in b["skel3d"]]
    nb = b["n_bodies"].to(dev).repeat_interleave(C).float()
    hm = (hyp_lab > 0).float()
    hl = hyp_lab.float() / nb[:, None, None]
    hd = torch.where(hyp_dep > 0, 1.0 / hyp_dep.clamp_min(0.05), torch.zeros_like(hyp_dep)) * 0.2
    rgb = b["rgb"].to(dev).flatten(0, 1).float() / 255.0
    rgb = (rgb - torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]) / torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    ray = b["ray"].to(dev).flatten(0, 1).float()
    x = torch.cat([rgb, hm[:, None], hl[:, None], hd[:, None], ray], 1)
    am = b["am_r"].to(dev).flatten(0, 1).long()
    vs = b["vis"].to(dev).flatten(0, 1).long()
    nbv = nb.long()[:, None, None]
    y_am = ((am >= 1) & (am <= nbv)).float()
    y_vis = ((vs >= 1) & (vs <= nbv)).float()
    valid = b["valid"].to(dev).flatten(0, 1).float()
    # hypothesis depth scale for the translation head: median robot depth, else distance to the base
    d = hyp_dep.flatten(1)
    zmed = torch.where(d > 0, d, torch.full_like(d, float("nan"))).nanmedian(1).values
    zref = torch.where(torch.isnan(zmed), T_noisy[:, :3, 3].norm(dim=-1), zmed)
    return x, y_am, y_vis, valid, hm, T_gt, T_noisy, zref.clamp_min(0.05), skel


def apply_delta(T_noisy, d, zref):
    return T_noisy @ make_T(so3_exp(d[:, :3]), d[:, 3:] * zref[:, None])


def pose_point_loss(T_pred, T_gt, skel, C):
    """Mean distance (relative to depth) of the robot skeleton points expressed in the predicted vs true camera."""
    losses = []
    for i, P in enumerate(skel):
        sl = slice(i * C, (i + 1) * C)
        a = transform(inv_T(T_pred[sl]), P[None].expand(C, -1, -1))
        g = transform(inv_T(T_gt[sl]), P[None].expand(C, -1, -1))
        losses.append(((a - g).norm(dim=-1) / g[..., 2].abs().clamp_min(0.05)).mean(-1))
    return torch.cat(losses)


def seg_loss(logit, y, valid):
    bce = Fn.binary_cross_entropy_with_logits(logit, y, reduction="none")
    bce = (bce * valid).sum() / valid.sum()
    p = torch.sigmoid(logit) * valid
    dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + (y * valid).sum((1, 2)) + 1)
    return bce + dice.mean()


def iou_sum(pred, y, valid):
    p, y = (pred & valid), (y & valid)
    return (p & y).sum((1, 2)).float(), (p | y).sum((1, 2)).float()


# ------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(net, dl, dev, max_batches):
    net.eval()
    acc = {}
    for k, b in enumerate(dl):
        if k >= max_batches:
            break
        x, y_am, y_vis, valid, hm, T_gt, T_noisy, zref, skel = prepare(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            seg, d = net(x)
        seg, d = seg.float(), d.float()
        T_pred = apply_delta(T_noisy, d, zref)
        vb = valid.bool()
        groups = ["real" if r in REAL else "procgen" for r in b["robot"] for _ in CAMS]
        cams = [c for _ in b["robot"] for c in CAMS]
        stats = {
            "iou_amodal": iou_sum(seg[:, 0] > 0, y_am.bool(), vb),
            "iou_visible": iou_sum(seg[:, 1] > 0, y_vis.bool(), vb),
            "iou_hypothesis": iou_sum(hm.bool(), y_am.bool(), vb),
        }
        rot_b = torch.rad2deg(rot_angle(T_noisy[:, :3, :3], T_gt[:, :3, :3]))
        rot_a = torch.rad2deg(rot_angle(T_pred[:, :3, :3], T_gt[:, :3, :3]))
        tr_b = (T_noisy[:, :3, 3] - T_gt[:, :3, 3]).norm(dim=-1) * 1000
        tr_a = (T_pred[:, :3, 3] - T_gt[:, :3, 3]).norm(dim=-1) * 1000
        for v in range(len(groups)):
            for key in (groups[v], f"{groups[v]}/{cams[v]}"):
                a = acc.setdefault(key, {})
                for s, (inter, union) in stats.items():
                    a.setdefault(s, [0.0, 0.0])
                    a[s][0] += inter[v].item(); a[s][1] += union[v].item()
                for s, val in (("rot_deg_before", rot_b), ("rot_deg_after", rot_a), ("trans_mm_before", tr_b), ("trans_mm_after", tr_a)):
                    a.setdefault(s, []).append(val[v].item())
    net.train()
    out = {}
    for g, a in sorted(acc.items()):
        out[g] = {s: (v[0] / max(v[1], 1) if isinstance(v[0], float) and len(v) == 2 and not s.startswith(("rot", "trans")) else float(np.median(v)))
                  for s, v in a.items()}
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval")
    ap.add_argument("--out", default="out/stage1_smoke")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=12, help="time steps per batch (x3 camera views)")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rot-deg", type=float, default=3.0)
    ap.add_argument("--trans-m", type=float, default=0.03)
    ap.add_argument("--spacing", type=float, default=0.003, help="approx-URDF point spacing for the hypothesis render")
    ap.add_argument("--r-max", type=int, default=3)
    ap.add_argument("--val-procgen", type=int, default=10, help="procgen robots held out for validation")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--mem-frac", type=float, default=0.0, help="cap this process's share of GPU memory (shared GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = a.device
    if dev.startswith("cuda") and a.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)

    procgen = sorted({json.loads(l)["robot"] for l in open(os.path.join(a.data, "manifest.jsonl"))} - set(REAL))
    rng = np.random.default_rng(a.seed)
    val_pg = sorted(rng.choice(procgen, a.val_procgen, replace=False).tolist())
    train_robots = [r for r in procgen if r not in val_pg]
    split = dict(train=train_robots, val_procgen=val_pg, val_real=list(REAL))
    json.dump(dict(args=vars(a), split=split), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    print(f"train robots {len(train_robots)} | val procgen {len(val_pg)} + real {len(REAL)}")

    hyp = dict(rot_deg=a.rot_deg, trans_m=a.trans_m, spacing=a.spacing, r_max=a.r_max)
    ds_tr = SegLoomFrames(a.data, robots=train_robots, scale=a.scale, hypothesis=hyp, seed=a.seed)
    ds_va = SegLoomFrames(a.data, robots=val_pg + list(REAL), scale=a.scale, hypothesis=hyp, seed=1234)
    print(f"train frames {len(ds_tr)} | val frames {len(ds_va)}")
    dl_tr = DataLoader(ds_tr, a.batch, sampler=EpisodeChunkSampler(ds_tr, 4, a.seed), num_workers=a.workers,
                       collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=4)
    dl_va = DataLoader(ds_va, a.batch, sampler=EpisodeChunkSampler(ds_va, 4, 999), num_workers=a.workers // 2,
                       collate_fn=collate)
    net = Stage1Net().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    res = evaluate(net, dl_va, dev, a.eval_batches)
    print("step 0 eval", json.dumps({g: {k: round(v, 3) for k, v in s.items()} for g, s in res.items() if "/" not in g}))
    log.write(json.dumps(dict(step=0, eval=res)) + "\n"); log.flush()

    step, t0, tl = 0, time.time(), {}
    t_data = t_prep = t_step = 0.0
    while step < a.steps:
        tq = time.time()
        for b in dl_tr:
            t1 = time.time(); t_data += t1 - tq
            x, y_am, y_vis, valid, hm, T_gt, T_noisy, zref, skel = prepare(b, dev)
            t2 = time.time(); t_prep += t2 - t1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                seg, d = net(x.contiguous(memory_format=torch.channels_last))
            seg, d = seg.float(), d.float()
            l_am = seg_loss(seg[:, 0], y_am, valid)
            l_vis = seg_loss(seg[:, 1], y_vis, valid)
            l_pose = pose_point_loss(apply_delta(T_noisy, d, zref), T_gt, skel, len(CAMS)).mean()
            loss = l_am + l_vis + 10 * l_pose
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            tq = time.time(); t_step += tq - t2
            step += 1
            for k, v in (("loss", loss), ("am", l_am), ("vis", l_vis), ("pose", l_pose)):
                tl[k] = tl.get(k, 0) + v.item()
            if step % 50 == 0:
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.startswith("cuda") else 0
                print(f"step {step} " + " ".join(f"{k} {v / 50:.4f}" for k, v in tl.items()) +
                      f" | {step / el:.2f} it/s, data {t_data / step:.3f}s prep {t_prep / step:.3f}s net {t_step / step:.3f}s"
                      f" | mem {mem:.1f}G", flush=True)
                log.write(json.dumps(dict(step=step, **{k: v / 50 for k, v in tl.items()}, it_s=step / el, mem_gb=mem)) + "\n")
                log.flush()
                tl = {}
            if step % a.eval_every == 0 or step == a.steps:
                res = evaluate(net, dl_va, dev, a.eval_batches)
                print(f"step {step} eval", json.dumps({g: {k: round(v, 3) for k, v in s.items()} for g, s in res.items() if "/" not in g}), flush=True)
                log.write(json.dumps(dict(step=step, eval=res)) + "\n"); log.flush()
                torch.save(dict(net=net.state_dict(), step=step, args=vars(a)), os.path.join(a.out, "last.pt"))
            if step >= a.steps:
                break
    print(f"done {a.steps} steps in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
