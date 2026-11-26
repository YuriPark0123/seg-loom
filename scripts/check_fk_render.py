"""Check the torch FK / projection / point renderer against a generated split (default data/ref).

For every episode:
  fk        FK(true.urdf, q_true) vs arrays body_pose           position (mm) / rotation (deg) error
  proj      torch projection of skel3d vs stored skel2d_{cam}    pixel error (OpenCV equations, all lens kinds)
  render    point splat of render_geoms (true geometry + attachments) at body_pose vs am_r labels: IoU, label acc
  urdf      point splat of true.urdf visuals at FK(q_true)       vs am_r (no attachments -> slightly lower)
  approx    approx.urdf visuals at FK(approx, q_enc)             vs am_r: size of the gap the model must close
  wrist     std of T_ee^-1 T_base_cam_wrist over the episode     (wrist camera rigidly mounted?)

    python scripts/check_fk_render.py --data data/ref --limit 200 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadow.datagen import unpack_image  # noqa: E402
from shadow.learn.camera import cams_from_meta, project  # noqa: E402
from shadow.learn.fk import URDFKinematics  # noqa: E402
from shadow.learn.geometry import inv_T, pose7_to_T, rot_angle, transform  # noqa: E402
from shadow.learn.points import point_spacing, points_from_render_geoms, points_from_urdf, splat_labels, world_points  # noqa: E402

CAMS = ("top", "front", "wrist")


def iou(a, b):
    i = (a & b).sum().item()
    u = (a | b).sum().item()
    return i / u if u else 1.0


def near_wrist_links(kin: URDFKinematics, ee: str) -> set[str]:
    """Links the wrist camera sees from close range: the ee link, its parent and everything below it."""
    out, stack = {ee}, [ee]
    while stack:
        p = stack.pop()
        for j in kin.joints:
            if j.parent == p and j.child not in out:
                out.add(j.child)
                stack.append(j.child)
    out |= {j.parent for j in kin.joints if j.child == ee}
    return out


class RobotCache:
    def __init__(self, root, spacing, device, fine_spacing=None):
        self.root, self.spacing, self.fine, self.dev, self.c = root, spacing, fine_spacing, device, {}

    def get(self, name, bodies):
        if name not in self.c:
            import yaml
            d = os.path.join(self.root, "robots", name)
            kt = URDFKinematics.from_file(os.path.join(d, "true.urdf"))
            ka = URDFKinematics.from_file(os.path.join(d, "approx.urdf"))
            sp = self.spacing
            if self.fine:
                ee = yaml.safe_load(open(os.path.join(d, "spec.yaml")))["ee_body"]
                sp = {bodies.index(l): self.fine for l in near_wrist_links(kt, ee) if l in bodies}
                sp[None] = self.spacing
            t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.dev)
            Pg, Ng, Lg, _ = points_from_render_geoms(os.path.join(d, "render_geoms.npz"), sp)
            Pt, Nt, Lt = points_from_urdf(kt, sp, link_names=bodies)
            Pa, Na, La = points_from_urdf(ka, sp, link_names=bodies)
            pack = lambda P, N, L: (t(P), t(N), t(L, torch.long), t(point_spacing(sp, L)))
            self.c[name] = dict(kt=kt, ka=ka, geoms=pack(Pg, Ng, Lg), urdf=pack(Pt, Nt, Lt), approx=pack(Pa, Na, La))
        return self.c[name]


def check_episode(ep_dir, rc: RobotCache, dev, stats, cull=True, znear=0.0):
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    A = {k: torch.as_tensor(v, device=dev) for k, v in np.load(os.path.join(ep_dir, "arrays.npz")).items()}
    F = np.load(os.path.join(ep_dir, "frames.npz"))
    bodies = meta["robot_bodies"]
    Nb = len(bodies)
    R = rc.get(meta["robot"], bodies)
    kind = "real" if not meta["robot"].startswith("pg") else "procgen"
    T_body = pose7_to_T(A["body_pose"][:, :Nb].double())                       # (T, Nb, 4, 4)

    # --- FK (true URDF + true joints) vs MuJoCo bodies
    for key, kin, q in (("fk_true", R["kt"], A["q_true"]), ("fk_approx", R["ka"], A["q_enc"])):
        Tl = kin.fk(q.double(), meta["joint_names"])
        Tl = Tl[:, [kin.links.index(b) for b in bodies]]
        Tl = T_body[:, :1] @ inv_T(Tl[:, :1]) @ Tl                              # express relative to the root body
        pe = (Tl[..., :3, 3] - T_body[..., :3, 3]).norm(dim=-1) * 1000
        re = torch.rad2deg(rot_angle(Tl[..., :3, :3], T_body[..., :3, :3]))
        stats[f"{key}_pos_mm_max"][kind].append(pe.max().item())
        stats[f"{key}_rot_deg_max"][kind].append(re.max().item())
    root_off = A["body_pose"][0, 0, :3].norm().item() * 1000
    stats["root_body_offset_mm"][kind].append(root_off)
    T_fk = T_body[:, :1] @ inv_T(R["kt"].fk(A["q_true"].double(), meta["joint_names"])[:, :1]) @ \
        R["kt"].fk(A["q_true"].double(), meta["joint_names"])[:, [R["kt"].links.index(b) for b in bodies]]
    T_ap = R["ka"].fk(A["q_enc"].double(), meta["joint_names"])
    T_ap = T_body[:, :1] @ inv_T(T_ap[:, :1]) @ T_ap[:, [R["ka"].links.index(b) for b in bodies]]

    for c in CAMS:
        model = meta["cameras"][c]["model"]
        T = A["T_base_cam_" + c].shape[0]
        cams = cams_from_meta([model] * T, dev, torch.float64)
        Tcb = inv_T(A["T_base_cam_" + c].double())                                # base -> cam
        # --- projection of the skeleton
        pc = transform(Tcb, A["skel3d"].double())
        uv, ok = project(pc, cams)
        ref = A["skel2d_" + c].double()
        m = (ref[..., 2] > 0.5) & ok
        if m.any():
            e = (uv - ref[..., :2]).norm(dim=-1)[m]
            stats[f"proj_px_max_{model['kind']}"][kind].append(e.max().item())
        # --- renders vs am_r
        H, W = model["height"], model["width"]
        gt = torch.as_tensor(np.stack([unpack_image(F[f"{c}_am_r"], F[f"{c}_am_r_off"], t, "png") for t in range(T)]),
                             device=dev).long()
        gt_fg = (gt >= 1) & (gt <= Nb)
        for key, Tl in (("render", T_body), ("urdf", T_fk), ("approx", T_ap)):
            P, Nrm, L, SP = R["geoms" if key == "render" else ("urdf" if key == "urdf" else "approx")]
            pw, nw = world_points(Tl.float(), P, L, Nrm)
            pcam = transform(Tcb.float(), pw)
            ncam = nw @ Tcb.float()[:, :3, :3].transpose(-1, -2)
            lab, _ = splat_labels(pcam, L + 1, cams_from_meta([model] * T, dev), H, W, SP,
                                  n_cam=ncam if cull else None, znear=znear)
            fg = lab > 0
            stats[f"{key}_iou_{c}"][kind].append(iou(fg, gt_fg))
            if key == "render":
                both = fg & gt_fg
                if both.any():
                    stats[f"render_label_acc_{c}"][kind].append((lab.long()[both] == gt[both]).float().mean().item())
    # --- wrist camera rigidly attached to the ee body?
    ee = bodies.index(meta.get("ee_body", "")) if meta.get("ee_body", "") in bodies else None
    if ee is None:
        spec = os.path.join(rc.root, "robots", meta["robot"], "spec.yaml")
        import yaml
        eb = yaml.safe_load(open(spec))["ee_body"]
        ee = bodies.index(eb)
    Trel = inv_T(T_body[:, ee]) @ A["T_base_cam_wrist"].double()
    stats["wrist_mount_std_mm"][kind].append((Trel[:, :3, 3].std(0).norm() * 1000).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ref")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spacing", type=float, default=0.0015)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write the summary as json")
    ap.add_argument("--no-cull", action="store_true", help="disable back-face culling")
    ap.add_argument("--znear", type=float, default=0.0)
    ap.add_argument("--fine-spacing", type=float, default=0.0003,
                    help="spacing for the links next to the wrist camera (0 = same as --spacing)")
    a = ap.parse_args()
    eps = [json.loads(l)["episode"] for l in open(os.path.join(a.data, "manifest.jsonl"))]
    if a.limit:
        rng = np.random.default_rng(0)
        eps = list(rng.choice(eps, min(a.limit, len(eps)), replace=False))
    rc = RobotCache(a.data, a.spacing, a.device, a.fine_spacing or None)
    stats = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    for i, e in enumerate(eps):
        check_episode(os.path.join(a.data, "episodes", e), rc, a.device, stats, not a.no_cull, a.znear)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(eps)}  {time.time() - t0:.0f}s", flush=True)
    summ = {}
    print(f"\n{len(eps)} episodes from {a.data}  ({time.time() - t0:.0f}s)")
    print(f"{'metric':32s} {'kind':8s} {'n':>5s} {'mean':>9s} {'p05':>9s} {'p50':>9s} {'p95':>9s} {'max':>9s} {'min':>9s}")
    for k in sorted(stats):
        for kind, v in sorted(stats[k].items()):
            v = np.array(v)
            row = dict(n=len(v), mean=v.mean(), p05=np.percentile(v, 5), p50=np.median(v), p95=np.percentile(v, 95),
                       max=v.max(), min=v.min())
            summ[f"{k}/{kind}"] = row
            print(f"{k:32s} {kind:8s} {row['n']:5d} " + " ".join(f"{row[s]:9.4f}" for s in ("mean", "p05", "p50", "p95", "max", "min")))
    if a.out:
        json.dump(summ, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
