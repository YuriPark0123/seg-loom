"""torch Dataset over generated splits (docs/DATASET.md).

One item = one time step t of one episode, all three cameras:

  rgb      (C, 3, H, W) uint8     clean RGB (apply camera_model.degrade-style augmentation on top if wanted)
  am_r     (C, H, W)    uint8     amodal robot labels (0 bg, 1..Nb robot_bodies index + 1)
  vis      (C, H, W)    uint8     visible labels (robot 1..Nb, objects 100+k, table 200/201, floor 202)
  valid    (C, H, W)    bool      False on padding (images are padded to a common size)
  ray      (C, 3, H, W) float16   unit ray per pixel (lens-independent camera representation, M0)
  cam_K    (C, 4)  cam_dist (C, 5)  cam_kind (C,)  cam_size (C, 2)   for shadow.learn.camera.project
  T_base_cam (C, 4, 4)            camera pose in the robot base frame (ground truth)
  skel2d   (C, K, 3)  skel3d (K, 3)
  q_enc, q_true (J,) padded to max_joints, q_mask (J,)    (order: joint_names)
  robot (str), robot_bodies, joint_names (lists), episode (str), t (int), split root (str)

Images can be downscaled with `scale` (intrinsics follow CameraModel.scaled). Robot-level data (URDF kinematics,
surface points) is not per item: use RobotBank, keyed by (root, robot).

Reading frames.npz pulls a whole camera stream per key, so use EpisodeChunkSampler: it yields `chunk`
time steps of the same episode back to back and every worker keeps the last few episodes cached.
"""
from __future__ import annotations

import json
import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from shadow.datagen import unpack_image

from .camera import KINDS, ray_map_np

CAMS = ("top", "front", "wrist")


def scale_model(m: dict, s: float) -> dict:
    if s == 1:
        return dict(m)
    return dict(m, width=int(round(m["width"] * s)), height=int(round(m["height"] * s)), fx=m["fx"] * s, fy=m["fy"] * s,
                cx=(m["cx"] + 0.5) * s - 0.5, cy=(m["cy"] + 0.5) * s - 0.5)


class _LRU(OrderedDict):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        v = fn()
        self[key] = v
        if len(self) > self.n:
            self.popitem(last=False)
        return v


class SegLoomFrames(Dataset):
    def __init__(self, roots, robots=None, exclude_robots=None, modes=None, scale=1.0, pad_hw=None, max_joints=16,
                 cache_episodes=4, with_rgb=True, hypothesis: dict | None = None, seed=0):
        """hypothesis: if given, every item also carries a render of the APPROX URDF at q_enc through a perturbed
        camera pose, computed on the CPU in the loader worker (keeps the shared GPU free):
            dict(rot_deg=3.0, trans_m=0.03, wrist_trans_scale=0.25, spacing=0.003, r_max=3)
        adds hyp_lab (C, H, W) uint8, hyp_dep (C, H, W) float32, T_noisy (C, 4, 4)."""
        roots = [roots] if isinstance(roots, str) else list(roots)
        self.scale, self.max_joints, self.with_rgb = scale, max_joints, with_rgb
        self.hyp, self.seed, self._bank, self._rng = hypothesis, seed, None, None
        self.pad_hw = pad_hw or (int(round(320 * scale)), int(round(320 * scale)))
        self.episodes = []                         # (root, episode, robot, T)
        for r in roots:
            for line in open(os.path.join(r, "manifest.jsonl")):
                e = json.loads(line)
                if robots is not None and e["robot"] not in robots:
                    continue
                if exclude_robots is not None and e["robot"] in exclude_robots:
                    continue
                if modes is not None and e["mode"] not in modes:
                    continue
                self.episodes.append((r, e["episode"], e["robot"], int(e["T"])))
        self.items = [(i, t) for i, (_, _, _, T) in enumerate(self.episodes) for t in range(T)]
        self._ep = _LRU(cache_episodes)
        self._ray = _LRU(cache_episodes * 6)

    def __len__(self):
        return len(self.items)

    def robots(self):
        return sorted({(r, rb) for r, _, rb, _ in self.episodes})

    def _load(self, i):
        root, ep, _, _ = self.episodes[i]
        d = os.path.join(root, "episodes", ep)
        meta = json.load(open(os.path.join(d, "meta.json")))
        A = dict(np.load(os.path.join(d, "arrays.npz")))
        F = np.load(os.path.join(d, "frames.npz"))
        streams = {}
        keys = [f"{c}_{k}" for c in CAMS for k in (("rgb", "am_r", "vis") if self.with_rgb else ("am_r", "vis"))]
        for k in keys:
            streams[k] = (F[k], F[k + "_off"])
        return meta, A, streams

    def _fold(self, i, c, m):
        def f():
            from shadow.camera_model import CameraModel
            return CameraModel(m["width"], m["height"], m["fx"], m["fy"], m["cx"], m["cy"], m["kind"],
                               np.asarray(m["dist"], float)).fold_radius()
        return self._ray.get_or((i, c, "fold"), f)

    def _img(self, streams, key, t, kind, hw):
        buf, off = streams[key]
        im = unpack_image(buf, off, t, kind)
        if self.scale != 1:
            im = cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_AREA if kind == "jpg" else cv2.INTER_NEAREST)
        return im

    def __getitem__(self, idx):
        i, t = self.items[idx]
        meta, A, S = self._ep.get_or(i, lambda: self._load(i))
        Hp, Wp = self.pad_hw
        C = len(CAMS)
        out = dict(rgb=torch.zeros(C, 3, Hp, Wp, dtype=torch.uint8), am_r=torch.zeros(C, Hp, Wp, dtype=torch.uint8),
                   vis=torch.zeros(C, Hp, Wp, dtype=torch.uint8), valid=torch.zeros(C, Hp, Wp, dtype=torch.bool),
                   ray=torch.zeros(C, 3, Hp, Wp, dtype=torch.float16))
        K, dist, kind, size, Tbc, s2d = [], [], [], [], [], []
        for ci, c in enumerate(CAMS):
            m = scale_model(meta["cameras"][c]["model"], self.scale)
            h, w = m["height"], m["width"]
            assert h <= Hp and w <= Wp, f"image {h}x{w} larger than pad {Hp}x{Wp}"
            if self.with_rgb:
                out["rgb"][ci, :, :h, :w] = torch.from_numpy(np.ascontiguousarray(self._img(S, f"{c}_rgb", t, "jpg", (h, w)))).permute(2, 0, 1)
            out["am_r"][ci, :h, :w] = torch.from_numpy(self._img(S, f"{c}_am_r", t, "png", (h, w)))
            out["vis"][ci, :h, :w] = torch.from_numpy(self._img(S, f"{c}_vis", t, "png", (h, w)))
            out["valid"][ci, :h, :w] = True
            ray = self._ray.get_or((i, c), lambda: torch.from_numpy(ray_map_np(m)).permute(2, 0, 1).half())
            out["ray"][ci, :, :h, :w] = ray
            K.append([m["fx"], m["fy"], m["cx"], m["cy"]])
            dist.append((list(m["dist"]) + [0] * 5)[:5])
            kind.append(KINDS[m["kind"]])
            size.append([w, h])
            Tbc.append(A[f"T_base_cam_{c}"][t])
            sk = A[f"skel2d_{c}"][t].copy()
            sk[:, :2] = (sk[:, :2] + 0.5) * self.scale - 0.5
            # splits generated before the lens-fold fix flag some far-off points as inside: recompute the flag
            T = A[f"T_base_cam_{c}"][t]
            pc = (A["skel3d"][t] - T[:3, 3]) @ T[:3, :3]
            z = pc[:, 2]
            r = np.linalg.norm(pc[:, :2], axis=1) / np.where(z > 1e-6, z, 1.0)
            sk[:, 2] *= (z > 1e-6) & (r < self._fold(i, c, m))
            s2d.append(sk)
        J = self.max_joints
        nq = A["q_enc"].shape[1]
        qe = torch.zeros(J); qt = torch.zeros(J); qm = torch.zeros(J, dtype=torch.bool)
        qe[:nq] = torch.from_numpy(A["q_enc"][t]); qt[:nq] = torch.from_numpy(A["q_true"][t]); qm[:nq] = True
        out.update(cam_K=torch.tensor(K, dtype=torch.float32), cam_dist=torch.tensor(dist, dtype=torch.float32),
                   cam_kind=torch.tensor(kind), cam_size=torch.tensor(size),
                   T_base_cam=torch.from_numpy(np.stack(Tbc)).float(), skel2d=torch.from_numpy(np.stack(s2d)).float(),
                   skel3d=torch.from_numpy(A["skel3d"][t]).float(), q_enc=qe, q_true=qt, q_mask=qm,
                   n_bodies=len(meta["robot_bodies"]), robot_bodies=meta["robot_bodies"],
                   joint_names=meta["joint_names"], robot=meta["robot"], episode=self.episodes[i][1],
                   root=self.episodes[i][0], t=t)
        if self.hyp is not None:
            out.update(self._hypothesis(out, A, meta, t))
        return out

    def _hypothesis(self, out, A, meta, t):
        from .geometry import inv_T, make_T, so3_exp, transform
        from .points import splat_labels, world_points
        h = self.hyp
        if self._bank is None:
            torch.set_num_threads(1)
            self._bank = RobotBank(h.get("spacing", 0.003), "cpu")
            wi = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng([self.seed, wi.id if wi else 0, os.getpid()])
        R = self._bank.get(out["root"], out["robot"], meta["robot_bodies"])
        C = len(CAMS)
        q = torch.from_numpy(A["q_enc"][t]).float()[None]
        Tl = R["kin_approx"].fk(q, meta["joint_names"])[:, R["body_idx_approx"]]
        pw, nw = world_points(Tl, R["pts_approx"], R["link_approx"], R["nrm_approx"])
        w = torch.from_numpy(self._rng.normal(0, np.deg2rad(h.get("rot_deg", 3.0)), (C, 3))).float()
        sig = np.array([h.get("trans_m", 0.03) * (h.get("wrist_trans_scale", 0.25) if c == "wrist" else 1) for c in CAMS])
        tr = torch.from_numpy(self._rng.normal(0, 1, (C, 3)) * sig[:, None]).float()
        T_noisy = out["T_base_cam"] @ make_T(so3_exp(w), tr)
        Tcb = inv_T(T_noisy)
        pc = transform(Tcb, pw.expand(C, -1, -1))
        nc = nw.expand(C, -1, -1) @ Tcb[:, :3, :3].transpose(-1, -2)
        cams = dict(K=out["cam_K"], dist=out["cam_dist"], kind=out["cam_kind"], size=out["cam_size"])
        Hp, Wp = self.pad_hw
        lab, dep = splat_labels(pc, R["link_approx"] + 1, cams, Hp, Wp, self._bank.spacing,
                                r_max=h.get("r_max", 3), n_cam=nc)
        lab = lab * out["valid"]
        return dict(hyp_lab=lab, hyp_dep=dep * out["valid"], T_noisy=T_noisy)


def collate(batch):
    out = {}
    for k in batch[0]:
        v = [b[k] for b in batch]
        if k == "skel2d" or k == "skel3d":
            out[k] = v                                   # K differs per robot
        elif isinstance(v[0], torch.Tensor):
            out[k] = torch.stack(v)
        elif isinstance(v[0], (int, float)):
            out[k] = torch.tensor(v)
        else:
            out[k] = v
    return out


class EpisodeChunkSampler(Sampler):
    """Shuffled chunks of `chunk` consecutive-in-storage time steps from one episode (random order inside)."""

    def __init__(self, ds: SegLoomFrames, chunk=8, seed=0):
        self.ds, self.chunk, self.seed, self.epoch = ds, chunk, seed, 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        by_ep = {}
        for idx, (i, _) in enumerate(self.ds.items):
            by_ep.setdefault(i, []).append(idx)
        chunks = []
        for idxs in by_ep.values():
            idxs = idxs[:]
            rng.shuffle(idxs)
            chunks += [idxs[k:k + self.chunk] for k in range(0, len(idxs), self.chunk)]
        rng.shuffle(chunks)
        for c in chunks:
            yield from c

    def __len__(self):
        return len(self.ds)


class RobotBank:
    """Per-robot static data on a device: kinematics of approx/true URDF and surface points."""

    def __init__(self, spacing=0.0015, device="cpu"):
        self.spacing, self.dev, self.c = spacing, device, {}

    def get(self, root, robot, bodies=None):
        key = (root, robot)
        if key not in self.c:
            from .fk import URDFKinematics
            from .points import points_from_render_geoms, points_from_urdf
            d = os.path.join(root, "robots", robot)
            kt = URDFKinematics.from_file(os.path.join(d, "true.urdf"))
            ka = URDFKinematics.from_file(os.path.join(d, "approx.urdf"))
            bodies = bodies or kt.links
            Pa, Na, La = points_from_urdf(ka, self.spacing, link_names=bodies)
            Pg, Ng, Lg, _ = points_from_render_geoms(os.path.join(d, "render_geoms.npz"), self.spacing)
            t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.dev)
            self.c[key] = dict(kin_true=kt, kin_approx=ka, bodies=bodies,
                               pts_approx=t(Pa), nrm_approx=t(Na), link_approx=t(La, torch.long),
                               pts_true=t(Pg), nrm_true=t(Ng), link_true=t(Lg, torch.long),
                               body_idx_approx=[ka.links.index(b) for b in bodies],
                               body_idx_true=[kt.links.index(b) for b in bodies])
        return self.c[key]
