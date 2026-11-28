"""torch Dataset over generated splits (docs/DATASET.md).

One item = one time step t of one episode, all three cameras:

  rgb      (C, 3, H, W) uint8     clean RGB (apply camera_model.degrade-style augmentation on top if wanted)
  am_r     (C, H, W)    uint8     amodal robot labels (0 bg, 1..Nb robot_bodies index + 1)
  vis      (C, H, W)    uint8     visible labels (robot 1..Nb, objects 100+k, table 200/201, floor 202, clutter: desk distractors 210+k, walls/furniture/occluders 250)
  valid    (C, H, W)    bool      False on padding (images are padded to a common size)
  ray      (C, 3, H, W) float16   unit ray per pixel (lens-independent camera representation, M0)
  cam_K    (C, 4)  cam_dist (C, 5)  cam_kind (C,)  cam_size (C, 2)   for shadow.learn.camera.project
  T_base_cam (C, 4, 4)            camera pose in the robot base frame (ground truth)
  skel2d   (C, K, 3)  skel3d (K, 3)
  q_enc, q_true (J,) padded to max_joints, q_mask (J,)    (order: joint_names = q_names(meta))
  robot (str), robot_bodies, joint_names (lists), episode (str), t (int), split root (str)

Images can be downscaled with `scale` (intrinsics follow CameraModel.scaled). Robot-level data (URDF kinematics,
surface points) is not per item: use RobotBank, keyed by (root, robot).

Reading frames.npz pulls a whole camera stream per key, so use EpisodeChunkSampler: it yields `chunk`
time steps of the same episode back to back and every worker keeps the last few episodes cached.
"""
from __future__ import annotations

import json
import os
import time
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


CAMERA_V2_SINCE = time.strptime("2026-11-28 15:41", "%Y-%m-%d %H:%M")   # 3090 commit 59dcde1 (new camera models)


def camera_v1(root) -> bool:
    """True if `root` was generated with the old camera models: config.json "camera_model" < 2 if present, else its
    "started" time before CAMERA_V2_SINCE (no config.json: old)."""
    p = os.path.join(root, "config.json")
    if not os.path.exists(p):
        return True
    cfg = json.load(open(p))
    if "camera_model" in cfg:
        return int(cfg["camera_model"]) < 2
    try:
        return time.strptime(cfg["started"], "%a %b %d %H:%M:%S %Y") < CAMERA_V2_SINCE
    except (KeyError, ValueError):
        return True


def noisy_model(m: dict, rng, level=1.0) -> dict:
    """A lens as a ChArUco calibration would report it (scripts/charuco_calib.py, results/charuco/README.md): focal
    x (1 + N(0, 0.4 %)) shared + N(0, 0.1 %) per axis, principal point N(0, f tan 0.3 deg) px, distortion coefficients
    + N(0, 2 % |k| + 0.001). level scales all of it (0 = the true lens)."""
    if level <= 0:
        return dict(m)
    g = rng.normal(0, 0.004 * level)
    fx = m["fx"] * (1 + g + rng.normal(0, 0.001 * level))
    fy = m["fy"] * (1 + g + rng.normal(0, 0.001 * level))
    pp = np.tan(np.deg2rad(0.3)) * level
    d = np.asarray(m["dist"], float)
    d = d + rng.normal(0, 1, d.shape) * (0.02 * np.abs(d) + 0.001) * level
    return dict(m, fx=fx, fy=fy, cx=m["cx"] + rng.normal(0, pp * m["fx"]), cy=m["cy"] + rng.normal(0, pp * m["fy"]),
                dist=d.tolist())


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


def q_names(meta, nq):
    """Names of the q_enc / q_true columns. Older data: meta joint_names (all robot joints, = the q columns). Real
    robots with passive gripper linkages (xarm, interbotix, ...): joint_names lists every model joint but q holds
    only arm_joints + gripper_joints; the followers are URDF mimic joints that FK fills from their master."""
    jn = meta["joint_names"]
    if len(jn) == nq:
        return jn
    qn = list(meta["arm_joints"]) + list(meta["gripper_joints"])
    assert len(qn) == nq, f"{meta['robot']}: q has {nq} columns, joint_names {len(jn)}, arm+gripper {len(qn)}"
    return qn


class SegLoomFrames(Dataset):
    def __init__(self, roots, robots=None, exclude_robots=None, modes=None, scale=1.0, pad_hw=None, max_joints=16,
                 cache_episodes=4, with_rgb=True, hypothesis: dict | None = None, seed=0, deterministic=False):
        """hypothesis: if given, every item also carries a render of the APPROX URDF at q_enc through a perturbed
        camera pose, computed on the CPU in the loader worker (keeps the shared GPU free):
            dict(rot_deg=3.0, trans_m=0.03, wrist_trans_scale=0.25, spacing=0.003, r_max=3)
        adds hyp_lab (C, H, W) uint8, hyp_dep (C, H, W) float32, T_noisy (C, 4, 4).
          target="approx" (default): flow target = approx point (approx FK at q_enc) seen by the true camera
                 (self-consistent with the approx URDF, so it hides the joint-zero / length errors)
          target="surface": where that link point really is (TRUE FK at q_true, same link-local point), kept only
                 where the true visible-label map shows that link (scripts/diag_pose_dq.py)
          lens_noise=L: ray map, hypothesis render and cam_K / cam_dist use a calibrated-like lens (noisy_model,
                 fixed per episode and camera); cam_K_true / cam_dist_true keep the true lens (flow target)."""
        roots = [roots] if isinstance(roots, str) else list(roots)
        self.scale, self.max_joints, self.with_rgb = scale, max_joints, with_rgb
        # deterministic: the hypothesis noise / correspondence sampling of item idx depends only on (seed, idx), so
        # every evaluation of every checkpoint sees exactly the same perturbations
        self.hyp, self.seed, self._bank, self._rng, self.det = hypothesis, seed, None, None, deterministic
        self._T_init = None
        self.pad_hw = pad_hw or (int(round(320 * scale)), int(round(320 * scale)))
        self.episodes = []                         # (root, episode, robot, T)
        self.v1 = {r: camera_v1(r) for r in roots}
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
            if self.v1[self.episodes[i][0]]:
                from shadow.camera_model_v1 import CameraModel
            else:
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

    def item_with_init(self, idx, T_init):
        """Item idx with the hypothesis rendered from camera poses T_init (C, 4, 4) base<-cam instead of a random
        perturbation of the true cameras (None: as __getitem__). For sequential / tracking evaluation."""
        self._T_init = T_init
        try:
            return self[idx]
        finally:
            self._T_init = None

    def __getitem__(self, idx):
        i, t = self.items[idx]
        meta, A, S = self._ep.get_or(i, lambda: self._load(i))
        Hp, Wp = self.pad_hw
        C = len(CAMS)
        out = dict(rgb=torch.zeros(C, 3, Hp, Wp, dtype=torch.uint8), am_r=torch.zeros(C, Hp, Wp, dtype=torch.uint8),
                   vis=torch.zeros(C, Hp, Wp, dtype=torch.uint8), valid=torch.zeros(C, Hp, Wp, dtype=torch.bool),
                   ray=torch.zeros(C, 3, Hp, Wp, dtype=torch.float16))
        K, dist, kind, size, Tbc, s2d, Kt, dt_ = [], [], [], [], [], [], [], []
        for ci, c in enumerate(CAMS):
            m_true = scale_model(meta["cameras"][c]["model"], self.scale)
            ln = (self.hyp or {}).get("lens_noise", 0.0)
            m = noisy_model(m_true, np.random.default_rng([self.seed, 99, i, ci]), ln) if ln else m_true
            Kt.append([m_true["fx"], m_true["fy"], m_true["cx"], m_true["cy"]])
            dt_.append((list(m_true["dist"]) + [0] * 5)[:5])
            h, w = m["height"], m["width"]
            assert h <= Hp and w <= Wp, f"image {h}x{w} larger than pad {Hp}x{Wp}"
            if self.with_rgb:
                out["rgb"][ci, :, :h, :w] = torch.from_numpy(np.ascontiguousarray(self._img(S, f"{c}_rgb", t, "jpg", (h, w)))).permute(2, 0, 1)
            out["am_r"][ci, :h, :w] = torch.from_numpy(self._img(S, f"{c}_am_r", t, "png", (h, w)))
            out["vis"][ci, :h, :w] = torch.from_numpy(self._img(S, f"{c}_vis", t, "png", (h, w)))
            out["valid"][ci, :h, :w] = True
            v1 = self.v1[self.episodes[i][0]]
            ray = self._ray.get_or((i, c), lambda: torch.from_numpy(ray_map_np(m, v1)).permute(2, 0, 1).half())
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
            sk[:, 2] *= (z > 1e-6) & (r < self._fold(i, c, m_true))
            s2d.append(sk)
        J = self.max_joints
        nq = A["q_enc"].shape[1]
        qe = torch.zeros(J); qt = torch.zeros(J); qm = torch.zeros(J, dtype=torch.bool)
        qe[:nq] = torch.from_numpy(A["q_enc"][t]); qt[:nq] = torch.from_numpy(A["q_true"][t]); qm[:nq] = True
        out.update(cam_K=torch.tensor(K, dtype=torch.float32), cam_dist=torch.tensor(dist, dtype=torch.float32),
                   cam_kind=torch.tensor(kind), cam_size=torch.tensor(size),
                   cam_K_true=torch.tensor(Kt, dtype=torch.float32), cam_dist_true=torch.tensor(dt_, dtype=torch.float32),
                   T_base_cam=torch.from_numpy(np.stack(Tbc)).float(), skel2d=torch.from_numpy(np.stack(s2d)).float(),
                   skel3d=torch.from_numpy(A["skel3d"][t]).float(), q_enc=qe, q_true=qt, q_mask=qm,
                   n_bodies=len(meta["robot_bodies"]), robot_bodies=meta["robot_bodies"],
                   joint_names=q_names(meta, nq), robot=meta["robot"], episode=self.episodes[i][1],
                   root=self.episodes[i][0], t=t)
        if self.hyp is not None:
            if self.det:
                self._rng = np.random.default_rng([self.seed, idx])
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
            if not self.det:
                self._rng = np.random.default_rng([self.seed, wi.id if wi else 0, os.getpid()])
        R = self._bank.get(out["root"], out["robot"], meta["robot_bodies"])
        C = len(CAMS)
        src = "true" if h.get("use_true") else "approx"        # use_true: oracle URDF + q_true (debug / upper bound)
        q = torch.from_numpy(A["q_true" if src == "true" else "q_enc"][t]).float()[None]
        names = q_names(meta, q.shape[1])
        Tl = R[f"kin_{src}"].fk(q, names)[:, R[f"body_idx_{src}"]]
        pw, nw = world_points(Tl, R[f"pts_{src}"], R[f"link_{src}"], R[f"nrm_{src}"])
        pw_tgt = None
        if h.get("target", "approx") == "surface" and src == "approx":
            qt = torch.from_numpy(A["q_true"][t]).float()[None]
            Tlt = R["kin_true"].fk(qt, names)[:, R["body_idx_true"]]
            pw_tgt = world_points(Tlt, R["pts_approx"], R["link_approx"])
        # per-view noise magnitude ~ U(lo, hi) x sigma, so the refiner also sees small residuals (later iterations)
        mag = self._rng.uniform(*h.get("mag_range", (1.0, 1.0)), (C, 1))
        w = torch.from_numpy(self._rng.normal(0, np.deg2rad(h.get("rot_deg", 3.0)), (C, 3)) * mag).float()
        sig = np.array([h.get("trans_m", 0.03) * (h.get("wrist_trans_scale", 0.25) if c == "wrist" else 1) for c in CAMS])
        tr = torch.from_numpy(self._rng.normal(0, 1, (C, 3)) * sig[:, None] * mag).float()
        T_noisy = out["T_base_cam"] @ make_T(so3_exp(w), tr)
        if self._T_init is not None:                            # sequential eval: hypothesis camera given by the caller
            T_noisy = self._T_init.to(T_noisy.dtype)
        Tcb = inv_T(T_noisy)
        pc = transform(Tcb, pw.expand(C, -1, -1))
        nc = nw.expand(C, -1, -1) @ Tcb[:, :3, :3].transpose(-1, -2)
        cams = dict(K=out["cam_K"], dist=out["cam_dist"], kind=out["cam_kind"], size=out["cam_size"])
        Hp, Wp = self.pad_hw
        lab, dep = splat_labels(pc, R[f"link_{src}"] + 1, cams, Hp, Wp, self._bank.spacing,
                                r_max=h.get("r_max", 3), n_cam=nc)
        lab = lab * out["valid"]
        res = dict(hyp_lab=lab, hyp_dep=dep * out["valid"], T_noisy=T_noisy)
        if h.get("corr", 0):
            res.update(self._correspondences(pw, pc, nc, dep, out, cams, h["corr"], pw_tgt=pw_tgt,
                                             P=R[f"pts_{src}"], L=R[f"link_{src}"]))
        return res

    def _correspondences(self, pw, pc, nc, dep, out, cams, M, pw_tgt=None, P=None, L=None):
        """M hypothesis-visible robot points per view: base-frame X, pixel in the hypothesis (noisy camera) and in
        the true camera. Their difference is the 2D displacement the refiner predicts; the pose then follows from
        the 2D-3D matches (Gauss-Newton), so the network never regresses a 6-DoF pose directly."""
        from .camera import project
        from .geometry import inv_T, transform
        C = pc.shape[0]
        uvh, okh = project(pc, cams)
        Hp, Wp = dep.shape[-2:]
        ui, vi = uvh[..., 0].round().long(), uvh[..., 1].round().long()
        inside = okh & (ui >= 0) & (vi >= 0) & (ui < cams["size"][:, None, 0]) & (vi < cams["size"][:, None, 1])
        ui, vi = ui.clamp(0, Wp - 1), vi.clamp(0, Hp - 1)
        zbuf = dep[torch.arange(C)[:, None], vi, ui]
        vis = inside & ((nc * pc).sum(-1) < 0) & (zbuf > 0) & ((pc[..., 2] - zbuf).abs() < 0.01 + 0.02 * zbuf)
        cams_t = dict(cams, K=out["cam_K_true"], dist=out["cam_dist_true"])     # the image comes from the TRUE lens
        tgt = pw if pw_tgt is None else pw_tgt
        uvt, okt = project(transform(inv_T(out["T_base_cam"]), tgt.expand(C, -1, -1)), cams_t)
        vis = vis & okt
        if pw_tgt is not None:                        # the true image must show that link at the target pixel
            ut, vt = uvt[..., 0].round().long(), uvt[..., 1].round().long()
            int_ = (ut >= 0) & (vt >= 0) & (ut < cams["size"][:, None, 0]) & (vt < cams["size"][:, None, 1])
            lab = out["vis"][torch.arange(C)[:, None], vt.clamp(0, Hp - 1), ut.clamp(0, Wp - 1)].long()
            vis = vis & int_ & (lab == (L + 1)[None])
        X = torch.zeros(C, M, 3); UH = torch.zeros(C, M, 2); UT = torch.zeros(C, M, 2); W = torch.zeros(C, M)
        PL = torch.zeros(C, M, 3); LL = torch.zeros(C, M, dtype=torch.long)
        for c in range(C):
            idx = torch.nonzero(vis[c])[:, 0]
            if len(idx) == 0:
                continue
            sel = idx[torch.from_numpy(self._rng.choice(len(idx), min(M, len(idx)), replace=False))]
            n = len(sel)
            X[c, :n] = pw[0, sel]; UH[c, :n] = uvh[c, sel]; UT[c, :n] = uvt[c, sel]; W[c, :n] = 1
            if P is not None:
                PL[c, :n] = P[sel]; LL[c, :n] = L[sel]
        # corr_P / corr_L: link-local point and link (robot_bodies order) of each match, for solves that move the
        # links (camera + joint zero offsets, register.solve_pose_dq)
        return dict(corr_X=X, corr_uvh=UH, corr_uvt=UT, corr_w=W, corr_P=PL, corr_L=LL)


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


class MixedChunkSampler(Sampler):
    """Endless stream over several datasets joined with ConcatDataset (in the same order): each chunk of `chunk`
    time steps comes from one episode of dataset k with probability weights[k], so a batch mixes the sources at
    about those ratios regardless of their sizes. Each source reshuffles its episode chunks when exhausted."""

    def __init__(self, datasets, weights, chunk=4, seed=0):
        self.ds, self.chunk, self.seed = datasets, chunk, seed
        w = np.asarray(weights, float)
        self.w = w / w.sum()
        self.off = np.cumsum([0] + [len(d) for d in datasets])[:-1]

    def _chunks(self, k, rng):
        by_ep = {}
        for idx, (i, _) in enumerate(self.ds[k].items):
            by_ep.setdefault(i, []).append(idx + int(self.off[k]))
        while True:
            ch = []
            for idxs in by_ep.values():
                idxs = idxs[:]
                rng.shuffle(idxs)
                ch += [idxs[j:j + self.chunk] for j in range(0, len(idxs), self.chunk)]
            rng.shuffle(ch)
            yield from ch

    def __iter__(self):
        rng = random.Random(self.seed)
        gens = [self._chunks(k, random.Random(self.seed * 1000 + k)) for k in range(len(self.ds))]
        while True:
            k = rng.choices(range(len(self.ds)), weights=self.w)[0]
            yield from next(gens[k])

    def __len__(self):
        return sum(len(d) for d in self.ds)


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
