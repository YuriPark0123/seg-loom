"""Camera models for 'any camera': intrinsics + lens distortion + image degradation.

A CameraModel maps 3D points (OpenCV camera frame: x right, y down, z forward) to pixels and back.
Supported lens models:
  pinhole  : K only
  brown    : K + radial/tangential (k1, k2, p1, p2, k3)    - ordinary lenses (OpenCV default)
  fisheye  : K + Kannala-Brandt (k1..k4)                   - wide / fisheye lenses (cv2.fisheye)

Distorted images are produced EXACTLY from a pinhole render: every output pixel is unprojected to
its ray and looked up in a wide pinhole render (cv2.remap). The same path renders the twin, so the
twin can be drawn through any camera model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    kind: str = "pinhole"                       # pinhole | brown | fisheye
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    _rays: np.ndarray | None = field(default=None, repr=False)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def describe(self):
        r = self.unproject(np.array([[0, self.cy], [self.width - 1, self.cy]]))
        hfov = np.rad2deg(np.arccos(np.clip(r[0] @ r[1], -1, 1)))
        return f"{self.kind} {self.width}x{self.height} f={self.fx:.0f} hfov~{hfov:.0f}deg dist={np.round(self.dist, 3).tolist()}"

    # ---------------- projection
    def project(self, pts_cam: np.ndarray):
        """(N,3) camera-frame points -> (N,2) pixels, valid mask (in front of the camera)."""
        p = np.asarray(pts_cam, np.float64).reshape(-1, 3)
        valid = p[:, 2] > 1e-6
        # beyond the monotonic range the lens polynomial folds back and far-off points would land in the image
        r = np.linalg.norm(p[:, :2], axis=1) / np.where(valid, p[:, 2], 1.0)
        valid &= r < self.fold_radius()
        rv = tv = np.zeros(3)
        if self.kind == "fisheye":
            uv, _ = cv2.fisheye.projectPoints(p[None], rv, tv, self.K, self.dist[:4].reshape(4, 1))
        else:
            d = self.dist if self.kind == "brown" else np.zeros(5)
            uv, _ = cv2.projectPoints(p, rv, tv, self.K, d)
        return uv.reshape(-1, 2), valid

    def fold_radius(self, r_hi=20.0, n=2048):
        """Largest undistorted radius |(X/Z, Y/Z)| up to which the lens mapping still increases (inf: pinhole)."""
        if self.kind == "pinhole":
            return np.inf
        r = np.linspace(1e-4, r_hi, n)
        d = np.asarray(self.dist, float)
        if self.kind == "fisheye":
            th = np.arctan(r)
            t2 = th * th
            m = th * (1 + d[0] * t2 + d[1] * t2 ** 2 + d[2] * t2 ** 3 + d[3] * t2 ** 4)
        else:
            r2 = r * r
            m = r * (1 + d[0] * r2 + d[1] * r2 ** 2 + d[4] * r2 ** 3)
        dec = np.nonzero(np.diff(m) <= 0)[0]
        return r[dec[0]] if len(dec) else r[-1]

    def unproject(self, uv: np.ndarray):
        """(N,2) pixels -> (N,3) unit rays in the camera frame."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if self.kind == "fisheye":
            n = cv2.fisheye.undistortPoints(uv, self.K, self.dist[:4].reshape(4, 1)).reshape(-1, 2)
        elif self.kind == "brown":
            n = cv2.undistortPointsIter(uv, self.K, self.dist, None, None,
                                        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)).reshape(-1, 2)
        else:
            n = ((uv.reshape(-1, 2) - [self.cx, self.cy]) / [self.fx, self.fy])
        r = np.c_[n, np.ones(len(n))]
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def ray_map(self):
        """(H,W,3) unit ray per pixel - the camera-model-independent representation."""
        if self._rays is None:
            u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
            self._rays = self.unproject(np.c_[u.ravel(), v.ravel()]).reshape(self.height, self.width, 3)
        return self._rays

    def scaled(self, s):
        return CameraModel(int(round(self.width * s)), int(round(self.height * s)), self.fx * s, self.fy * s,
                           (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5, self.kind, self.dist.copy())


def random_camera(rng: np.random.Generator, kinds=("pinhole", "brown", "brown", "fisheye")) -> CameraModel:
    """Sample a plausible real camera: resolution, FOV, principal-point offset, lens distortion."""
    W, H = [(640, 480), (848, 480), (1280, 720), (1920, 1080), (640, 640), (1280, 960)][rng.integers(6)]
    kind = kinds[rng.integers(len(kinds))]
    if kind == "fisheye":
        hfov = rng.uniform(100, 175)
        fx = (W / 2) / np.tan(np.deg2rad(hfov / 2) * 0.82)            # KB lens: approx equidistant
        dist = np.r_[rng.normal(0, 0.03), rng.normal(0, 0.02), rng.normal(0, 0.01), rng.normal(0, 0.005), 0]
    else:
        hfov = rng.uniform(45, 100)
        fx = (W / 2) / np.tan(np.deg2rad(hfov / 2))
        dist = np.zeros(5)
        if kind == "brown":
            k1 = rng.uniform(-0.35, 0.15)
            dist = np.array([k1, rng.uniform(-0.1, 0.15) * (abs(k1) > 0.1), rng.normal(0, 0.002), rng.normal(0, 0.002),
                             0.0])
    fy = fx * rng.uniform(0.995, 1.005)
    cx = W / 2 - 0.5 + rng.normal(0, 0.015 * W)
    cy = H / 2 - 0.5 + rng.normal(0, 0.015 * H)
    return CameraModel(W, H, fx, fy, cx, cy, kind, dist)


# ------------------------------------------------------------------------------- rendering
class ModelRenderer:
    """Render a MuJoCo camera through an arbitrary CameraModel.

    narrow lenses : one wide pinhole render + exact per-pixel remap
    wide / fisheye: cube map (front, right, left, down, up 90-degree pinhole faces) + per-pixel lookup,
                    so fields of view up to ~180 degrees work
    """

    # face axes as columns (x, y, z) in the OpenCV camera frame
    FACES = [np.eye(3),
             np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float).T,     # right  (+x)
             np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float).T,     # left   (-x)
             np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T,     # down   (+y)
             np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float).T]     # up     (-y)

    def __init__(self, model: mujoco.MjModel, cam: CameraModel, max_src=1600):
        self.m, self.cam = model, cam
        rays = cam.ray_map()
        z = rays[..., 2]
        zc = np.clip(z, 1e-3, None)
        t = max(np.abs(rays[..., 0] / zc).max(), np.abs(rays[..., 1] / zc).max()) if (z > 0.2).all() else np.inf
        if t <= 2.5:                                     # <= ~136 deg: single render
            S = int(min(max_src, max(cam.width, cam.height) * 1.6))
            self.faces = [np.eye(3)]
            fs = (S / 2) / (t * 1.02)
            self.fovy = float(np.rad2deg(2 * np.arctan((S / 2) / fs)))
            mx, my = fs * rays[..., 0] / zc + S / 2 - 0.5, fs * rays[..., 1] / zc + S / 2 - 0.5
        else:                                            # cube map
            S = int(min(max_src // 2, max(cam.width, cam.height)))
            self.faces = self.FACES
            fs = S / 2 / 1.0                             # 90-degree faces
            self.fovy = 90.0
            loc = np.stack([rays @ F for F in self.faces], 0)          # (5,H,W,3) ray in each face frame
            face = np.argmax(loc[..., 2], 0)
            lf = np.take_along_axis(loc, face[None, ..., None].repeat(3, -1), 0)[0]
            lz = np.clip(lf[..., 2], 1e-3, None)
            mx = fs * lf[..., 0] / lz + S / 2 - 0.5 + face * S
            my = fs * lf[..., 1] / lz + S / 2 - 0.5
            behind = loc[..., 2].max(0) < 1e-3
            mx[behind] = -10; my[behind] = -10
        self.S = S
        self.map_x, self.map_y = mx.astype(np.float32), my.astype(np.float32)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, S)
        model.vis.global_.offheight = max(model.vis.global_.offheight, S)
        self.r = mujoco.Renderer(model, S, S)

    def _faces(self, data, cam_id, fn):
        """Call fn() once per face with the camera rotated onto that face; returns the face atlas."""
        m = self.m
        old_q, old_f = m.cam_quat[cam_id].copy(), float(m.cam_fovy[cam_id])
        m.cam_fovy[cam_id] = self.fovy
        outs = []
        try:
            for F in self.faces:
                if len(self.faces) > 1:
                    Rf = CV_TO_MJ @ F @ CV_TO_MJ                   # face rotation in MuJoCo camera axes
                    qf = np.zeros(4); mujoco.mju_mat2Quat(qf, Rf.flatten())
                    qn = np.zeros(4); mujoco.mju_mulQuat(qn, old_q, qf)
                    m.cam_quat[cam_id] = qn
                    mujoco.mj_camlight(m, data)
                outs.append(fn())
        finally:
            m.cam_quat[cam_id], m.cam_fovy[cam_id] = old_q, old_f
            if len(self.faces) > 1:
                mujoco.mj_camlight(m, data)
        return np.concatenate(outs, 1)

    def rgb(self, data, cam_id, opt):
        def f():
            self.r.update_scene(data, camera=cam_id, scene_option=opt)
            return self.r.render().copy()
        return cv2.remap(self._faces(data, cam_id, f), self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def geom_ids(self, data, cam_id, opt):
        from .kinematic_mask import fast_geom_segmentation
        atlas = self._faces(data, cam_id, lambda: fast_geom_segmentation(self.r, data, cam_id, opt).astype(np.float32))
        return cv2.remap(atlas, self.map_x, self.map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=-1).astype(np.int32)

    def depth(self, data, cam_id, opt):
        """Euclidean range along each pixel's ray (meters)."""
        def f():
            self.r.enable_depth_rendering()
            self.r.update_scene(data, camera=cam_id, scene_option=opt)
            z = self.r.render().copy()
            self.r.disable_depth_rendering()
            return z
        atlas = self._faces(data, cam_id, f)
        z = cv2.remap(atlas, self.map_x, self.map_y, cv2.INTER_NEAREST, borderValue=np.inf)
        rays = self.cam.ray_map()
        if len(self.faces) == 1:
            return z / np.clip(rays[..., 2], 1e-3, None)
        loc = np.stack([rays @ F for F in self.faces], 0)
        return z / np.clip(loc[..., 2].max(0), 1e-3, None)        # z-depth of the chosen face -> range


# ------------------------------------------------------------------------------- image quality
def degrade(img: np.ndarray, rng: np.random.Generator, strength=1.0):
    """Random real-camera artifacts: exposure/white balance, vignetting, blur, noise, low-res, JPEG."""
    x = img.astype(np.float32) / 255.0
    H, W = x.shape[:2]
    x = np.clip(x * rng.uniform(0.6, 1.4) ** strength, 0, 1) ** rng.uniform(0.8, 1.25)     # exposure + gamma
    x = np.clip(x * (1 + rng.normal(0, 0.06 * strength, 3)), 0, 1)                         # white balance
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    x *= (1 - rng.uniform(0, 0.45) * strength * r2)[..., None]                               # vignetting
    if rng.random() < 0.5:                                                                  # motion blur
        L = int(rng.integers(3, 4 + int(10 * strength)))
        k = np.zeros((L, L), np.float32); k[L // 2] = 1
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L / 2, L / 2), rng.uniform(0, 180), 1), (L, L))
        x = cv2.filter2D(x, -1, k / max(k.sum(), 1e-6))
    elif rng.random() < 0.5:                                                                # defocus
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 2.0) * strength)
    s = rng.uniform(0.35, 1.0) if rng.random() < 0.4 else 1.0                               # low-res sensor
    if s < 1:
        x = cv2.resize(cv2.resize(x, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA), (W, H))
    x = x + rng.normal(0, rng.uniform(0.0, 0.04) * strength, x.shape) + \
        rng.normal(0, 0.02 * strength, x.shape) * np.sqrt(np.clip(x, 0, 1))               # read + shot noise
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    q = int(rng.uniform(25, 95))
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
