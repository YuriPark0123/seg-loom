"""Differentiable projection for the dataset's camera models (same equations as cv2.projectPoints and
cv2.fisheye.projectPoints, which is what shadow/camera_model.py and the dataset labels use).

A batch of cameras is a dict of tensors:
  K      (B, 4)   fx, fy, cx, cy
  dist   (B, 5)   brown: k1 k2 p1 p2 k3 | fisheye: k1 k2 k3 k4 (5th unused) | pinhole: zeros
  kind   (B,)     0 pinhole, 1 brown, 2 fisheye
  size   (B, 2)   width, height
"""
from __future__ import annotations

import numpy as np
import torch

KINDS = {"pinhole": 0, "brown": 1, "fisheye": 2}


def cams_from_meta(models: list[dict], device=None, dtype=torch.float32) -> dict:
    """models: list of meta['cameras'][c]['model'] dicts -> camera batch."""
    K = torch.tensor([[m["fx"], m["fy"], m["cx"], m["cy"]] for m in models], dtype=dtype, device=device)
    dist = torch.tensor([(list(m["dist"]) + [0] * 5)[:5] for m in models], dtype=dtype, device=device)
    kind = torch.tensor([KINDS[m["kind"]] for m in models], device=device)
    size = torch.tensor([[m["width"], m["height"]] for m in models], device=device)
    return dict(K=K, dist=dist, kind=kind, size=size)


def project(p_cam: torch.Tensor, cams: dict, eps: float = 1e-6):
    """p_cam (B, N, 3) OpenCV camera-frame points -> uv (B, N, 2), valid (B, N) (in front of the camera)."""
    X, Y, Z = p_cam.unbind(-1)
    valid = Z > eps
    Zs = torch.where(valid, Z, torch.ones_like(Z))
    x, y = X / Zs, Y / Zs
    d = cams["dist"][:, None, :]
    kind = cams["kind"][:, None]
    # brown (pinhole = brown with zero coefficients)
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = d.unbind(-1)
    is_brown = (kind == 1).to(x.dtype)
    k1b, k2b, p1b, p2b, k3b = (c * is_brown for c in (k1, k2, p1, p2, k3))
    radial = 1 + k1b * r2 + k2b * r2 * r2 + k3b * r2 * r2 * r2
    xb = x * radial + 2 * p1b * x * y + p2b * (r2 + 2 * x * x)
    yb = y * radial + p1b * (r2 + 2 * y * y) + 2 * p2b * x * y
    # fisheye (Kannala-Brandt, cv2.fisheye): theta = atan(r)
    f1, f2, f3, f4 = d[..., 0], d[..., 1], d[..., 2], d[..., 3]     # fisheye k1..k4 = dist[0:4]
    r = torch.sqrt(r2 + 1e-18)
    th = torch.atan(r)
    th2 = th * th
    thd = th * (1 + f1 * th2 + f2 * th2 ** 2 + f3 * th2 ** 3 + f4 * th2 ** 4)
    s = torch.where(r > 1e-9, thd / r, torch.ones_like(r))
    xf, yf = x * s, y * s
    fe = kind == 2
    xn = torch.where(fe, xf, xb)
    yn = torch.where(fe, yf, yb)
    K = cams["K"][:, None, :]
    uv = torch.stack([K[..., 0] * xn + K[..., 2], K[..., 1] * yn + K[..., 3]], -1)
    # Lens polynomials fold back far outside the field of view (strong barrel Brown, fisheye past its range):
    # points there would land inside the image although no pixel ray reaches them. Keep only the monotonic part.
    valid = valid & (r < monotonic_limit(cams)[:, None])
    return uv, valid


def monotonic_limit(cams: dict, r_hi: float = 20.0, n: int = 2048) -> torch.Tensor:
    """(B,) largest undistorted radius r = |(X/Z, Y/Z)| up to which the radial mapping still increases
    (brown: r * radial(r); fisheye: theta_d(atan r)). inf for pinhole."""
    d = cams["dist"]
    dev, dt = d.device, d.dtype
    r = torch.linspace(1e-4, r_hi, n, device=dev, dtype=dt)[None]          # (1, n)
    k1, k2, k3 = d[:, 0:1], d[:, 1:2], d[:, 4:5]
    r2 = r * r
    rb = r * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
    th = torch.atan(r)
    t2 = th * th
    rf = th * (1 + d[:, 0:1] * t2 + d[:, 1:2] * t2 ** 2 + d[:, 2:3] * t2 ** 3 + d[:, 3:4] * t2 ** 4)
    kind = cams["kind"][:, None]
    m = torch.where(kind == 2, rf, rb)
    dec = (m[:, 1:] - m[:, :-1]) <= 0                                        # first non-increasing step
    first = torch.where(dec.any(1), dec.float().argmax(1), torch.full_like(dec[:, 0], n - 1, dtype=torch.long))
    lim = r[0, first]
    return torch.where(kind[:, 0] == 0, torch.full_like(lim, float("inf")), lim)


def in_image(uv: torch.Tensor, valid: torch.Tensor, cams: dict):
    W = cams["size"][:, None, 0]
    H = cams["size"][:, None, 1]
    return valid & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)


def ray_map_np(model: dict, v1: bool = False) -> np.ndarray:
    """(H, W, 3) unit ray per pixel from the dataset's own CameraModel. v1: data generated before 2026-11-28 15:41
    (shadow/camera_model_v1.py, the code those images were made with)."""
    if v1:
        from shadow.camera_model_v1 import CameraModel
    else:
        from shadow.camera_model import CameraModel
    cm = CameraModel(model["width"], model["height"], model["fx"], model["fy"], model["cx"], model["cy"],
                     model["kind"], np.asarray(model["dist"], float))
    return cm.ray_map().astype(np.float32)
