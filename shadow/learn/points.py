"""Robot surface points and a z-buffered point-splat label renderer (torch, any camera model).

Two sources of robot geometry:
  points_from_render_geoms(npz)  exact MuJoCo visual geometry of the TRUE robot, incl. attachments (checks, labels)
  points_from_urdf(kin)          visuals of a URDF, e.g. approx.urdf: what the model is allowed to see

Both return (P (N, 3) in the link frame, link (N,) index) so that world points = T_link[link] @ P.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import trimesh

from .camera import project
from .geometry import quat_wxyz_to_R

# MuJoCo mjtGeom ids used in render_geoms.npz
G_SPHERE, G_CAPSULE, G_ELLIPSOID, G_CYLINDER, G_BOX, G_MESH = 2, 3, 4, 5, 6, 7


def _sample(mesh: trimesh.Trimesh, spacing: float, rng: np.random.Generator, n_min=64):
    n = int(np.clip(mesh.area / spacing ** 2, n_min, 1_500_000))
    pts, fi = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    return np.asarray(pts, np.float32), np.asarray(mesh.face_normals[fi], np.float32)


def _primitive(gtype, size):
    if gtype == G_BOX:
        return trimesh.creation.box(extents=2 * np.asarray(size[:3]))
    if gtype == G_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=48)
    if gtype == G_CAPSULE:
        return trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[24, 24])
    if gtype == G_SPHERE:
        return trimesh.creation.icosphere(subdivisions=3, radius=size[0])
    if gtype == G_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        m.apply_scale(np.asarray(size[:3]))
        return m
    raise ValueError(f"geom type {gtype}")


def _spacing_for(spacing, link):
    """spacing: float, or dict {link index: spacing, None: default} (e.g. fine sampling of links next to a camera)."""
    return spacing.get(link, spacing[None]) if isinstance(spacing, dict) else spacing


def point_spacing(spacing, link: np.ndarray) -> np.ndarray:
    """Per-point spacing array (for splat_labels) matching the sampling of points_from_*."""
    if not isinstance(spacing, dict):
        return np.full(len(link), spacing, np.float32)
    return np.array([_spacing_for(spacing, int(l)) for l in link], np.float32)


def points_from_render_geoms(path: str, spacing=0.0015, seed=0):
    """render_geoms.npz -> P (N, 3) body-frame points, Nrm (N, 3) outward normals, link (N,) robot_bodies index,
    attach (N,) bool. spacing: float or {link index: spacing, None: default}."""
    g = np.load(path)
    rng = np.random.default_rng(seed)
    P, Nm, L, A = [], [], [], []
    for i in range(len(g["body"])):
        t = int(g["type"][i])
        mesh = trimesh.Trimesh(g[f"mesh{g['mesh'][i]}_v"], g[f"mesh{g['mesh'][i]}_f"], process=False) if t == G_MESH \
            else _primitive(t, g["size"][i])
        p, nrm = _sample(mesh, _spacing_for(spacing, int(g["body"][i])), rng)
        R = quat_wxyz_to_R(torch.tensor(g["quat"][i], dtype=torch.float64)).numpy().astype(np.float32)
        P.append(p @ R.T + g["pos"][i].astype(np.float32))
        Nm.append(nrm @ R.T)
        L.append(np.full(len(p), int(g["body"][i])))
        A.append(np.full(len(p), bool(g["is_attachment"][i])))
    return np.concatenate(P), np.concatenate(Nm), np.concatenate(L), np.concatenate(A)


def _vec(s, default):
    return np.array([float(x) for x in s.split()]) if s else np.array(default, float)


def points_from_urdf(kin, spacing=0.0015, seed=0, link_names: list[str] | None = None):
    """Visual geometry of a URDF (box / cylinder / sphere / mesh with scale). link index refers to
    link_names (default kin.links). Returns P (N, 3), Nrm (N, 3), link (N,).
    spacing: float or {link index: spacing, None: default}."""
    from .geometry import rpy_to_R
    rng = np.random.default_rng(seed)
    names = link_names or kin.links
    idx = {n: i for i, n in enumerate(names)}
    P, Nm, L = [], [], []
    for ln, el in kin.link_el.items():
        if ln not in idx:
            continue
        for vis in el.findall("visual"):
            geo = vis.find("geometry")
            if geo is None or len(geo) == 0:
                continue
            g = geo[0]
            if g.tag == "box":
                mesh = trimesh.creation.box(extents=_vec(g.get("size"), [0, 0, 0]))
            elif g.tag == "cylinder":
                mesh = trimesh.creation.cylinder(radius=float(g.get("radius")), height=float(g.get("length")), sections=48)
            elif g.tag == "sphere":
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(g.get("radius")))
            elif g.tag == "mesh":
                mesh = trimesh.load(os.path.join(kin.urdf_dir, g.get("filename")), force="mesh")
                if g.get("scale"):
                    mesh.apply_scale(_vec(g.get("scale"), [1, 1, 1]))
            else:
                continue
            o = vis.find("origin")
            xyz = _vec(o.get("xyz") if o is not None else None, [0, 0, 0])
            rpy = _vec(o.get("rpy") if o is not None else None, [0, 0, 0])
            R = rpy_to_R(torch.tensor(rpy)).numpy()
            p, nrm = _sample(mesh, _spacing_for(spacing, idx[ln]), rng)
            P.append((p @ R.T + xyz).astype(np.float32))
            Nm.append((nrm @ R.T).astype(np.float32))
            L.append(np.full(len(p), idx[ln]))
    return np.concatenate(P), np.concatenate(Nm), np.concatenate(L)


def world_points(T_link: torch.Tensor, P: torch.Tensor, link: torch.Tensor, Nrm: torch.Tensor | None = None):
    """T_link (B, L, 4, 4), P (N, 3), link (N,) -> (B, N, 3) [, rotated normals (B, N, 3)]."""
    T = T_link[:, link]                                         # (B, N, 4, 4)
    pw = (T[..., :3, :3] @ P[None, :, :, None])[..., 0] + T[..., :3, 3]
    if Nrm is None:
        return pw
    return pw, (T[..., :3, :3] @ Nrm[None, :, :, None])[..., 0]


def splat_labels(p_cam: torch.Tensor, labels: torch.Tensor, cams: dict, H: int, W: int, spacing,
                 r_max: int = 6, mask: torch.Tensor | None = None, k: float = 1.2,
                 n_cam: torch.Tensor | None = None, znear: float = 0.0):
    """Nearest-point label image. p_cam (B, N, 3) camera-frame points, labels (N,) or (B, N) int in 1..255.
    A point covers every pixel CENTER within k * (point spacing in pixels) of it (capped at r_max), which matches
    the dataset labels (one ray through each pixel center) and keeps the surface closed from close range
    (wrist camera) to far away. spacing: float or per-point tensor (N,) / (B, N) (see point_spacing). n_cam (B, N, 3) camera-frame normals enable back-face culling (OpenGL / MuJoCo
    culls back faces, so geometry around a camera that sits inside a link is invisible); points closer than
    znear are clipped. Returns (B, H, W) uint8 (0 = empty) and depth (B, H, W)."""
    B, N, _ = p_cam.shape
    dev = p_cam.device
    uv, valid = project(p_cam, cams)
    if mask is not None:
        valid = valid & mask
    if n_cam is not None:
        valid = valid & ((n_cam * p_cam).sum(-1) < 0)
    if znear > 0:
        valid = valid & (p_cam[..., 2] > znear)
    if labels.dim() == 1:
        labels = labels[None].expand(B, N)
    f = cams["K"][:, None, :2].mean(-1)
    z = p_cam[..., 2].clamp_min(1e-6)
    if torch.is_tensor(spacing) and spacing.dim() == 1:
        spacing = spacing[None]
    rad = (spacing * f / z * k).clamp(0, r_max)                  # pixels
    zq = (z * 1e4).long().clamp(0, (1 << 40) - 1)                # 0.1 mm depth units
    key = (zq << 8) | labels.long()
    big = torch.iinfo(torch.long).max
    buf = torch.full((B * H * W,), big, dtype=torch.long, device=dev)
    u0, v0 = uv[..., 0].round().long(), uv[..., 1].round().long()
    bidx = torch.arange(B, device=dev)[:, None].expand(B, N)
    R = int(min(r_max, max(0, int(torch.ceil(rad[valid].max()).item()) if valid.any() else 0)))
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            u, v = u0 + dx, v0 + dy
            d2 = (uv[..., 0] - u) ** 2 + (uv[..., 1] - v) ** 2
            ok = valid & (d2 <= rad * rad)
            ok = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            flat = (bidx[ok] * H + v[ok]) * W + u[ok]
            buf.scatter_reduce_(0, flat, key[ok], reduce="amin")
    buf = buf.view(B, H, W)
    hit = buf != big
    lab = torch.where(hit, buf & 255, torch.zeros_like(buf)).to(torch.uint8)
    depth = torch.where(hit, (buf >> 8).float() * 1e-4, torch.zeros_like(buf, dtype=torch.float))
    return lab, depth
