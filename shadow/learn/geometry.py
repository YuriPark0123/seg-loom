"""Small batched SE(3) helpers (torch). Quaternions follow the dataset conventions:
arrays.npz poses are [x, y, z, qx, qy, qz, qw] (scipy), render_geoms / meta quats are [w, x, y, z] (MuJoCo)."""
from __future__ import annotations

import torch


def rpy_to_R(rpy: torch.Tensor) -> torch.Tensor:
    """URDF fixed-axis roll-pitch-yaw (..., 3) -> (..., 3, 3), R = Rz(y) Ry(p) Rx(r)."""
    r, p, y = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        torch.stack([-sp, cp * sr, cp * cr], -1)], -2)


def axis_angle_to_R(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis (..., 3) unit, angle (...) -> (..., 3, 3)."""
    x, y, z = axis.unbind(-1)
    c, s = angle.cos(), angle.sin()
    C = 1 - c
    return torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) -> (..., 3, 3); smooth at 0."""
    th = w.norm(dim=-1, keepdim=True)
    small = th < 1e-6
    axis = torch.where(small, torch.zeros_like(w) + torch.tensor([1.0, 0, 0], device=w.device, dtype=w.dtype),
                       w / th.clamp_min(1e-12))
    R = axis_angle_to_R(axis, th.squeeze(-1))
    # first-order for tiny angles keeps the gradient finite
    K = skew(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return torch.where(small[..., None], I + K, R)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([torch.stack([o, -z, y], -1), torch.stack([z, o, -x], -1), torch.stack([-y, x, o], -1)], -2)


def make_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.zeros(*R.shape[:-2], 4, 4, device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def inv_T(T: torch.Tensor) -> torch.Tensor:
    R, t = T[..., :3, :3], T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    return make_T(Rt, -(Rt @ t[..., None])[..., 0])


def transform(T: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """T (..., 4, 4), p (..., N, 3) -> (..., N, 3)."""
    return p @ T[..., :3, :3].transpose(-1, -2) + T[..., None, :3, 3]


def quat_xyzw_to_R(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def quat_wxyz_to_R(q: torch.Tensor) -> torch.Tensor:
    return quat_xyzw_to_R(torch.cat([q[..., 1:], q[..., :1]], -1))


def pose7_to_T(p: torch.Tensor) -> torch.Tensor:
    """[x, y, z, qx, qy, qz, qw] (..., 7) -> (..., 4, 4)."""
    return make_T(quat_xyzw_to_R(p[..., 3:]), p[..., :3])


def rot_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (rad) between rotation matrices."""
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.arccos(c.clamp(-1 + 1e-7, 1 - 1e-7))
