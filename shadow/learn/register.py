"""Camera pose from 2D-3D correspondences through any lens model (the geometric half of the M3 refiner).

The network predicts where hypothesis points really are in the image (a 2D displacement); this module turns those
matches into a camera pose by Levenberg-Marquardt on the reprojection error, using the same differentiable lens
projection as everything else (pinhole / Brown / fisheye), with Huber weights against bad matches.

    T = solve_pose(T_init, X_base, uv_target, weight, cams)      # T_base_cam (V, 4, 4)
"""
from __future__ import annotations

import torch

from .camera import project
from .geometry import inv_T, make_T, so3_exp, transform


def _reproject(T_init, xi, X, cams):
    T = T_init @ make_T(so3_exp(xi[:, :3]), xi[:, 3:])
    return project(transform(inv_T(T), X), cams)


def solve_pose(T_init: torch.Tensor, X: torch.Tensor, uv: torch.Tensor, w: torch.Tensor, cams: dict,
               iters: int = 12, huber_px: float = 2.0, eps: float = 1e-5, min_pts: int = 6,
               prior_sigma: tuple | None = None, prior_px: float = 2.0):
    """T_init (V, 4, 4) base<-camera, X (V, M, 3) base-frame points, uv (V, M, 2) target pixels, w (V, M) >= 0.
    Optimises a right-multiplied correction T = T_init @ exp(xi) (camera-frame rotation vector + translation).
    prior_sigma = (rad, m): Gaussian prior on xi around T_init (MAP). One view constrains depth / the rotation-
    translation trade-off only weakly, so without it noisy matches push the camera far along those directions.
    Views with fewer than min_pts usable matches keep T_init. Returns T (V, 4, 4) and the final RMS error (V,)."""
    dt = torch.float64
    T0, X, uv, w = T_init.to(dt), X.to(dt), uv.to(dt), w.to(dt)
    cams = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in cams.items()}
    V = T0.shape[0]
    xi = torch.zeros(V, 6, dtype=dt, device=T0.device)
    lam = torch.full((V,), 1e-3, dtype=dt, device=T0.device)
    cams7 = {k: v.repeat_interleave(7, 0) for k, v in cams.items()}
    E = torch.eye(6, dtype=dt, device=T0.device) * eps

    if prior_sigma is not None:
        # a 1-sigma correction costs as much as `prior_px` of RMS reprojection error over all of the view's matches
        inv = torch.tensor([1 / prior_sigma[0] ** 2] * 3 + [1 / prior_sigma[1] ** 2] * 3, dtype=dt, device=T0.device)
        pw = (w > 0).sum(1, keepdim=True).to(dt) * prior_px ** 2 * inv                       # (V, 6)
    else:
        pw = torch.zeros(V, 6, dtype=dt, device=T0.device)

    def cost(r, ww, x):
        return (ww * (r ** 2).sum(-1)).sum(-1) + (pw * x * x).sum(-1)

    for _ in range(iters):
        xis = torch.cat([xi[:, None], xi[:, None] + E[None]], 1).reshape(V * 7, 6)
        p, ok = _reproject(T0.repeat_interleave(7, 0), xis, X.repeat_interleave(7, 0), cams7)
        p, ok = p.view(V, 7, -1, 2), ok.view(V, 7, -1)
        r = p[:, 0] - uv                                             # (V, M, 2)
        J = (p[:, 1:] - p[:, :1]) / eps                              # (V, 6, M, 2)
        okall = ok.all(1) & (w > 0)
        e = r.norm(dim=-1)
        ww = w * okall * torch.where(e < huber_px, torch.ones_like(e), huber_px / e.clamp_min(1e-9))
        Jf = J.permute(0, 2, 3, 1).reshape(V, -1, 6)                 # (V, 2M, 6)
        wf = ww.repeat_interleave(2, 1)
        H = Jf.transpose(1, 2) @ (Jf * wf[..., None]) + torch.diag_embed(pw)
        g = Jf.transpose(1, 2) @ (wf * r.reshape(V, -1))[..., None] + (pw * xi)[..., None]
        Hd = H + lam[:, None, None] * torch.diag_embed(H.diagonal(dim1=1, dim2=2).clamp_min(1e-9))
        step = -torch.linalg.solve(Hd, g)[..., 0]
        # accept / reject (LM)
        pn, okn = _reproject(T0, xi + step, X, cams)
        c0 = cost(r, ww, xi)
        c1 = cost(pn - uv, ww * okn, xi + step)
        better = c1 < c0
        xi = torch.where(better[:, None], xi + step, xi)
        lam = torch.where(better, lam * 0.3, lam * 10).clamp(1e-7, 1e7)
    n_ok = (w > 0).sum(1)
    use = n_ok >= min_pts
    T = T0 @ make_T(so3_exp(xi[:, :3]), xi[:, 3:])
    T = torch.where(use[:, None, None], T, T0)
    p, ok = project(transform(inv_T(T), X), cams)
    rms = (((p - uv) ** 2).sum(-1) * w * ok).sum(1).div((w * ok).sum(1).clamp_min(1)).sqrt()
    return T.to(T_init.dtype), rms
