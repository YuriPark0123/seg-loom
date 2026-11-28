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


def solve_pose_dq(Ta_init, P, L, q, kin, names, bidx, uv, w, cams, cam_of, anchor_link, dqmask,
                  dq0=None, iters=10, huber_px=2.0, eps=1e-5, prior_sigma=None, prior_px=0.3, dq_sigma=0.05):
    """Cameras + joint zero offsets from the matches of a window of frames of ONE robot (scripts/diag_pose_dq.py:
    with true-surface targets a camera-only solve keeps ~4 deg from the approx URDF's joint zeros; jointly ~0.9).

    Every camera c is constant in its anchor frame over the window: base (anchor_link[c] = None) or a robot link
    (wrist camera on the end effector, anchor_link[c] = link index of kin.links). The approx-URDF links move with
    q + dq, dq (J,) one offset per joint, the same for every frame (encoder zero error), masked by dqmask (J,).
      Ta_init (C, 4, 4)  anchor<-camera initial guesses
      P (N, M, 3), L (N, M) link-local match points and their link (index into bidx); N = window views
      q (N, J) encoder values of each view's frame, uv / w (N, M, 2) / (N, M) target pixels and weights
      cams dict of (N, ...) lens of each view, cam_of (N,) camera index of each view
    prior_sigma = (rad, m): Gaussian prior on each camera around Ta_init (weight like solve_pose); dq_sigma (rad).
    Returns Ta (C, 4, 4), dq (J,), base<-camera of every view (N, 4, 4)."""
    dt = torch.float64
    dev = Ta_init.device
    Ta0, P, uv, w, q = Ta_init.to(dt), P.to(dt), uv.to(dt), w.to(dt), q.to(dt)
    cams = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in cams.items()}
    C, J, (N, M) = Ta0.shape[0], q.shape[1], L.shape
    dqmask = dqmask.to(dt)
    xi = torch.zeros(C, 6, dtype=dt, device=dev)
    dq = dq0.to(dt).clone() if dq0 is not None else torch.zeros(J, dtype=dt, device=dev)   # fixed where dqmask = 0
    jf = torch.nonzero(dqmask > 0)[:, 0]
    nf = 6 * C + len(jf)
    cam_of = cam_of.to(dev)
    if prior_sigma is not None:
        inv = torch.tensor([1 / prior_sigma[0] ** 2] * 3 + [1 / prior_sigma[1] ** 2] * 3, dtype=dt, device=dev)
        cnt = torch.zeros(C, dtype=dt, device=dev).index_add_(0, cam_of, (w > 0).sum(1).to(dt))
        pw = (cnt[:, None] * prior_px ** 2 * inv).reshape(-1)
    else:
        pw = torch.zeros(6 * C, dtype=dt, device=dev)
    # dq prior: weak, absolute (1 sigma costs 1 px^2 in total); scaling it with the match count like the camera prior
    # held the offsets near 0 (joint error 1.7 deg instead of 0.6 deg with oracle matches)
    pdq = torch.full((len(jf),), 1 / dq_sigma ** 2, dtype=dt, device=dev)
    prior = torch.cat([pw, pdq])

    def links(dqs):                                            # dqs (S, J) -> link poses (S, N, Lall, 4, 4)
        S = dqs.shape[0]
        Tl = kin.fk((q[None] + dqs[:, None]).reshape(S * N, J), names)
        return Tl.reshape(S, N, *Tl.shape[1:])

    def views(Tl, xis):                                        # base<-cam (S, N, 4, 4)
        Ta = Ta0[None] @ make_T(so3_exp(xis[..., :3].reshape(-1, 3)), xis[..., 3:].reshape(-1, 3)).reshape(xis.shape[0], C, 4, 4)
        Tv = Ta[:, cam_of]                                      # (S, N, 4, 4)
        anc = torch.eye(4, dtype=dt, device=dev).expand(Tl.shape[0], N, 4, 4).clone()
        for c, li in enumerate(anchor_link):
            if li is not None:
                sel = cam_of == c
                anc[:, sel] = Tl[:, sel, li]
        return anc @ Tv

    def points(Tl):                                            # (S, N, M, 3) base frame
        Tb = Tl[:, :, bidx]                                     # (S, N, B, 4, 4)
        T = torch.gather(Tb, 2, L[None, :, :, None, None].expand(Tl.shape[0], N, M, 4, 4))
        return (T[..., :3, :3] @ P[None, ..., None])[..., 0] + T[..., :3, 3]

    def proj(X, Tv):                                           # (S, N, M, 2), ok
        S = X.shape[0]
        pc = transform(inv_T(Tv).reshape(S * N, 4, 4), X.reshape(S * N, M, 3))
        c = {k: v.repeat(S, *([1] * (v.dim() - 1))) for k, v in cams.items()}
        p, ok = project(pc, c)
        return p.reshape(S, N, M, 2), ok.reshape(S, N, M)

    def evaluate(xi_, dq_):
        Tl = links(dq_[None])
        p, ok = proj(points(Tl), views(Tl, xi_[None]))
        return p[0] - uv, ok[0] & (w > 0)

    def cost(r, ww, xi_, dq_):
        th = torch.cat([xi_.reshape(-1), dq_[jf]])
        return float((ww * (r ** 2).sum(-1)).sum() + (prior * th * th).sum())

    lam = 1e-3
    for _ in range(iters):
        # camera columns: perturb xi only (links unchanged); dq columns: perturb dq (links recomputed)
        Tl0 = links(dq[None])
        X0 = points(Tl0)
        E = torch.eye(6 * C, dtype=dt, device=dev).reshape(6 * C, C, 6) * eps
        xis = torch.cat([xi[None], xi[None] + E], 0)                               # (1 + 6C, C, 6)
        pc_, okc = proj(X0.expand(xis.shape[0], -1, -1, -1), views(Tl0.expand(xis.shape[0], *Tl0.shape[1:]), xis))
        Ed = torch.zeros(len(jf), J, dtype=dt, device=dev)
        Ed[torch.arange(len(jf)), jf] = eps
        if len(jf):
            Tld = links(dq[None] + Ed)
            pd_, okd = proj(points(Tld), views(Tld, xi[None].expand(len(jf), C, 6)))
        r = pc_[0] - uv
        Jc = (pc_[1:] - pc_[:1]) / eps                                             # (6C, N, M, 2)
        Jd = (pd_ - pc_[:1]) / eps if len(jf) else Jc[:0]
        Jm = torch.cat([Jc, Jd], 0)
        ok = okc.all(0) & (okd.all(0) if len(jf) else True) & (w > 0)
        e = r.norm(dim=-1)
        ww = w * ok * torch.where(e < huber_px, torch.ones_like(e), huber_px / e.clamp_min(1e-9))
        Jf = Jm.reshape(nf, -1).T                                                  # (2NM, F)
        wf = ww.reshape(-1).repeat_interleave(2)
        th = torch.cat([xi.reshape(-1), dq[jf]])
        H = Jf.T @ (Jf * wf[:, None]) + torch.diag(prior)
        g = Jf.T @ (wf * r.reshape(-1)) + prior * th
        step = -torch.linalg.solve(H + lam * torch.diag(H.diagonal().clamp_min(1e-9)), g)
        xn = xi + step[:6 * C].reshape(C, 6)
        dn = dq.clone()
        dn[jf] = dq[jf] + step[6 * C:]
        rn, okn = evaluate(xn, dn)
        if cost(rn, ww * okn, xn, dn) < cost(r, ww, xi, dq):
            xi, dq, lam = xn, dn, lam * 0.3
        else:
            lam = min(lam * 10, 1e7)
    Tl = links(dq[None])
    Ta = Ta0 @ make_T(so3_exp(xi[:, :3]), xi[:, 3:])
    return Ta, dq, views(Tl, xi[None])[0]
