"""Lens intrinsics from an A4-printed ChArUco board, and how wrong that estimate is (sim).

The model does not assume a known lens, but in practice each camera's intrinsics can be measured once in a minute
by waving a printed board in front of it. This script
  * makes the board (A4, 7x5 squares of 38 mm, DICT_4X4_50) -> --print board.png (300 dpi, print at 100 %)
  * calibrates a camera from images of it (--images DIR): fits pinhole+Brown and fisheye (Kannala-Brandt), keeps the
    model with the lower held-out reprojection error, writes the dataset's camera-model dict (kind, fx, fy, cx, cy,
    dist) as JSON
  * sim study (--sim N): N random cameras from the dataset's camera distribution (shadow.camera_model.random_camera),
    the board rendered exactly through each lens (ray / plane intersection) at random hand-held poses, with image
    degradation; calibrated as above and compared with the truth. Error = angle between the true and the estimated
    pixel rays over the whole image, in pixels (x true focal length). The distribution of that error is what the
    learned model should be trained with when it is given measured instead of true intrinsics.

    python scripts/charuco_calib.py --print board.png
    python scripts/charuco_calib.py --images shots/ --out cam_front.json
    python scripts/charuco_calib.py --sim 150 --views 20 --out results/charuco_sim.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadow.camera_model import CameraModel, degrade, random_camera  # noqa: E402

A4 = (0.297, 0.210)                    # m, landscape
SQUARES, SQ, MK = (7, 5), 0.038, 0.028  # 266 x 190 mm board centred on the sheet


def make_board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard(SQUARES, SQ, MK, d)


def sheet_image(board, ppm=12000 / 1.016):
    """The whole A4 sheet as an image (ppm pixels per metre; default 300 dpi) and the board offset on it (m)."""
    W, H = int(round(A4[0] * ppm)), int(round(A4[1] * ppm))
    bw, bh = SQUARES[0] * SQ, SQUARES[1] * SQ
    img = np.full((H, W), 255, np.uint8)
    b = board.generateImage((int(round(bw * ppm)), int(round(bh * ppm))), marginSize=0, borderBits=1)
    off = ((A4[0] - bw) / 2, (A4[1] - bh) / 2)
    y0, x0 = int(round(off[1] * ppm)), int(round(off[0] * ppm))
    img[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b
    return img, off, ppm


# ------------------------------------------------------------------------------------------- sim rendering
def hfov_of(cam: CameraModel):
    """Horizontal field of view (rad): twice the largest angle from the optical axis along the centre row (robust to
    rays that fail to undistort at the very edge of wide lenses)."""
    u = np.linspace(0, cam.width - 1, 64)
    r = cam.unproject(np.c_[u, np.full_like(u, cam.cy)])
    ang = np.arccos(np.clip(r[:, 2], -1, 1))
    ang = ang[np.isfinite(ang)]
    h = 2 * float(ang.max()) if len(ang) else 0.0
    return h if h > 1e-3 else 2 * np.arctan(cam.width / 2 / cam.fx)


def random_board_pose(cam: CameraModel, rng, tilt_deg=50, target=None):
    """Board pose (R, t: p_cam = R p_board + t) as if held by hand: centre on the ray of a random pixel (edges
    included, so the distortion is observed), size 25-60 % of the field of view, tilted up to tilt_deg."""
    u, v = target if target is not None else (rng.uniform(0.1, 0.9) * cam.width, rng.uniform(0.1, 0.9) * cam.height)
    d = cam.unproject(np.array([[u, v]]))[0]
    frac = rng.uniform(0.25, 0.6)                              # board width on screen / image width
    z = -d                                                     # board normal towards the camera
    a = rng.normal(size=3); a -= (a @ z) * z; a /= np.linalg.norm(a)
    tilt = np.deg2rad(rng.uniform(0, tilt_deg))
    z = np.cos(tilt) * z + np.sin(tilt) * a
    # board frame like an OpenCV image seen from the front: x along the sheet width, y down the sheet, z INTO the
    # paper (away from the viewer), right-handed (a reflection would print the markers mirrored)
    zb = -z
    x = np.cross(rng.normal(size=3), zb); x /= np.linalg.norm(x)
    y = np.cross(zb, x)
    R = np.stack([x, y, zb], 1)
    corners = np.array([[0, 0, 0], [A4[0], 0, 0], [0, A4[1], 0], [A4[0], A4[1], 0]]) - [A4[0] / 2, A4[1] / 2, 0]
    dist = float(np.clip(A4[0] / (frac * min(hfov_of(cam), np.deg2rad(150))), 0.08, 3.0))
    for _ in range(4):                     # distance at which the sheet spans `frac` of the image width there
        uv, ok = cam.project(corners @ R.T + dist * d)
        if not ok.all():
            dist *= 1.5
            continue
        w = np.ptp(uv[:, 0]) if np.ptp(uv[:, 0]) > 1 else 1.0
        dist = float(np.clip(dist * w / (frac * cam.width), 0.08, 3.0))
    t = dist * d - R @ np.array([A4[0] / 2, A4[1] / 2, 0])
    return R, t


def render_view(cam: CameraModel, sheet, off, ppm, R, t, rng):
    """Exact image of the sheet through the lens (every pixel ray intersected with the paper plane) + background,
    lighting and real-camera degradation. Board corner coordinates are relative to the board, the sheet adds off."""
    rays = cam.ray_map().reshape(-1, 3)
    n = R[:, 2]
    den = rays @ n
    s = (t @ n) / np.where(np.abs(den) > 1e-9, den, np.nan)
    p = (s[:, None] * rays - t) @ R                            # board-plane coordinates (sheet origin)
    ok = (s > 0) & np.isfinite(s)
    mx = np.where(ok, p[:, 0] * ppm, -1).reshape(cam.height, cam.width).astype(np.float32)
    my = np.where(ok, p[:, 1] * ppm, -1).reshape(cam.height, cam.width).astype(np.float32)
    paper = cv2.remap(sheet, mx, my, cv2.INTER_AREA if False else cv2.INTER_LINEAR, borderValue=-1)
    inside = (mx >= 0) & (my >= 0) & (mx < sheet.shape[1] - 1) & (my < sheet.shape[0] - 1)
    bg = cv2.GaussianBlur(rng.uniform(0, 255, (cam.height // 8 + 1, cam.width // 8 + 1)).astype(np.float32), (0, 0), 2)
    bg = cv2.resize(bg, (cam.width, cam.height)) * rng.uniform(0.3, 0.9)
    img = np.where(inside, paper.astype(np.float32) * rng.uniform(0.6, 1.0), bg)
    yy, xx = np.mgrid[0:cam.height, 0:cam.width]
    g = rng.normal(size=2) * 0.3
    img *= 1 + g[0] * (xx / cam.width - 0.5) + g[1] * (yy / cam.height - 0.5)            # uneven light
    img = np.clip(img, 0, 255).astype(np.uint8)
    return degrade(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), rng, strength=0.6)


# ------------------------------------------------------------------------------------------- calibration
def detect(board, img):
    det = cv2.aruco.CharucoDetector(board)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) < 8:
        return None
    obj, imgp = board.matchImagePoints(cc, ci)
    return obj.reshape(-1, 3).astype(np.float64), imgp.reshape(-1, 2).astype(np.float64)


def _fit(kind, obs, size):
    W, H = size
    O = [o.astype(np.float32) for o, _ in obs]
    P = [p.astype(np.float32) for _, p in obs]
    if kind == "fisheye":
        K = np.array([[W / 2, 0, W / 2], [0, W / 2, H / 2], [0, 0, 1.0]])
        D = np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, _, _ = cv2.fisheye.calibrate([o[None].astype(np.float64) for o in O], [p[None].astype(np.float64) for p in P],
                                                (W, H), K, D, flags=flags,
                                                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10))
        dist = np.r_[D.ravel(), 0.0]
    else:
        flags = cv2.CALIB_FIX_K3 if kind == "brown3" else 0
        rms, K, D, _, _ = cv2.calibrateCamera(O, P, (W, H), None, None, flags=flags)
        dist = np.asarray(D, float).ravel()[:5]
        kind = "brown"
    return CameraModel(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2], kind, dist), float(rms)


def _holdout_px(cam: CameraModel, obs):
    """Reprojection error on views not used in the fit: pose from PnP on the undistorted rays, then lens projection."""
    errs = []
    for o, p in obs:
        rays = cam.unproject(p)
        n = rays[:, :2] / rays[:, 2:3]
        ok, rv, tv = cv2.solvePnP(o, n, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        Rm = cv2.Rodrigues(rv)[0]
        uv, valid = cam.project(o @ Rm.T + tv.ravel())
        errs.append(np.linalg.norm(uv[valid] - p[valid], axis=1))
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else np.inf


def calibrate(obs, size, rng, kinds=("brown3", "brown", "fisheye")):
    """obs: list of (board points (N,3), pixels (N,2)). Picks the lens model by held-out error (every 5th view held
    out), then refits it on all views. Returns (CameraModel, info)."""
    idx = np.arange(len(obs)); rng.shuffle(idx)
    ho = set(idx[::5].tolist())
    tr = [obs[i] for i in range(len(obs)) if i not in ho]
    te = [obs[i] for i in sorted(ho)]
    info, fits = {}, {}
    for k in kinds:
        try:
            c, rms = _fit(k, tr, size)
            fits[k] = c
            info[k] = dict(rms=rms, holdout=_holdout_px(c, te))
        except cv2.error as e:
            info[k] = dict(error=str(e)[:80])
    ok = {k: v["holdout"] for k, v in info.items() if "holdout" in v and np.isfinite(v["holdout"])}
    if not ok:
        return None, info
    # near-ties go to the simpler model: brown3 (k1 k2 p1 p2) < fisheye (k1..k4) < brown (+k3); extra terms only
    # fit the covered area better and extrapolate badly to the rest of the image
    lo = min(ok.values())
    best = next(k for k in ("brown3", "fisheye", "brown") if k in ok and ok[k] <= lo * 1.15 + 0.02)
    try:
        cam, rms = _fit(best, obs, size)
        info["rms_all"] = rms
    except cv2.error as e:                  # the all-view refit can fail to initialise: keep the training-view fit
        cam = fits[best]
        info["refit_error"] = str(e)[:80]
    info["chosen"] = best
    return cam, info


def ray_error_px(gt: CameraModel, est: CameraModel, n=(48, 27)):
    """Angle between true and estimated pixel rays over an image grid, as pixels (x true focal length)."""
    u, v = np.meshgrid(np.linspace(0, gt.width - 1, n[0]), np.linspace(0, gt.height - 1, n[1]))
    uv = np.c_[u.ravel(), v.ravel()]
    a, b = gt.unproject(uv), est.unproject(uv)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[ok], b[ok]
    ang = np.arccos(np.clip((a * b).sum(1), -1, 1))
    # a common rotation of all rays is the same as a slightly rotated camera: the pose estimate absorbs it, so the
    # error that matters for registration is what remains after the best rotation (Kabsch on the unit rays)
    U, _, Vt = np.linalg.svd(b.T @ a)
    Q = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    ang_r = np.arccos(np.clip(((b @ Q.T) * a).sum(1), -1, 1))
    px, px_r = ang * gt.fx, ang_r * gt.fx
    return dict(median_px=float(np.median(px)), p95_px=float(np.percentile(px, 95)),
                rot_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1)))),
                aligned_median_px=float(np.median(px_r)), aligned_p95_px=float(np.percentile(px_r, 95)),
                aligned_max_px=float(px_r.max()), valid_frac=float(ok.mean()))


def cam_dict(c: CameraModel):
    return dict(width=c.width, height=c.height, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy, kind=c.kind, dist=np.asarray(c.dist).tolist())


# ------------------------------------------------------------------------------------------- main
GRID = (6, 4)


def coverage(cells, uv, size):
    i = np.clip((uv[:, 0] / size[0] * GRID[0]).astype(int), 0, GRID[0] - 1)
    j = np.clip((uv[:, 1] / size[1] * GRID[1]).astype(int), 0, GRID[1] - 1)
    cells[j, i] = True


def sim_study(n, views, seed, max_tries=80):
    """Guided capture, as the interactive tool will do it: the board is brought to cells of a 6x4 image grid that no
    detected corner has covered yet, until all cells are covered and `views` views are detected (or max_tries)."""
    board = make_board()
    sheet, off, ppm = sheet_image(board, ppm=6000)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        gt = random_camera(rng)
        obs = []
        cells = np.zeros(GRID[::-1], bool)
        for _ in range(max_tries):
            if len(obs) >= views and cells.all():
                break
            free = np.argwhere(~cells)
            if len(free):
                cj, ci = free[rng.integers(len(free))]
                target = ((ci + rng.uniform(0.2, 0.8)) / GRID[0] * gt.width, (cj + rng.uniform(0.2, 0.8)) / GRID[1] * gt.height)
            else:
                target = None
            try:
                R, t = random_board_pose(gt, rng, target=target)
                img = render_view(gt, sheet, off, ppm, R, t, rng)
                d = detect(board, img)
            except Exception as e:                      # one bad view must not stop the study
                print(f"  view failed: {type(e).__name__}: {e}", flush=True)
                continue
            if d is not None:
                obj, uv = d
                obs.append((obj, uv))
                coverage(cells, uv, (gt.width, gt.height))
        row = dict(coverage=float(cells.mean()), i=i, gt=cam_dict(gt), hfov_deg=float(np.rad2deg(hfov_of(gt))), views_ok=len(obs))
        if len(obs) >= 6:
            try:
                est, info = calibrate(obs, (gt.width, gt.height), rng)
            except Exception as e:
                est, info = None, dict(error=f"{type(e).__name__}: {str(e)[:80]}")
            row["fit"] = info
            if est is not None:
                row["est"] = cam_dict(est)
                row["err"] = ray_error_px(gt, est)
                row["f_rel_err"] = float(est.fx / gt.fx - 1) if est.kind == gt.kind or gt.kind != "fisheye" else None
        rows.append(row)
        e = row.get("err", {})
        print(f"[{i + 1}/{n}] {gt.kind:7s} hfov {row['hfov_deg']:5.1f} {gt.width}x{gt.height} views {len(obs):2d} -> "
              f"{row.get('fit', {}).get('chosen', '-'):7s} ray err median {e.get('median_px', float('nan')):6.2f} px "
              f"(common rotation {e.get('rot_deg', float('nan')):.2f} deg) -> after it median "
              f"{e.get('aligned_median_px', float('nan')):5.2f} p95 {e.get('aligned_p95_px', float('nan')):5.2f} px "
              f"f err {100 * (row.get('f_rel_err') or float('nan')):+.1f} %", flush=True)
    return rows


def summary(rows):
    out = {}
    for k in ("pinhole", "brown", "fisheye", "all"):
        R = [r for r in rows if k == "all" or r["gt"]["kind"] == k]
        E = [r for r in R if "err" in r]
        if not R:
            continue
        g = lambda key: np.array([r["err"][key] for r in E])
        out[k] = dict(cameras=len(R), calibrated=len(E),
                      aligned_median_px=dict(median=float(np.median(g("aligned_median_px"))), p90=float(np.percentile(g("aligned_median_px"), 90))),
                      aligned_p95_px=dict(median=float(np.median(g("aligned_p95_px"))), p90=float(np.percentile(g("aligned_p95_px"), 90))),
                      common_rot_deg=dict(median=float(np.median(g("rot_deg"))), p90=float(np.percentile(g("rot_deg"), 90))),
                      ray_err_median_px=dict(median=float(np.median(g("median_px"))), p90=float(np.percentile(g("median_px"), 90))),
                      ray_err_p95_px=dict(median=float(np.median(g("p95_px"))), p90=float(np.percentile(g("p95_px"), 90))),
                      f_rel_err_abs=dict(median=float(np.median([abs(r["f_rel_err"]) for r in E if r.get("f_rel_err") is not None] or [np.nan])),
                                         p90=float(np.percentile([abs(r["f_rel_err"]) for r in E if r.get("f_rel_err") is not None] or [np.nan], 90))),
                      chosen={c: sum(r["fit"].get("chosen") == c for r in E) for c in ("brown3", "brown", "fisheye")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="print_png", help="write the A4 sheet (300 dpi) to this PNG")
    ap.add_argument("--images", help="calibrate a real camera from the images in this directory")
    ap.add_argument("--sim", type=int, default=0, help="sim study over this many random cameras")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.print_png:
        img, _, _ = sheet_image(make_board())
        cv2.imwrite(a.print_png, img)
        print(f"wrote {a.print_png} ({img.shape[1]}x{img.shape[0]} px = A4 at 300 dpi; print at 100 %, measure one square: "
              f"{SQ * 1000:.0f} mm)")
    if a.images:
        board = make_board()
        files = sorted(glob.glob(os.path.join(a.images, "*")))
        obs, size = [], None
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            size = (img.shape[1], img.shape[0])
            d = detect(board, img)
            if d is not None:
                obs.append(d)
        print(f"{len(obs)}/{len(files)} images with the board")
        cam, info = calibrate(obs, size, np.random.default_rng(a.seed))
        print(json.dumps(info, indent=1))
        if cam is not None and a.out:
            json.dump(dict(model=cam_dict(cam), fit=info), open(a.out, "w"), indent=1)
            print(f"wrote {a.out}: {cam.describe()}")
    if a.sim:
        rows = sim_study(a.sim, a.views, a.seed)
        s = summary(rows)
        print(json.dumps(s, indent=1))
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(dict(summary=s, views=a.views, seed=a.seed, cameras=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
