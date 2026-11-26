"""Recompute the in-image flags of skel2d_{cam} / obj2d_{cam} for a split generated before the lens-fold fix
(commit a411656, camera_model.CameraModel.fold_radius).

Before the fix, project() reported points far outside a strongly distorted (Brown / fisheye) lens as valid because
the lens polynomial folds back, so some flags [..., 2] were 1 for points no pixel ray reaches. The fixed generator
computes  inside = (z > 0) & (r < fold_radius) & in_bounds, the old one  (z > 0) & in_bounds, and the pixel values
themselves are unchanged, so the new flags are exactly  old & (r < fold_radius). Images and label maps were never
affected (they are rendered per pixel ray).

    python scripts/fix_fold_flags.py data/eval --dry-run       # report only
    python scripts/fix_fold_flags.py data/eval                 # rewrite arrays.npz in place (atomic per file)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadow.camera_model import CameraModel  # noqa: E402

CAMS = ("top", "front", "wrist")


def fold_ok(P_base, T_base_cam, fold):
    """P_base (T, K, 3), T_base_cam (T, 4, 4) -> (T, K) bool: in front of the camera and inside the fold radius."""
    R, t = T_base_cam[:, :3, :3].astype(np.float64), T_base_cam[:, :3, 3].astype(np.float64)
    pc = np.einsum("tji,tkj->tki", R, P_base.astype(np.float64) - t[:, None])          # R^T (P - t)
    z = pc[..., 2]
    r = np.linalg.norm(pc[..., :2], axis=-1) / np.where(z > 1e-6, z, 1.0)
    return (z > 1e-6) & (r < fold)


def fix_episode(ed):
    meta = json.load(open(os.path.join(ed, "meta.json")))
    A = dict(np.load(os.path.join(ed, "arrays.npz")))
    changed = {}
    for c in CAMS:
        m = meta["cameras"][c]["model"]
        fold = CameraModel(m["width"], m["height"], m["fx"], m["fy"], m["cx"], m["cy"], m["kind"],
                           np.asarray(m["dist"], float)).fold_radius()
        for key, P in ((f"skel2d_{c}", A["skel3d"]), (f"obj2d_{c}", A["obj_pose"][..., :3])):
            if P.shape[1] == 0:
                continue
            old = A[key][..., 2] > 0.5
            new = old & fold_ok(P, A[f"T_base_cam_{c}"], fold)
            n = int((old & ~new).sum())
            if n:
                A[key] = A[key].copy()
                A[key][..., 2] = new.astype(A[key].dtype)
                changed[key] = n
    return A, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    eps = [json.loads(l)["episode"] for l in open(os.path.join(a.split, "manifest.jsonl"))]
    tot, n_ep, per = 0, 0, {}
    for e in eps:
        ed = os.path.join(a.split, "episodes", e)
        A, changed = fix_episode(ed)
        if not changed:
            continue
        n_ep += 1
        for k, n in changed.items():
            per[k] = per.get(k, 0) + n
            tot += n
        if not a.dry_run:
            tmp = os.path.join(ed, "arrays.tmp.npz")
            np.savez_compressed(tmp, **A)
            os.replace(tmp, os.path.join(ed, "arrays.npz"))
    print(f"{a.split}: {tot} flags cleared in {n_ep}/{len(eps)} episodes {'(dry run)' if a.dry_run else ''}")
    for k in sorted(per):
        print(f"  {k:16s} {per[k]}")


if __name__ == "__main__":
    main()
