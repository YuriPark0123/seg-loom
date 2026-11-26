"""Visual check of a generated episode: RGB | visible labels | amodal robot | skeleton+objects, per camera.

  python scripts/view_episode.py data/eval_motion/episodes/<ep> --frames 0 30 59 --out out.jpg
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from shadow.datagen import CAMS, LABEL_OBJ0, unpack_image

PAL = (np.random.default_rng(0).uniform(40, 255, (256, 3))).astype(np.uint8)
PAL[0] = 0


def colorize(lab):
    return PAL[lab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ep")
    ap.add_argument("--frames", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.ep, "meta.json")))
    A = np.load(os.path.join(a.ep, "arrays.npz"))
    F = np.load(os.path.join(a.ep, "frames.npz"))
    T = meta["T"]
    frames = a.frames or [0, T // 2, T - 1]
    rows = []
    for t in frames:
        for c in CAMS:
            get = lambda k, kind: unpack_image(F[f"{c}_{k}"], F[f"{c}_{k}_off"], t, kind)
            rgb, vis, amr, amo = get("rgb", "jpg"), get("vis", "png"), get("am_r", "png"), get("am_o", "png")
            sk = rgb.copy()
            uv = A[f"skel2d_{c}"][t]
            for i in range(len(uv) - 1):
                if uv[i, 2] and uv[i + 1, 2] and i + 1 < len(uv) - len(meta["gripper_joints"]):
                    cv2.line(sk, tuple(int(x) for x in uv[i, :2]), tuple(int(x) for x in uv[i + 1, :2]), (255, 220, 0), 2, cv2.LINE_AA)
            for p in uv:
                if p[2]:
                    cv2.circle(sk, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)
            for k, p in enumerate(A[f"obj2d_{c}"][t]):
                if p[2]:
                    cv2.drawMarker(sk, (int(p[0]), int(p[1])), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
            row = np.concatenate([rgb, colorize(vis), colorize(amr), colorize(amo), sk], 1)
            cv2.putText(row, f"t={t} {c} {meta['cameras'][c]['model']['kind']} ({meta['cameras'][c]['motion']})", (4, 14), 0, 0.45, (255, 255, 255), 1)
            rows.append(cv2.resize(row, (5 * 256, int(256 * rgb.shape[0] / rgb.shape[1]))))
    out = a.out or os.path.join(a.ep, "preview.jpg")
    cv2.imwrite(out, np.concatenate(rows, 0)[..., ::-1])
    print(out, "|", meta["robot"], meta["mode"], (meta["task"] or {}).get("instruction"), "success", (meta["task"] or {}).get("success"))


if __name__ == "__main__":
    main()
