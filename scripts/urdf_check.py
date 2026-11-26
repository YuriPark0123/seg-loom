"""Bring up any URDF robot (configs/robots/<name>.yaml): build the real world, calibrate all
cameras with a checkerboard, place the kinematic twin with the ESTIMATED calibration, and score the
twin projection against the real images at random poses. Saves calibrations for the web demo.

  python scripts/urdf_check.py --robot so101
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mujoco
import numpy as np

from shadow.calibration import calibrate, pose_error
from shadow.ik import IKSolver
from shadow.kinematic_mask import KinematicMaskRenderer, RobotKinematics, fast_geom_segmentation
from shadow.mj_world import CAMS, Layout, World, twin_xml, wrist_calib_pose
from shadow.skeleton import draw_skeleton, project, skeleton_3d
from shadow.urdf_robot import RobotSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scene_key(spec, layout):
    """Hash of the full real-world scene (robot, table, camera placements) the calibration belongs to."""
    import hashlib
    from shadow.mj_world import build_scene_xml
    return hashlib.md5(build_scene_xml(spec, layout).encode()).hexdigest()


def save_camera(path, cam):
    np.savez(path, K=cam.K, T=cam.T, height=cam.height, width=cam.width, dist=cam.dist,
             parent_body="" if cam.parent_body is None else cam.parent_body)


def calibrate_all(spec, layout, res=512, cams=CAMS, verbose=True):
    """Checkerboard calibration of every camera of the world. Returns {cam: (Camera, info, gt_T)}."""
    kin = RobotKinematics(xml=twin_xml(spec, layout), base_body="robot_mount")
    ws = tuple(zip(*layout.ws_box))
    out = {}
    for c in cams:
        mode = "wrist" if c == "wrist" else "static"
        scene = World(spec, layout, board="table" if mode == "wrist" else "hand", res=(res, res))
        t0 = time.time()
        kw = dict(arm_joints=spec.arm_joints, tcp_site="tcp", rot_weight=spec.rot_weight, hand_body=spec.ee_body,
                  center=np.array(spec.rest_q), pos_tol=0.02 * spec.reach, verbose=False)
        if mode == "wrist":
            kw["center"] = wrist_calib_pose(spec, layout)          # TCP lifted above the table board
            cam, info = calibrate(scene, kin, c, mode, sampler="joint", spread=np.full(len(spec.arm_joints), 0.25), **kw)
        else:
            cam, info = calibrate(scene, kin, c, mode, sampler="workspace", ws_box=ws, n_try=200, n_keep=30, **kw)
        gt = scene.T_base_cam_gt(c)
        et, er = pose_error(cam.T, gt)
        out[c] = (cam, info, gt)
        if verbose:
            print(f"  [{c:5s}] views={info['n_views']:2d} reproj={info['reproj_rms_px_joint']:.2f}px "
                  f"pose err {et:.2f} mm / {er:.3f} deg  ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--n_poses", type=int, default=40)
    args = ap.parse_args()
    spec = RobotSpec.load(args.robot)
    layout = Layout(spec)
    out_dir = os.path.join(ROOT, "out/urdf", spec.name); os.makedirs(out_dir, exist_ok=True)
    calib_dir = os.path.join(ROOT, "out/calib", spec.name); os.makedirs(calib_dir, exist_ok=True)
    print(f"== {spec.name}: reach {spec.reach} m, arm {len(spec.arm_joints)} joints, gripper {spec.gripper_joints}")

    cal = calibrate_all(spec, layout)
    open(os.path.join(calib_dir, "scene.key"), "w").write(scene_key(spec, layout))
    report = {}
    for c, (cam, info, gt) in cal.items():
        save_camera(f"{calib_dir}/{c}.npz", cam)
        et, er = pose_error(cam.T, gt)
        report[c] = dict(trans_err_mm=et, rot_err_deg=er, reproj_px=info["reproj_rms_px_joint"], views=info["n_views"])
        img = info["sample_image"].copy()
        b = (layout.board_table if c == "wrist" else layout.board_hand)
        cv2.drawChessboardCorners(img, (b["cols"] - 1, b["rows"] - 1), info["sample_corners"], True)
        cv2.imwrite(f"{out_dir}/calib_{c}.jpg", img[..., ::-1])

    # ---- twin with ESTIMATED calibration vs real images at random reachable poses
    real = World(spec, layout, res=(args.res, args.res))
    txml = twin_xml(spec, layout)
    twins = {c: KinematicMaskRenderer(xml=txml, camera=cal[c][0].resized(args.res, args.res), base_body="robot_mount")
             for c in CAMS}
    ik_kin = RobotKinematics(xml=txml, base_body="robot_mount")
    ik = IKSolver(ik_kin, site="tcp", joint_names=spec.arm_joints, rest_q=spec.rest_q, rot_weight=spec.rot_weight)
    rng = np.random.default_rng(1)
    lo, hi = layout.ws_box
    ious = {c: [] for c in CAMS}
    ik_res, rows = [], []
    q = np.array(spec.rest_q)
    for k in range(args.n_poses):
        p = rng.uniform(lo, hi)
        ik_kin.set_joints(dict(zip(spec.arm_joints, spec.rest_q))); _, R0 = ik.site_pose()
        q, ep, er = ik.solve(spec.rest_q, p, R0)
        ik_res.append(ep)
        js = dict(zip(spec.arm_joints, q))
        if spec.gripper:
            g = spec.gripper["open"] if k % 2 == 0 else spec.gripper["close"]
            js.update(dict(zip(spec.gripper_joints, g)))
        real.set_joints(js)
        enc = twins["top"].joints_from_mj(real.model, real.data)          # encoders only
        row = []
        for c in CAMS:
            tw = twins[c]
            mask = tw.render(enc)
            gid = fast_geom_segmentation(real.renderer, real.data, c, real.opt)
            # amodal GT: hide everything but the robot
            saved = real.model.geom_group.copy(); real.model.geom_group[~real.is_robot[:-1]] = 5
            g_am = real.is_robot[fast_geom_segmentation(real.renderer, real.data, c, real.opt)]
            real.model.geom_group[:] = saved
            u = (g_am | mask).sum()
            ious[c].append(float((g_am & mask).sum() / u) if u else 1.0)
            if k < 4:
                img = real.image(c).copy()
                img[mask] = (0.45 * img[mask] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
                _, P, edges = skeleton_3d(tw, base_body="robot_mount", arm_joints=spec.arm_joints, ee_body=spec.ee_body,
                                          tcp_site="tcp", finger_joints=spec.gripper_joints or [])
                uv, ok = project(P, tw.camera, tw)
                img = draw_skeleton(img, uv, ok, edges, radius=3)
                cv2.putText(img, f"{c} IoU {ious[c][-1]:.3f}", (5, 15), 0, 0.45, (255, 255, 255), 1)
                row.append(img)
        if row:
            rows.append(np.concatenate(row, 1))
    cv2.imwrite(f"{out_dir}/overlay.jpg", np.concatenate(rows, 0)[..., ::-1])
    summary = dict(robot=spec.name, calibration=report, ik_pos_err_mm_median=float(np.median(ik_res) * 1000),
                   iou_amodal={c: dict(mean=float(np.mean(v)), min=float(np.min(v))) for c, v in ious.items()})
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
    print("  IoU (amodal, estimated calib):", {c: f"{np.mean(v):.3f} (min {np.min(v):.3f})" for c, v in ious.items()},
          f"| IK median err {np.median(ik_res) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
