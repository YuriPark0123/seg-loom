"""Run the checkerboard calibration in the 'real world' sim and save the ESTIMATED cameras."""
import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import robosuite as suite

from shadow.calibration import RealWorldCalibScene, calibrate, pose_error
from shadow.kinematic_mask import Camera, RobotKinematics, gt_camera_from_sim

CALIB_MODE = {"birdview": "static", "agentview": "static", "sideview": "static", "robot0_eye_in_hand": "wrist"}


def save_camera(path, cam: Camera):
    np.savez(path, K=cam.K, T=cam.T, height=cam.height, width=cam.width, dist=cam.dist,
             parent_body="" if cam.parent_body is None else cam.parent_body)


def load_camera(path) -> Camera:
    z = np.load(path)
    pb = str(z["parent_body"])
    return Camera(z["K"], z["T"], int(z["height"]), int(z["width"]), pb or None, z["dist"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Lift")
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--cams", nargs="+", default=["birdview", "agentview", "robot0_eye_in_hand"])
    ap.add_argument("--calib_res", type=int, default=512)
    ap.add_argument("--out", default="out/calib")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = suite.make(args.task, robots=args.robot, has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False)
    env.reset()
    base_pos = env.robots[0].robot_model.base_xpos_offset["table"](env.table_full_size[0])
    kin = RobotKinematics(args.robot, base_pos=base_pos)
    q0 = np.array(env.robots[0].init_qpos)
    spread = np.array([0.5, 0.35, 0.5, 0.35, 0.7, 0.35, 0.9])
    report = {}
    for c in args.cams:
        mode = CALIB_MODE[c]
        scene = RealWorldCalibScene(env.model.get_xml(), "hand" if mode == "static" else "table",
                                    args.calib_res, args.calib_res)
        cam, info = calibrate(scene, kin, c, mode, center=q0, spread=spread)
        gt = gt_camera_from_sim(env.sim, c, args.calib_res, args.calib_res)
        et, er = pose_error(cam.T, gt.T)
        report[c] = dict(trans_err_mm=et, rot_err_deg=er, reproj_rms_px=info["reproj_rms_px_joint"])
        print(f"[{c}] {et:.2f} mm, {er:.3f} deg")
        save_camera(f"{args.out}/{c}.npz", cam)
        del scene
    json.dump(dict(task=args.task, robot=args.robot, base_pos=list(map(float, base_pos)), report=report),
              open(f"{args.out}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
