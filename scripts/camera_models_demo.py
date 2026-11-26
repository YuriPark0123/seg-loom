"""Any-camera check: render the real world through random camera models (pinhole / Brown / fisheye,
random resolution + FOV + principal point) with image degradation, and draw the twin through
(a) the true camera model and (b) a pinhole model that ignores the lens distortion.

  python scripts/camera_models_demo.py --robot panda
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mujoco
import numpy as np

from shadow.camera_model import CameraModel, ModelRenderer, degrade, random_camera
from shadow.kinematic_mask import RobotKinematics
from shadow.mj_world import Layout, World, twin_xml
from shadow.skeleton import draw_skeleton, skeleton_3d
from shadow.urdf_robot import RobotSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="panda")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    spec = RobotSpec.load(args.robot); L = Layout(spec)
    real = World(spec, L, res=(64, 64))
    m, d = real.model, real.data
    js = dict(zip(spec.arm_joints, np.array(spec.rest_q) + rng.normal(0, 0.15, len(spec.arm_joints))))
    real.set_joints(js)
    cid = m.camera("front").id
    # move the front camera closer so wide lenses still show the robot large
    m.cam_pos[cid] = L.front_target + (m.cam_pos[cid] - L.front_target) * 0.75
    mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)

    # twin: robot-only model, its camera placed at the (true) camera pose -> isolates the lens model
    kin = RobotKinematics(xml=twin_xml(spec, L), base_body="robot_mount")
    kin.set_joints(kin.joints_from_mj(m, d))
    tm, td = kin.model, kin.data
    tcid = tm.camera("maskcam").id
    tm.cam_pos[tcid] = d.cam_xpos[cid]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, d.cam_xmat[cid]); tm.cam_quat[tcid] = q
    mujoco.mj_kinematics(tm, td); mujoco.mj_camlight(tm, td)
    topt = mujoco.MjvOption(); topt.geomgroup[:] = 0; topt.geomgroup[1] = 1; topt.sitegroup[:] = 0
    _, P, edges = skeleton_3d(kin, base_body="robot_mount", arm_joints=spec.arm_joints, ee_body=spec.ee_body,
                              tcp_site="tcp", finger_joints=spec.gripper_joints or [])
    T_wc = np.eye(4); T_wc[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ CV_TO_MJ; T_wc[:3, 3] = d.cam_xpos[cid]
    Pc = (P - T_wc[:3, 3]) @ T_wc[:3, :3]

    tiles, rows = [], []
    for k in range(args.n):
        kinds = ["pinhole", "brown", "brown", "fisheye", "fisheye", "brown"]
        cam = random_camera(rng, kinds=(kinds[k % len(kinds)],))
        disp = 320 / cam.width
        rr = ModelRenderer(m, cam); tr = ModelRenderer(tm, cam)
        img = rr.rgb(d, cid, real.opt)
        gid = rr.geom_ids(d, cid, real.opt)
        saved = m.geom_group.copy(); m.geom_group[~real.is_robot[:-1]] = 5
        g_am = real.is_robot[rr.geom_ids(d, cid, real.opt)]
        m.geom_group[:] = saved
        tw = tr.geom_ids(td, tcid, topt) >= 0                     # twin through the TRUE camera model
        pin = CameraModel(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, "pinhole")
        tw_pin = ModelRenderer(tm, pin).geom_ids(td, tcid, topt) >= 0   # twin assuming no distortion
        i_true, i_pin = iou(g_am, tw), iou(g_am, tw_pin)
        deg = degrade(img, rng)
        vis = deg.copy()
        vis[tw] = (0.45 * vis[tw] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
        cnt, _ = cv2.findContours(tw_pin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnt, -1, (255, 60, 200), max(1, int(2 / disp)), cv2.LINE_AA)
        uv, ok = cam.project(Pc)
        vis = draw_skeleton(vis, uv, ok, edges, radius=max(3, int(4 / disp)), thick=max(2, int(2 / disp)))
        vis = cv2.resize(vis, (320, int(cam.height * disp)))
        canvas = np.zeros((240, 320, 3), np.uint8); h = min(240, vis.shape[0]); canvas[:h] = vis[:h]
        cv2.putText(canvas, f"{cam.kind} {cam.width}x{cam.height}", (4, 14), 0, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"IoU true {i_true:.3f} | pinhole {i_pin:.3f}", (4, 232), 0, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
        rows.append((cam.describe(), i_true, i_pin))
        print(f"{cam.describe():70s} IoU(true model) {i_true:.4f}  IoU(pinhole, distortion ignored) {i_pin:.4f}")
    grid = np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles) - len(tiles) % 3, 3)], 0)
    os.makedirs(f"{ROOT}/out/any_camera", exist_ok=True)
    cv2.imwrite(f"{ROOT}/out/any_camera/{spec.name}.jpg", grid[..., ::-1])


CV_TO_MJ = np.diag([1.0, -1.0, -1.0])

if __name__ == "__main__":
    main()
