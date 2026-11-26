"""Robot skeleton (joints = points, links = lines) from forward kinematics, projected with the
estimated camera calibration."""
from __future__ import annotations

import cv2
import mujoco
import numpy as np

from .kinematic_mask import Camera, RobotKinematics, inv_T


def skeleton_3d(kin: RobotKinematics, dedupe_m=0.01, *, base_body="robot0_base", arm_joints=None,
                ee_body="robot0_right_hand", tcp_site="gripper0_grip_site", finger_joints=None,
                gripper_prefix="gripper0_"):
    """World-frame keypoints + edges: base -> arm joint axes -> ee body -> TCP -> fingertips.
    Defaults match robosuite; URDF robots pass names from their RobotSpec."""
    m, d = kin.model, kin.data
    names, pts = [], []

    def add(n, p):
        if pts and np.linalg.norm(pts[-1] - p) < dedupe_m:
            return
        names.append(n); pts.append(np.array(p))

    add("base", d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body)])
    if arm_joints is None:
        arm_joints = [m.joint(j).name for j in range(m.njnt)
                      if m.joint(j).name.startswith("robot0_") and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    for n in arm_joints:
        add(n.replace("robot0_", ""), d.xanchor[m.joint(n).id])
    add("ee", d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ee_body)])
    add("tcp", d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tcp_site)])
    chain = len(pts)
    edges = [(i, i + 1) for i in range(chain - 1)]
    if finger_joints is not None:          # fingertip ~ farthest point of each finger body's geoms
        tips = []
        for jn in finger_joints:
            b = m.jnt_bodyid[m.joint(jn).id]
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            far = max((d.geom_xpos[g] for g in gs), key=lambda p: np.linalg.norm(p - pts[chain - 2]), default=d.xpos[b])
            tips.append((m.body(b).name, far.copy()))
    else:
        gb = [b for b in range(m.nbody) if m.body(b).name.startswith(gripper_prefix)]
        parents = {m.body_parentid[b] for b in gb}
        tips = [(m.body(b).name.replace(gripper_prefix, ""), d.xpos[b].copy()) for b in gb if b not in parents]
    for n, p in tips:
        names.append(n); pts.append(p)
        edges.append((chain - 1, len(pts) - 1))
    return names, np.array(pts), edges


def project(points_w, cam: Camera, kin: RobotKinematics):
    T_cw = inv_T(kin.T_world_cam(cam))
    pc = points_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = pc[:, 2]
    uv = (pc @ cam.K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
    return uv, z > 1e-3


_EDGE_COLORS = [(255, 80, 80), (255, 160, 60), (240, 220, 60), (120, 220, 80), (60, 200, 200),
                (70, 130, 255), (170, 90, 255), (255, 90, 200), (220, 220, 220), (255, 255, 255)]


def draw_skeleton(img, uv, valid, edges, scale=1.0, color=None, radius=4, thick=2):
    out = img.copy()
    s = lambda p: (int(round((p[0] + 0.5) * scale - 0.5)), int(round((p[1] + 0.5) * scale - 0.5)))
    for k, (a, b) in enumerate(edges):
        if valid[a] and valid[b]:
            cv2.line(out, s(uv[a]), s(uv[b]), color or _EDGE_COLORS[k % len(_EDGE_COLORS)], thick, cv2.LINE_AA)
    for i in range(len(uv)):
        if valid[i]:
            cv2.circle(out, s(uv[i]), radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, s(uv[i]), radius, (0, 0, 0), 1, cv2.LINE_AA)
    return out
