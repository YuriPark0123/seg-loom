"""Generic MuJoCo 'real world' + kinematic twin for any robot described by URDF + RobotSpec.

Scene layout scales with spec.reach: table, cube, a top-down camera, a front camera, and a wrist
camera mounted on the end-effector body. Robot base frame = body 'robot_mount'.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .calibration import checkerboard_body
from .kinematic_mask import apply_joint_couplings
from .urdf_robot import RobotSpec, urdf_to_mjcf

CAMS = ["top", "front", "wrist"]
ROBOT_CONTYPE, ROBOT_CONAFF = 2, 1          # robot never collides with itself, does with the scene
SCENE_CONTYPE, SCENE_CONAFF = 1, 3


def _s(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def look_at_quat(pos, target, up_hint):
    """MuJoCo camera quat (camera looks along -z, y up) looking from pos at target."""
    f = np.asarray(target, float) - pos; f /= np.linalg.norm(f)
    right = np.cross(f, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(f, [1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.stack([right, up, -f], 1).flatten())
    return q


def _quat_z_to(v):
    """Quat rotating +z onto unit vector v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0, 0, 1.0])
    c = z @ v
    if c < -0.999999:
        return np.array([0, 1.0, 0, 0])
    axis = np.cross(z, v); s = np.sqrt((1 + c) * 2)
    return np.r_[s / 2, axis / s]


class Layout:
    """All scene dimensions, derived from the robot's reach."""

    def __init__(self, spec: RobotSpec, table_h=0.75):
        R = spec.reach
        self.R, self.table_h = R, table_h
        self.base_pos = np.array([spec.base_pos_rel[0] * R, spec.base_pos_rel[1] * R, table_h])
        self.table_center = np.array([0.5 * R, 0, table_h - 0.025])
        self.table_half = np.array([max(0.8 * R, 0.3), max(0.95 * R, 0.3), 0.025])
        self.cube_half = float(np.clip(0.045 * R, 0.012, 0.025))
        self.cube_pos = np.array([0.55 * R, 0.25 * R, table_h + self.cube_half])
        self.top_pos = np.array([0.5 * R, 0.0, table_h + 2.5 * R + 0.15])
        self.front_pos = np.array([1.75 * R + 0.15, 0.0, table_h + 1.15 * R + 0.1])
        self.front_target = np.array([0.35 * R, 0.0, table_h + 0.15 * R])
        self.ws_box = (np.array([0.15 * R, -0.6 * R, table_h + 0.12 * R]),
                       np.array([0.85 * R, 0.6 * R, table_h + 0.75 * R]))
        sq = float(np.clip(0.042 * R, 0.012, 0.03))
        self.board_hand = dict(cols=9, rows=6, square=round(sq, 4))
        self.board_table = dict(cols=9, rows=6, square=round(float(np.clip(0.03 * R, 0.01, 0.022)), 4))
        self.cam_box = {"top": (np.array([-0.5 * R, -1.2 * R, table_h + 0.9 * R]), np.array([1.5 * R, 1.2 * R, table_h + 4.5 * R])),
                        "front": (np.array([-0.5 * R, -3.0 * R, table_h + 0.05]), np.array([3.5 * R, 3.0 * R, table_h + 3.5 * R]))}


def _wrist_cam(spec: RobotSpec, R):
    """Wrist camera pose in the ee-body frame: behind/beside the TCP, looking past the fingertips."""
    wc = spec.wrist_cam or {}
    a = np.array(wc.get("approach_axis", [0, 0, 1.0])); s = np.array(wc.get("side_axis", [0, 1.0, 0]))
    tcp = np.array(spec.tcp_offset)
    pos = np.array(wc["pos"]) if "pos" in wc else tcp - a * 0.08 * R + s * 0.16 * R   # outside the wrist links
    look = np.array(wc["look_at"]) if "look_at" in wc else tcp + a * 0.35 * R
    return pos, look_at_quat(pos, look, s), float(wc.get("fovy", 75))


def _robot_tree(spec: RobotSpec):
    x = ET.fromstring(urdf_to_mjcf(spec))
    for g in x.iter("geom"):
        if g.get("group") == "0":
            g.set("contype", str(ROBOT_CONTYPE)); g.set("conaffinity", str(ROBOT_CONAFF))
    return x


def build_scene_xml(spec: RobotSpec, layout: Layout, board: str | None = None, robot_only=False,
                    offscreen=(640, 640)):
    """Full 'real world' MJCF (or a robot-only twin when robot_only=True)."""
    rob = _robot_tree(spec)
    root = ET.Element("mujoco", model=f"{spec.name}_world")
    root.append(rob.find("compiler"))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast" if hasattr(mujoco.mjtIntegrator, "mjINT_IMPLICITFAST") else "Euler")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(offscreen[0]), offheight=str(offscreen[1]))
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.55 0.55 0.55")
    ET.SubElement(vis, "map", znear="0.005")
    for tag in ("default", "asset", "equality", "contact"):
        el = rob.find(tag)
        if el is not None:
            root.append(el)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")
    mount = ET.SubElement(wb, "body", name="robot_mount", pos=_s(layout.base_pos))
    for child in list(rob.find("worldbody")):
        mount.append(child)
    R = layout.R
    wpos, wquat, wfovy = _wrist_cam(spec, R)
    ee = next(b for b in mount.iter("body") if b.get("name") == spec.ee_body)
    if robot_only:
        ET.SubElement(wb, "camera", name="maskcam", pos="0 0 1", quat="1 0 0 0", fovy="45")
        return ET.tostring(root, encoding="unicode")

    ee.append(ET.Element("camera", name="wrist", pos=_s(wpos), quat=_s(wquat), fovy=str(wfovy)))
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.75 0.8 0.88",
                  rgb2="0.3 0.33 0.38", width="256", height="256")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1="0.55 0.52 0.48",
                  rgb2="0.47 0.44 0.41", width="256", height="256")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8", reflectance="0.05")
    ET.SubElement(wb, "light", pos="0.5 -1 3", dir="-0.2 0.3 -1", directional="true", diffuse="0.6 0.6 0.6")
    ET.SubElement(wb, "light", pos="-1 1 2.5", dir="0.4 -0.3 -1", directional="true", diffuse="0.3 0.3 0.3")
    sc = dict(contype=str(SCENE_CONTYPE), conaffinity=str(SCENE_CONAFF))
    ET.SubElement(wb, "geom", name="floor", type="plane", size="5 5 0.1", material="floor_mat", group="1", **sc)
    ET.SubElement(wb, "geom", name="table", type="box", pos=_s(layout.table_center), size=_s(layout.table_half),
                  rgba="0.93 0.93 0.91 1", group="1", **sc)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lp = layout.table_center + np.array([sx * (layout.table_half[0] - 0.04), sy * (layout.table_half[1] - 0.04), 0])
            h = layout.table_h - 0.05
            ET.SubElement(wb, "geom", type="cylinder", pos=_s([lp[0], lp[1], h / 2]), size=_s([0.02, h / 2]),
                          rgba="0.35 0.35 0.37 1", group="1", contype="0", conaffinity="0")
    cube = ET.SubElement(wb, "body", name="cube", pos=_s(layout.cube_pos))
    ET.SubElement(cube, "freejoint", name="cube_free")
    ET.SubElement(cube, "geom", name="cube", type="box", size=_s([layout.cube_half] * 3), rgba="0.8 0.12 0.12 1",
                  friction="1.5 0.02 0.002", density="400", group="1", **sc)
    ET.SubElement(wb, "camera", name="top", pos=_s(layout.top_pos),
                  quat=_s(look_at_quat(layout.top_pos, layout.top_pos - [0, 0, 1], [-1, 0, 0])), fovy="45")
    ET.SubElement(wb, "camera", name="front", pos=_s(layout.front_pos),
                  quat=_s(look_at_quat(layout.front_pos, layout.front_target, [0, 0, 1])), fovy="45")
    if board == "hand":                         # board on a bracket beside the TCP, facing back along -approach
        wc = spec.wrist_cam or {}
        a = np.array(wc.get("approach_axis", [0, 0, 1.0])); s = np.array(wc.get("side_axis", [0, 1.0, 0]))
        b = layout.board_hand
        W = b["cols"] * b["square"] + 0.03
        H = b["rows"] * b["square"] + 0.03
        # stick the board out AWAY from the robot base (never hidden behind the arm), whatever the robot
        c = np.array(spec.tcp_offset) - a * 0.12 * R + board_out_dir(spec, layout, a) * (max(W, H) / 2 + 0.04 * R)
        ee.append(checkerboard_body("calib_board", c, _quat_z_to(-a), **b))
    elif board == "table":
        b = layout.board_table
        wb.append(checkerboard_body("calib_board", board_table_pos(spec, layout), (1, 0, 0, 0), **b))
    return ET.tostring(root, encoding="unicode")


def _rest_fk(spec, layout, q=None):
    m = mujoco.MjModel.from_xml_string(build_scene_xml(spec, layout, robot_only=True))
    d = mujoco.MjData(m)
    for j, v in zip(spec.arm_joints, spec.rest_q if q is None else q):
        d.qpos[m.joint(j).qposadr[0]] = v
    mujoco.mj_kinematics(m, d)
    return m, d


_WRIST_Q = {}


def wrist_calib_pose(spec, layout):
    """Arm pose for the wrist-camera calibration: the rest pose with the TCP lifted well above the
    table (so the board under the camera fits in the image and is not hidden by the fingers)."""
    key = (spec.name, layout.R)
    if key not in _WRIST_Q:
        from .ik import IKSolver
        from .kinematic_mask import RobotKinematics
        kin = RobotKinematics(xml=build_scene_xml(spec, layout, robot_only=True), base_body="robot_mount")
        ik = IKSolver(kin, site="tcp", joint_names=spec.arm_joints, rest_q=spec.rest_q, rot_weight=spec.rot_weight)
        kin.set_joints(dict(zip(spec.arm_joints, spec.rest_q)))
        p, R = ik.site_pose()
        p[2] = max(p[2], layout.table_h + max(0.3 * layout.R, 0.12))
        q, _, _ = ik.solve(spec.rest_q, p, R, iters=300)
        _WRIST_Q[key] = q
    return _WRIST_Q[key]


def board_out_dir(spec, layout, a):
    """Unit vector in the ee-body frame, perpendicular to the approach axis, pointing away from the
    robot base (horizontally) at the rest pose."""
    m, d = _rest_fk(spec, layout)
    b = m.body(spec.ee_body).id
    v = d.site_xpos[m.site("tcp").id] - d.xpos[m.body("robot_mount").id]
    v[2] = 0
    v = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-6 else np.array([1.0, 0, 0])
    v_ee = d.xmat[b].reshape(3, 3).T @ v
    v_ee -= (v_ee @ a) * a
    if np.linalg.norm(v_ee) < 1e-6:
        v_ee = np.cross(a, [0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    return v_ee / np.linalg.norm(v_ee)


def board_table_pos(spec, layout):
    """Table board where the wrist camera's optical axis hits the table at the rest pose
    (falls back to under the TCP), so the fingers do not hide it."""
    m, d = _rest_fk(spec, layout, wrist_calib_pose(spec, layout))
    b = m.body(spec.ee_body).id
    Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
    cpos, cquat, _ = _wrist_cam(spec, layout.R)
    Rc = np.zeros(9); mujoco.mju_quat2Mat(Rc, cquat)
    o, f = pb + Rb @ cpos, Rb @ (-Rc.reshape(3, 3)[:, 2])
    p = d.site_xpos[m.site("tcp").id].copy()
    if f[2] < -1e-3:
        p = o + f * (layout.table_h - o[2]) / f[2]
    lo = layout.table_center - layout.table_half + 0.1
    hi = layout.table_center + layout.table_half - 0.1
    return np.array([np.clip(p[0], lo[0], hi[0]), np.clip(p[1], lo[1], hi[1]), layout.table_h + 0.002])


class World:
    """A MuJoCo world with name-based joint access, camera images and robot segmentation."""

    def __init__(self, spec: RobotSpec, layout: Layout | None = None, board=None, res=(256, 256)):
        self.spec, self.layout = spec, layout or Layout(spec)
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(spec, self.layout, board=board))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jadr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
        self.dadr = {m.joint(j).name: m.jnt_dofadr[j] for j in range(m.njnt)}
        mount = m.body("robot_mount").id
        self.is_robot = np.array([m.body_rootid[m.geom_bodyid[g]] == mount for g in range(m.ngeom)] + [False])
        free_roots = {m.body_rootid[m.jnt_bodyid[j]] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        self.is_obj = np.array([m.body_rootid[m.geom_bodyid[g]] in free_roots for g in range(m.ngeom)] + [False])
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[0] = 0; self.opt.sitegroup[:] = 0
        self.h, self.w = res
        self.renderer = mujoco.Renderer(m, *res)
        self.board_thick = 0.003
        self.board = (self.layout.board_hand if board == "hand" else self.layout.board_table) if board else None
        self.set_joints(dict(zip(spec.arm_joints, spec.rest_q)))

    # -- state
    def set_joints(self, js: dict):
        for n, v in js.items():
            if n in self.jadr:
                self.data.qpos[self.jadr[n]] = v
        apply_joint_couplings(self.model, self.data, set(self.spec.arm_joints) | set(self.spec.gripper_joints or []))
        mujoco.mj_forward(self.model, self.data)

    def encoders(self, names=None):
        names = names if names is not None else [n for n in self.jadr if n in self.spec.arm_joints + self.spec.gripper_joints]
        return {n: float(self.data.qpos[self.jadr[n]]) for n in names if n in self.jadr}

    def image(self, cam):
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        return self.renderer.render()

    def T_base_cam_gt(self, cam):
        """Ground-truth camera pose in the robot base frame (OpenCV convention). Scoring only."""
        from .kinematic_mask import CV_TO_MJ, inv_T, make_T
        m, d = self.model, self.data
        b = m.body("robot_mount").id
        cid = m.camera(cam).id
        T_wb = make_T(d.xmat[b].reshape(3, 3), d.xpos[b])
        T_wc = make_T(d.cam_xmat[cid].reshape(3, 3) @ CV_TO_MJ, d.cam_xpos[cid])
        if m.cam_bodyid[cid] == 0:
            return inv_T(T_wb) @ T_wc
        pb = m.cam_bodyid[cid]
        return inv_T(make_T(d.xmat[pb].reshape(3, 3), d.xpos[pb])) @ T_wc   # T_parent_cam


def twin_xml(spec: RobotSpec, layout: Layout | None = None):
    layout = layout or Layout(spec)
    return build_scene_xml(spec, layout, robot_only=True)
