"""Dataset generation in the MuJoCo 'real world' (the part that needs an OpenGL renderer -> RTX box).

Per robot   : random attachments (cables, boxes - not in any URDF), wrist-camera mount error,
              self-contained export (true.urdf, approx.urdf with known errors, meshes, render geoms).
Per episode : random scene (table / floor / light / blocks incl. look-alikes), random camera models
              (pinhole / Brown / fisheye, any FOV) and poses, camera motion (static / drift / shake /
              jump), actuator model (lag, delay, backlash, encoder noise), then either
              phase 1 'motion'  : random EEF waypoints + joint jitter + gripper toggles (no task)
              phase 1 'ref'     : independent random poses (renderer-validation reference)
              phase 2 'task'    : scripted keyframe pick / stack with a language instruction
Per frame   : RGB, visible labels (robot link / object / table), amodal robot labels, amodal object
              labels, all poses (bodies, joints, TCP, objects, cameras), commands and encoders.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import clutter as CL
from .camera_model import CV_TO_MJ, CameraModel, ModelRenderer, random_camera
from .ik import IKSolver
from .kinematic_mask import RobotKinematics, apply_joint_couplings
from .mj_world import Layout, _s, build_scene_xml, look_at_quat, twin_xml
from .skeleton import skeleton_3d
from .urdf_robot import RobotSpec

CAMS = ["top", "front", "wrist"]
FPS = 10                      # recorded frames / actions per second
SUB = 5                       # actuator sub-steps per frame (50 Hz)
OBJ_COLORS = {
    "white": (0.93, 0.93, 0.92), "red": (0.8, 0.1, 0.1), "blue": (0.1, 0.25, 0.8), "green": (0.1, 0.6, 0.2),
    "yellow": (0.95, 0.82, 0.1), "black": (0.07, 0.07, 0.08), "orange": (0.95, 0.5, 0.1),
    "purple": (0.5, 0.2, 0.65), "gray": (0.5, 0.5, 0.52), "pink": (0.95, 0.55, 0.7), "brown": (0.45, 0.28, 0.12),
}
LABEL_OBJ0, LABEL_TABLE, LABEL_LEG, LABEL_FLOOR = 100, 200, 201, 202


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# =============================================================================== per-robot extras
def sample_robot_extras(spec: RobotSpec, layout: Layout, rng: np.random.Generator, recolor=0.0):
    """Things the URDF does not know: cables / boxes on links, wrist-camera mount error, and (with
    probability recolor; 0 = rng untouched) a repaint of the robot: per body a tint or a new colour."""
    m = mujoco.MjModel.from_xml_string(twin_xml(spec, layout))
    bodies = [m.body(b).name for b in range(m.nbody) if m.body_rootid[b] == m.body("robot_mount").id
              and m.body(b).name not in ("robot_mount",)]
    arm_bodies = [m.body(m.jnt_bodyid[m.joint(j).id]).name for j in spec.arm_joints]
    R = layout.R
    att = []
    for _ in range(int(rng.integers(0, 4))):
        b = str(rng.choice(arm_bodies))
        if rng.random() < 0.6:                       # cable: thin capsule running beside the link
            r = float(rng.uniform(0.003, 0.007))
            L = float(rng.uniform(0.08, 0.3) * R)
            side = _unit(rng.normal(size=2))
            off = float(rng.uniform(0.03, 0.06) * R)
            p0 = np.r_[side * off, rng.uniform(0, 0.1) * R]
            att.append(dict(body=b, type="capsule", size=[r, L / 2], pos=(p0 + [0, 0, L / 2]).tolist(),
                            quat=[1, 0, 0, 0], rgba=[0.05, 0.05, 0.05, 1], kind="cable"))
        else:                                        # box: controller / camera / tool mount
            sz = rng.uniform(0.012, 0.035, 3) * max(R, 0.4)
            p = np.r_[_unit(rng.normal(size=2)) * rng.uniform(0.03, 0.07) * R, rng.uniform(0, 0.15) * R]
            q = Rotation.from_euler("z", rng.uniform(0, np.pi)).as_quat()[[3, 0, 1, 2]]
            col = list(rng.uniform(0.05, 0.9, 3)) + [1]
            att.append(dict(body=b, type="box", size=sz.tolist(), pos=p.tolist(), quat=q.tolist(), rgba=col, kind="box"))
    wrist = dict(dpos=(rng.normal(0, 0.006, 3)).tolist(), drot=(rng.normal(0, np.deg2rad(2.5), 3)).tolist())
    out = dict(attachments=att, wrist_delta=wrist, bodies=bodies)
    if recolor > 0 and rng.random() < recolor:
        base = rng.uniform(0.05, 0.95, 3); acc = rng.uniform(0.05, 0.95, 3)
        mode = str(rng.choice(["tint", "two_tone", "random"]))
        rc = {}
        for b in bodies:
            if mode == "tint":
                rc[b] = dict(mul=np.clip(rng.normal(1, 0.25) * base / base.mean(), 0.2, 1.6).tolist())
            elif mode == "two_tone":
                rc[b] = dict(rgb=(base if rng.random() < 0.7 else acc).tolist())
            else:
                rc[b] = dict(rgb=rng.uniform(0.05, 0.95, 3).tolist())
        out["recolor"] = dict(mode=mode, bodies=rc)
    return out


# =============================================================================== scene
def sample_scene(spec: RobotSpec, layout: Layout, rng: np.random.Generator, n_obj=None):
    R, th = layout.R, layout.table_h
    table = list(rng.uniform(0.35, 0.95) * np.ones(3) * rng.uniform(0.9, 1.1, 3))
    if rng.random() < 0.3:
        table = list(OBJ_COLORS["white"])                       # white table: low contrast with white blocks
    floor = [list(rng.uniform(0.2, 0.8, 3)), None]
    floor[1] = list(np.clip(np.array(floor[0]) * rng.uniform(0.7, 0.95), 0, 1))
    lights = []
    for _ in range(int(rng.integers(1, 4))):
        d = _unit([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1])
        lights.append(dict(dir=d.tolist(), diffuse=float(rng.uniform(0.2, 0.7))))
    n = int(rng.integers(2, 6)) if n_obj is None else n_obj
    names = list(OBJ_COLORS)
    colors = list(rng.choice(names, n, replace=True))
    if rng.random() < 0.5 and "white" not in colors:
        colors[0] = "white"
    if rng.random() < 0.3 and n >= 2:                           # look-alike pair
        colors[1] = colors[0]
    base = layout.cube_half
    objs, placed = [], []
    for k in range(n):
        half = np.r_[rng.uniform(0.75, 1.25, 2) * base, rng.uniform(0.7, 1.6) * base]
        for _ in range(100):
            r = rng.uniform(0.3, 0.72) * R
            a = rng.uniform(-1.2, 1.2)
            p = layout.base_pos[:2] + r * np.array([np.cos(a), np.sin(a)])
            if p[0] < layout.base_pos[0] + 0.12 * R:
                continue
            if all(np.linalg.norm(p - q) > (np.linalg.norm(half[:2]) + hq) * 1.3 for q, hq in placed):
                break
        placed.append((p, np.linalg.norm(half[:2])))
        objs.append(dict(name=f"obj{k}", color=colors[k], rgba=list(OBJ_COLORS[colors[k]]) + [1],
                         half=half.tolist(), pos=[float(p[0]), float(p[1]), th + float(half[2])],
                         yaw=float(rng.uniform(-np.pi, np.pi))))
    return dict(table_rgba=table + [1], floor=floor, lights=lights, ambient=float(rng.uniform(0.15, 0.45)),
                objects=objs)


def build_world_xml(spec: RobotSpec, layout: Layout, scene: dict, extras: dict, offscreen=1600, assets_out=None):
    """assets_out: dict filled with in-memory texture files when the scene has clutter."""
    x = ET.fromstring(build_scene_xml(spec, layout, offscreen=(offscreen, offscreen)))
    wb = x.find("worldbody")
    for b in list(wb):
        if b.tag == "body" and b.get("name") == "cube":
            wb.remove(b)
        if b.tag == "light":
            wb.remove(b)
    for g in wb.iter("geom"):
        if g.get("name") == "table":
            g.set("rgba", _s(scene["table_rgba"]))
    vis = x.find("visual")
    hl = vis.find("headlight")
    hl.set("ambient", _s([scene["ambient"]] * 3)); hl.set("diffuse", _s([0.25] * 3))
    for t in x.find("asset").iter("texture"):
        if t.get("name") == "floor_tex":
            t.set("rgb1", _s(scene["floor"][0])); t.set("rgb2", _s(scene["floor"][1]))
    for L in scene["lights"]:
        ET.SubElement(wb, "light", pos="0 0 3", dir=_s(L["dir"]), directional="true", diffuse=_s([L["diffuse"]] * 3),
                      castshadow="true")
    for o in scene["objects"]:
        q = Rotation.from_euler("z", o["yaw"]).as_quat()[[3, 0, 1, 2]]
        b = ET.SubElement(wb, "body", name=o["name"], pos=_s(o["pos"]), quat=_s(q))
        ET.SubElement(b, "freejoint", name=f"{o['name']}_free")
        ET.SubElement(b, "geom", name=o["name"], type="box", size=_s(o["half"]), rgba=_s(o["rgba"]), group="1",
                      contype="1", conaffinity="3", density="400", friction="1.5 0.02 0.002")
    bodies = {b.get("name"): b for b in wb.iter("body")}
    for k, a in enumerate(extras["attachments"]):
        ET.SubElement(bodies[a["body"]], "geom", name=f"attach{k}", type=a["type"], size=_s(a["size"]),
                      pos=_s(a["pos"]), quat=_s(a["quat"]), rgba=_s(a["rgba"]), group="1", contype="0",
                      conaffinity="0", density="0")
    if extras.get("recolor"):                       # repaint robot visual geoms (attachments keep their colour)
        rc = extras["recolor"]["bodies"]
        for b in wb.iter("body"):
            if b.get("name") in rc:
                for g in b.findall("geom"):
                    if g.get("group") != "1" or (g.get("name") or "").startswith("attach"):
                        continue
                    old = np.array([float(v) for v in (g.get("rgba") or "0.75 0.75 0.78 1").split()])
                    new = np.clip(old[:3] * rc[b.get("name")]["mul"], 0, 1) if "mul" in rc[b.get("name")] else np.array(rc[b.get("name")]["rgb"])
                    g.set("rgba", _s(np.r_[new, old[3]]))
                    if g.get("material"):
                        g.attrib.pop("material")
    for c in wb.iter("camera"):
        if c.get("name") == "wrist":
            wd = extras["wrist_delta"]
            c.set("pos", _s(np.array([float(v) for v in c.get("pos").split()]) + wd["dpos"]))
            q0 = np.array([float(v) for v in c.get("quat").split()])
            r = Rotation.from_quat(q0[[1, 2, 3, 0]]) * Rotation.from_rotvec(wd["drot"])
            c.set("quat", _s(r.as_quat()[[3, 0, 1, 2]]))
    if scene.get("clutter"):
        a = CL.apply_xml(x, layout, scene["clutter"], scene)
        if assets_out is not None:
            assets_out.update(a)
    return ET.tostring(x, encoding="unicode")


# =============================================================================== camera poses
@dataclass
class CamPose:
    pos: np.ndarray
    target: np.ndarray
    up: np.ndarray

    def quat(self):
        return look_at_quat(self.pos, self.target, self.up)

    def lerp(self, o: "CamPose", s):
        return CamPose(self.pos + (o.pos - self.pos) * s, self.target + (o.target - self.target) * s,
                       _unit(self.up + (o.up - self.up) * s))


def hfov_deg(cam: CameraModel):
    r = cam.unproject(np.array([[0, cam.cy], [cam.width - 1, cam.cy]]))
    return float(np.rad2deg(np.arccos(np.clip(r[0] @ r[1], -1, 1))))


def fov_distance_factor(cam: CameraModel):
    """Distance multiplier so the robot fills a similar part of the image whatever the lens."""
    f = hfov_deg(cam)
    f = 120.0 if not np.isfinite(f) or f < 1 else f
    h = np.deg2rad(min(f, 150) / 2)
    return float(np.clip(np.tan(np.deg2rad(30)) / np.tan(h), 0.3, 1.4))


def sample_cam_pose(kind, layout: Layout, rng, near: CamPose | None = None, scale=1.0, dist_k=1.0):
    R, th = layout.R, layout.table_h
    c = np.array([0.4 * R, 0.0, th + 0.12 * R])
    if near is not None:                                      # a nearby pose (drift / jump)
        p = near.pos + rng.normal(0, 0.15 * R * scale, 3)
        p[2] = max(p[2], th + 0.25 * R)
        t = near.target + rng.normal(0, 0.06 * R * scale, 3)
        up = _unit(near.up + rng.normal(0, 0.08 * scale, 3))
        return CamPose(p, t, up)
    if kind == "top":
        h = (rng.uniform(1.4, 2.4) * R + 0.1) * dist_k + 0.25 * R * (1 - dist_k)
        p = c + np.r_[rng.normal(0, 0.12 * R), rng.normal(0, 0.2 * R), h]
        t = p - np.r_[0, 0, h] + np.r_[rng.normal(0, 0.12 * h, 2), 0]
        a = rng.uniform(-np.pi, np.pi)
        return CamPose(p, t, np.r_[np.cos(a), np.sin(a), 0])
    az = np.deg2rad(rng.uniform(-75, 75)); el = np.deg2rad(rng.uniform(12, 60)); dist = max((rng.uniform(1.1, 2.1) * R + 0.15) * dist_k, 0.9 * R + 0.1)
    p = c + dist * np.r_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    t = c + rng.normal(0, 0.08 * R, 3)
    f = _unit(t - p); right = _unit(np.cross(f, [0, 0, 1])); up = np.cross(right, f)
    roll = np.deg2rad(rng.normal(0, 4))
    return CamPose(p, t, _unit(np.cos(roll) * up + np.sin(roll) * right))


def camera_track(kind, layout, rng, T, mode, dist_k=1.0):
    """Per-frame CamPose list for a world camera."""
    p0 = sample_cam_pose(kind, layout, rng, dist_k=dist_k)
    if mode == "static":
        return [p0] * T
    if mode == "drift":
        p1 = sample_cam_pose(kind, layout, rng, near=p0, scale=1.5)
        return [p0.lerp(p1, 0.5 - 0.5 * np.cos(np.pi * k / max(T - 1, 1))) for k in range(T)]
    if mode == "shake":
        p1 = sample_cam_pose(kind, layout, rng, near=p0, scale=0.5)
        out = []
        for k in range(T):
            b = p0.lerp(p1, k / max(T - 1, 1))
            out.append(CamPose(b.pos + rng.normal(0, 0.004, 3), b.target + rng.normal(0, 0.006, 3), b.up))
        return out
    if mode == "jump":
        p1 = sample_cam_pose(kind, layout, rng, near=p0, scale=2.0)
        k0 = int(rng.integers(T // 4, 3 * T // 4))
        return [p0 if k < k0 else p1 for k in range(T)]
    raise ValueError(mode)


# =============================================================================== world wrapper
class DataWorld:
    def __init__(self, spec: RobotSpec, layout: Layout, scene: dict, extras: dict):
        self.spec, self.layout, self.scene, self.extras = spec, layout, scene, extras
        assets = {}
        xml = build_world_xml(spec, layout, scene, extras, assets_out=assets)
        self.model = m = mujoco.MjModel.from_xml_string(xml, assets) if assets else mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(m)
        mount = m.body("robot_mount").id
        self.robot_bodies = [b for b in range(m.nbody) if m.body_rootid[b] == mount and b != mount]
        self.body_label = {b: i + 1 for i, b in enumerate(self.robot_bodies)}
        self.obj_bodies = [m.body(o["name"]).id for o in scene["objects"]]
        self.obj_qadr = [m.jnt_qposadr[m.joint(f"{o['name']}_free").id] for o in scene["objects"]]
        lut = np.zeros(m.ngeom + 1, np.uint8)                     # index -1 -> background
        self.is_robot = np.zeros(m.ngeom, bool); self.is_obj = np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b in self.body_label:
                lut[g] = self.body_label[b]; self.is_robot[g] = True
            elif b in self.obj_bodies:
                lut[g] = LABEL_OBJ0 + self.obj_bodies.index(b); self.is_obj[g] = True
            elif CL.label_for(m.geom(g).name) is not None:
                lut[g] = CL.label_for(m.geom(g).name)
            else:
                nm = m.geom(g).name
                lut[g] = LABEL_TABLE if nm == "table" else (LABEL_FLOOR if nm == "floor" else (LABEL_LEG if b == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else 0))
        self.lut = lut
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[0] = 0; self.opt.sitegroup[:] = 0
        self.jadr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
        self.cam_ids = {c: m.camera(c).id for c in CAMS}
        self.base_id = mount
        self.tcp = m.site("tcp").id
        self.joint_ids = [j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == mount]
        self.renderers = {}
        self._driven = set(self.spec.arm_joints) | set(self.spec.gripper_joints or [])
        mujoco.mj_forward(m, self.data)

    # ---- state
    def set_q(self, q_arm, q_grip):
        d = self.data
        for n, v in zip(self.spec.arm_joints, q_arm):
            d.qpos[self.jadr[n]] = v
        for n, v in zip(self.spec.gripper_joints, q_grip):
            d.qpos[self.jadr[n]] = v
        apply_joint_couplings(self.model, d, self._driven)
        mujoco.mj_kinematics(self.model, d)

    def set_obj(self, k, pos, quat):
        a = self.obj_qadr[k]
        self.data.qpos[a:a + 3] = pos; self.data.qpos[a + 3:a + 7] = quat

    def set_world_cam(self, c, pose: CamPose):
        cid = self.cam_ids[c]
        self.model.cam_pos[cid] = pose.pos; self.model.cam_quat[cid] = pose.quat()

    def refresh(self):
        mujoco.mj_kinematics(self.model, self.data); mujoco.mj_camlight(self.model, self.data)

    # ---- frames
    def T_world_base(self):
        d = self.data
        T = np.eye(4); T[:3, :3] = d.xmat[self.base_id].reshape(3, 3); T[:3, 3] = d.xpos[self.base_id]
        return T

    def T_base_cam(self, c):
        """OpenCV camera pose in the robot base frame."""
        d, cid = self.data, self.cam_ids[c]
        T = np.eye(4); T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ CV_TO_MJ; T[:3, 3] = d.cam_xpos[cid]
        return np.linalg.inv(self.T_world_base()) @ T

    def to_base(self, P):
        Ti = np.linalg.inv(self.T_world_base())
        return P @ Ti[:3, :3].T + Ti[:3, 3]

    def renderer(self, c, cam: CameraModel):
        key = (c, cam.width, cam.height, cam.fx, cam.kind)
        if key not in self.renderers:
            self.renderers = {k: v for k, v in self.renderers.items() if k[0] != c}
            self.renderers[key] = ModelRenderer(self.model, cam)
        return self.renderers[key]

    def render(self, c, cam: CameraModel):
        """rgb, visible labels, amodal robot labels, amodal object labels."""
        m, d, cid = self.model, self.data, self.cam_ids[c]
        r = self.renderer(c, cam)
        rgb = r.rgb(d, cid, self.opt)
        vis = self.lut[r.geom_ids(d, cid, self.opt)]
        saved = m.geom_group.copy()
        try:
            m.geom_group[~self.is_robot] = 5
            am_r = self.lut[r.geom_ids(d, cid, self.opt)]
            m.geom_group[:] = saved
            m.geom_group[~self.is_obj] = 5
            am_o = self.lut[r.geom_ids(d, cid, self.opt)]
        finally:
            m.geom_group[:] = saved
        return rgb, vis, am_r, am_o


# =============================================================================== actuator model
class Actuator:
    """Command -> true joint motion: delay, first-order lag, backlash; encoder = true + noise."""

    def __init__(self, n, rng, ideal=False):
        self.delay = 0 if ideal else int(rng.integers(0, 3))                   # sub-steps (20 ms each)
        self.tau = 1e-6 if ideal else float(rng.uniform(0.02, 0.12))
        self.backlash = np.zeros(n) if ideal else rng.uniform(0, 0.008, n)
        self.noise = 0.0 if ideal else float(rng.uniform(0, 0.003))
        self.rng = rng
        self.buf, self.lag, self.q = [], None, None

    def reset(self, q):
        self.lag, self.q, self.buf = q.copy(), q.copy(), [q.copy()] * (self.delay + 1)

    def step(self, cmd, dt):
        self.buf.append(cmd.copy())
        c = self.buf.pop(0)
        a = 1 - np.exp(-dt / self.tau)
        self.lag = self.lag + a * (c - self.lag)
        e = self.lag - self.q
        b = self.backlash / 2
        self.q = np.where(e > b, self.lag - b, np.where(e < -b, self.lag + b, self.q))
        return self.q

    def encoder(self):
        return self.q + self.rng.normal(0, self.noise, len(self.q)) if self.noise else self.q.copy()


# =============================================================================== generator
def _is_under(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def gripper_width_table(spec: RobotSpec, layout: Layout, n=21, raw=False):
    """Fingertip distance (relative to closed) as a function of the open fraction.
    raw=True: absolute distances, unsigned/unclamped (to check that 'open' is the wide end)."""
    if not spec.gripper:
        return np.linspace(0, 1, n), np.zeros(n)
    kin = RobotKinematics(xml=twin_xml(spec, layout), base_body="robot_mount")
    kin.set_joints(dict(zip(spec.arm_joints, spec.rest_q)))
    o, c = np.array(spec.gripper["open"]), np.array(spec.gripper["close"])
    al, w = np.linspace(0, 1, n), []
    m = kin.model
    follow = {m.joint(m.eq_obj1id[e]).name for e in range(m.neq)
              if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_obj2id[e] >= 0}
    if follow:
        # coupled (mimic) fingers: the commanded joints alone do not give two fingertips; use the largest
        # distance between geoms moved by any gripper joint (leaders + followers)
        mov = {m.jnt_bodyid[m.joint(j).id] for j in list(spec.gripper_joints) + sorted(follow)}
        geoms = [g for g in range(m.ngeom) if any(_is_under(m, m.geom_bodyid[g], b) for b in mov)]
        for a in al:
            kin.set_joints(dict(zip(spec.gripper_joints, c + a * (o - c))))
            X = kin.data.geom_xpos[geoms]
            w.append(float(np.max(np.linalg.norm(X[:, None] - X[None], axis=-1))) if len(geoms) > 1 else 0.0)
        if raw:
            return al, np.array(w)
        w = np.array(w); w = w - w[0]
        return al, np.maximum.accumulate(np.abs(w))
    for a in al:
        kin.set_joints(dict(zip(spec.gripper_joints, c + a * (o - c))))
        _, P, _ = skeleton_3d(kin, base_body="robot_mount", arm_joints=spec.arm_joints, ee_body=spec.ee_body,
                              tcp_site="tcp", finger_joints=spec.gripper_joints)
        tips = P[-len(spec.gripper_joints):]
        w.append(np.linalg.norm(tips[0] - tips[-1]) if len(tips) > 1 else 0.0)
    if raw:
        return al, np.array(w)
    w = np.array(w); w = w - w[0]
    return al, np.maximum.accumulate(np.abs(w))


class EpisodeGen:
    def __init__(self, spec: RobotSpec, extras: dict, rng: np.random.Generator, out_w=320, clutter=0.0):
        self.spec, self.rng, self.out_w = spec, rng, out_w
        self.clutter = clutter                      # probability of a cluttered scene (0 = plain scenes, rng untouched)
        self.layout = Layout(spec)
        self.extras = extras
        self.kin = RobotKinematics(xml=twin_xml(spec, self.layout), base_body="robot_mount")
        self.ik = IKSolver(self.kin, site="tcp", joint_names=spec.arm_joints, rest_q=spec.rest_q,
                           rot_weight=spec.rot_weight)
        self.grip_al, self.grip_w = gripper_width_table(spec, self.layout)
        self.go, self.gc = (np.array(spec.gripper["open"]), np.array(spec.gripper["close"])) if spec.gripper else (np.zeros(0), np.zeros(0))
        self.n = len(spec.arm_joints)

    # ---------- helpers
    def grip_q(self, frac):
        return self.gc + frac * (self.go - self.gc)

    def frac_for_width(self, w):
        """Open fraction that holds an object of width w (always < 1 so 'close' is a real command)."""
        if self.grip_w[-1] > 5e-3:
            f = float(np.interp(w, self.grip_w, self.grip_al))
        else:                                       # one moving jaw / no gripper: no width model
            f = w / (0.12 * self.layout.R)
        return float(np.clip(f, 0.05, 0.9))

    def ik_to(self, q0, p, Rm):
        q, ep, er = self.ik.solve(q0, p, Rm, iters=120)
        return q, ep, er

    def down_R(self, yaw=0.0, tilt=0.0, tilt_dir=0.0):
        Rd = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
        r = Rotation.from_rotvec([0, 0, yaw]) * Rotation.from_rotvec(tilt * np.r_[np.cos(tilt_dir), np.sin(tilt_dir), 0])
        return r.as_matrix() @ Rd

    def sample_cams(self, T, world: DataWorld, cam_mode=None):
        rng = self.rng
        models, tracks, modes = {}, {}, {}
        for c in CAMS:
            cam = random_camera(rng)
            models[c] = cam.scaled(self.out_w / cam.width)
        for c in ("top", "front"):
            mode = cam_mode or str(rng.choice(["static", "static", "drift", "shake", "jump"]))
            for _ in range(30):                                   # robot must be in view at the start
                tr = camera_track(c, self.layout, rng, T, mode, dist_k=fov_distance_factor(models[c]))
                world.set_world_cam(c, tr[0]); world.refresh()
                if self._in_view(world, c, models[c]):
                    break
            tracks[c], modes[c] = tr, mode
        modes["wrist"] = "mounted"
        return models, tracks, modes

    def _in_view(self, world: DataWorld, c, cam: CameraModel):
        d = world.data
        P = np.array([d.xpos[world.base_id], d.site_xpos[world.tcp], [0.45 * self.layout.R, 0, self.layout.table_h]])
        T = np.linalg.inv(world.T_world_base() @ world.T_base_cam(c))
        uv, ok = cam.project(P @ T[:3, :3].T + T[:3, 3])
        inside = ok & (uv[:, 0] > 5) & (uv[:, 0] < cam.width - 5) & (uv[:, 1] > 5) & (uv[:, 1] < cam.height - 5)
        return inside.sum() >= 2 and inside[1]

    # ---------- command sequences (10 Hz arm/gripper commands)
    def motion_commands(self, T, world: DataWorld):
        """Phase-1: random EEF waypoints (mostly gripper-down) + joint jitter + gripper toggles."""
        rng, L = self.rng, self.layout
        lo, hi = L.ws_box
        top = max([o["pos"][2] + o["half"][2] for o in world.scene["objects"]] + [L.table_h])
        lo = lo.copy(); lo[2] = max(lo[2], top + 0.06 * L.R)
        q = np.array(self.spec.rest_q, float)
        cmds, grip, frac = [], [], 1.0
        while len(cmds) < T:
            if rng.random() < 0.75:
                p = rng.uniform(lo, hi)
                Rm = self.down_R(rng.uniform(-np.pi, np.pi), rng.uniform(0, 0.5), rng.uniform(-np.pi, np.pi))
                q1, ep, _ = self.ik_to(q, p, Rm)
                if ep > 0.03 * L.R:
                    continue
            else:
                q1 = np.clip(q + rng.normal(0, 0.35, self.n), self.ik.lo, self.ik.hi)
            dur = max(0.6, np.abs(q1 - q).max() / rng.uniform(0.5, 1.2))
            k = max(2, int(dur * FPS))
            for i in range(1, k + 1):
                s = i / k; s = s * s * (3 - 2 * s)
                cmds.append(q + (q1 - q) * s)
                if rng.random() < 0.04:
                    frac = 1.0 - frac if rng.random() < 0.7 else float(rng.uniform(0, 1))
                grip.append(frac)
            if rng.random() < 0.3:                                  # hold
                for _ in range(int(rng.integers(1, 6))):
                    cmds.append(q1.copy()); grip.append(frac)
            q = q1
        return np.array(cmds[:T]), np.array(grip[:T]), None

    def ref_commands(self, T):
        """Renderer reference: independent random configurations (within limits, TCP above table)."""
        rng, L = self.rng, self.layout
        lo, hi = L.ws_box
        out, grip = [], []
        q = np.array(self.spec.rest_q, float)
        while len(out) < T:
            if rng.random() < 0.6:
                q1, ep, _ = self.ik_to(q, rng.uniform(lo, hi), self.down_R(rng.uniform(-3, 3), rng.uniform(0, 0.8), rng.uniform(-3, 3)))
                if ep > 0.03 * L.R:
                    continue
            else:
                q1 = np.clip(np.array(self.spec.rest_q) + rng.normal(0, 0.6, self.n), self.ik.lo, self.ik.hi)
                self.kin.set_joints(dict(zip(self.spec.arm_joints, q1)))
                if (self.kin.data.xpos[2:, 2] < L.table_h + 0.01).any():
                    continue
            out.append(q1); grip.append(float(rng.uniform(0, 1)))
        return np.array(out), np.array(grip), None

    def task_commands(self, world: DataWorld):
        """Phase-2: scripted keyframes for 'pick up X' / 'put X on Y'. Returns commands + task info."""
        rng, L, R = self.rng, self.layout, self.layout.R
        objs = world.scene["objects"]
        Tb = np.linalg.inv(world.T_world_base())
        kind = "stack" if (len(objs) >= 2 and rng.random() < 0.6) else "pick"
        # choose a feasible source (and target): grasp / place poses reachable by IK
        q_rest = np.array(self.spec.rest_q, float)
        tol = max(0.012 * R, 0.006)

        def grasp_pose(o):
            hs_ = np.array(o["half"])
            gy = o["yaw"] if hs_[0] <= hs_[1] else o["yaw"] + np.pi / 2
            best = None
            for cand in (gy, gy + np.pi):                          # parallel jaws: both hand yaws hold it
                cand = (cand + np.pi) % (2 * np.pi) - np.pi
                _, ep, er = self.ik_to(q_rest, np.array(o["pos"]) + [0, 0, max(hs_[2] * 0.3, 0.0)], self.down_R(cand))
                if best is None or ep + 0.01 * er < best[1] + 0.01 * best[2]:
                    best = (cand, ep, er)
            return best

        order = list(rng.permutation(len(objs)))
        i = j = None
        for ci in order:
            gy, ep, _ = grasp_pose(objs[ci])
            if ep > tol:
                continue
            if kind == "stack":
                for cj in rng.permutation(len(objs)):
                    if cj == ci:
                        continue
                    pd_ = np.array(objs[cj]["pos"]); top_ = pd_[2] + objs[cj]["half"][2] + objs[ci]["half"][2] + 0.004
                    _, e2, _ = self.ik_to(q_rest, np.r_[pd_[:2], top_ + max(objs[ci]["half"][2] * 0.3, 0)], self.down_R(gy))
                    if e2 <= tol:
                        j = int(cj); break
                if j is None:
                    continue
            i = int(ci); g_yaw = gy
            break
        if i is None:                                              # nothing feasible: still record the attempt
            i = int(order[0]); g_yaw = grasp_pose(objs[i])[0]
            j = int(next(k for k in order if k != i)) if kind == "stack" else None
        src = objs[i]
        ps = np.array(src["pos"]); hs = np.array(src["half"])
        width = 2 * min(hs[0], hs[1])
        Rg = self.down_R(g_yaw)
        frac_open = 1.0
        frac_close = self.frac_for_width(width)
        keys = []                                                  # (name, tcp pos world, R, grip frac)
        lift = max(0.12 * R, 0.06)
        keys.append(("pregrasp", ps + [0, 0, hs[2] + lift], Rg, frac_open))
        keys.append(("grasp", ps + [0, 0, max(hs[2] * 0.3, 0.0)], Rg, frac_open))
        keys.append(("close", ps + [0, 0, max(hs[2] * 0.3, 0.0)], Rg, frac_close))
        keys.append(("lift", ps + [0, 0, hs[2] + lift], Rg, frac_close))
        if kind == "stack":
            dst = objs[j]; pd = np.array(dst["pos"]); hd = np.array(dst["half"])
            place_z = pd[2] + hd[2] + hs[2] + 0.004
            off = place_z - (ps[2])                                  # TCP height relative to object center at grasp
            tcp_off = max(hs[2] * 0.3, 0.0)
            Rp = self.down_R(g_yaw + float(rng.normal(0, 0.1)))
            keys.append(("preplace", np.r_[pd[:2], place_z + tcp_off + lift], Rp, frac_close))
            keys.append(("place", np.r_[pd[:2], place_z + tcp_off], Rp, frac_close))
            keys.append(("release", np.r_[pd[:2], place_z + tcp_off], Rp, frac_open))
            keys.append(("retreat", np.r_[pd[:2], place_z + tcp_off + lift], Rp, frac_open))
        else:
            keys.append(("hold", ps + [0, 0, hs[2] + 2 * lift], Rg, frac_close))
        # IK through the keyframes (cartesian interpolation)
        q = np.array(self.spec.rest_q, float)
        self.kin.set_joints(dict(zip(self.spec.arm_joints, q)))
        p_prev, R_prev = self.ik.site_pose()
        g_prev = frac_open
        cmds, grip, kf_index, kf_err = [], [], [], []
        for name, p, Rm, g in keys:
            dist = np.linalg.norm(p - p_prev)
            k = max(3, int(np.ceil(dist / (0.35 * R / FPS)))) if name not in ("close", "release") else 6
            r0, r1 = Rotation.from_matrix(R_prev), Rotation.from_matrix(Rm)
            for s in np.linspace(0, 1, k + 1)[1:]:
                ss = s * s * (3 - 2 * s)
                pi = p_prev + (p - p_prev) * ss
                Ri = Rotation.from_rotvec((r1 * r0.inv()).as_rotvec() * ss).as_matrix() @ R_prev
                q, ep, _ = self.ik_to(q, pi, Ri)
                cmds.append(q.copy()); grip.append(g_prev + (g - g_prev) * ss)
            _, ep, er = self.ik_to(q, p, Rm)
            kf_index.append(len(cmds) - 1); kf_err.append((float(ep), float(er)))
            p_prev, R_prev, g_prev = p, Rm, g
        for _ in range(4):                                         # settle
            cmds.append(q.copy()); grip.append(g_prev)
        kf = [dict(name=n, frame=int(f), tcp_world=np.asarray(p).tolist(), R_world=np.asarray(Rm).tolist(), grip=float(g),
                   tcp_base=(Tb[:3, :3] @ p + Tb[:3, 3]).tolist(), ik_err_m=e[0], ik_err_rad=e[1])
              for (n, p, Rm, g), f, e in zip(keys, kf_index, kf_err)]
        task = dict(kind=kind, src=i, dst=j, keyframes=kf, grasp_width=float(width),
                    instruction=self._instruction(objs, kind, i, j))
        return np.array(cmds), np.array(grip), task

    def _ref(self, objs, i):
        o = objs[i]
        same = [k for k, x in enumerate(objs) if x["color"] == o["color"]]
        if len(same) == 1:
            return f"the {o['color']} block"
        vol = [np.prod(objs[k]["half"]) for k in same]
        if max(vol) > 1.4 * min(vol):
            return f"the {'larger' if np.prod(o['half']) == max(vol) else 'smaller'} {o['color']} block" if len(same) == 2 \
                else f"the {o['color']} block at y={o['pos'][1]:+.2f}"
        ys = [objs[k]["pos"][1] for k in same]
        return f"the {o['color']} block on the {'left' if o['pos'][1] == max(ys) else 'right'}"

    def _instruction(self, objs, kind, i, j):
        if kind == "pick":
            return f"pick up {self._ref(objs, i)}"
        return f"put {self._ref(objs, i)} on {self._ref(objs, j)}"

    # ---------- run one episode
    def run(self, mode, T=60, ideal_actuator=False, cam_mode=None, scene=None):
        rng, spec, L = self.rng, self.spec, self.layout
        if scene is None:
            scene = sample_scene(spec, L, rng)
            if self.clutter > 0 and rng.random() < self.clutter:
                scene["clutter"] = CL.sample_clutter(L, scene, rng)
        world = DataWorld(spec, L, scene, self.extras)
        if mode == "motion":
            cmds, grip, task = self.motion_commands(T, world)
        elif mode == "ref":
            cmds, grip, task = self.ref_commands(T)
            ideal_actuator, cam_mode = True, "static"
        elif mode == "task":
            cmds, grip, task = self.task_commands(world)
        else:
            raise ValueError(mode)
        T = len(cmds)
        world.set_q(cmds[0], self.grip_q(grip[0])); world.refresh()
        models, tracks, cmodes = self.sample_cams(T, world, cam_mode)
        if scene.get("clutter"):
            CL.place_after_cams(world, scene["clutter"], tracks, rng)
            world.refresh()
            CL.limit_occlusion(world, scene["clutter"], lambda c: world.render(c, models[c]))
        act = Actuator(self.n + len(self.go), rng, ideal=ideal_actuator)
        x0 = np.r_[cmds[0], self.grip_q(grip[0])]
        act.reset(x0)
        attached, T_rel = None, None
        rec = {k: [] for k in ["q_cmd", "grip_cmd", "q_true", "q_enc", "tcp_pose", "tcp_cmd_pose", "body_pose",
                               "joint_anchor", "joint_axis", "obj_pose", "skel3d", "attached"]}
        rec.update({f"T_base_cam_{c}": [] for c in CAMS})
        rec.update({f"skel2d_{c}": [] for c in CAMS}); rec.update({f"obj2d_{c}": [] for c in CAMS})
        frames = {c: dict(rgb=[], vis=[], am_r=[], am_o=[]) for c in CAMS}
        for t in range(T):
            cmd = np.r_[cmds[t], self.grip_q(grip[t])]
            if mode == "ref":
                x = act.q = cmd.copy()
            else:
                for _ in range(SUB):
                    x = act.step(cmd, 1.0 / (FPS * SUB))
            world.set_q(x[:self.n], x[self.n:])
            # kinematic grasp: attach when the gripper is commanded closed around the object
            if task is not None:
                tcpw = world.data.site_xpos[world.tcp]; tcpR = world.data.site_xmat[world.tcp].reshape(3, 3)
                src = task["src"]
                if attached is None and grip[t] < 0.999 and t > 0 and grip[t] < grip[t - 1] - 1e-6:
                    op = world.data.xpos[world.obj_bodies[src]]
                    hs = np.array(scene["objects"][src]["half"])
                    if np.linalg.norm((tcpw - op)[:2]) < max(0.7 * hs[:2].min(), 0.01) and abs(tcpw[2] - op[2]) < hs[2] + 0.01:
                        attached = src
                        Tw = np.eye(4); Tw[:3, :3] = tcpR; Tw[:3, 3] = tcpw
                        To = np.eye(4); To[:3, :3] = world.data.xmat[world.obj_bodies[src]].reshape(3, 3); To[:3, 3] = op
                        T_rel = np.linalg.inv(Tw) @ To
                elif attached is not None and grip[t] > grip[t - 1] + 1e-6:
                    attached = None
                if attached is not None:
                    Tw = np.eye(4); Tw[:3, :3] = tcpR; Tw[:3, 3] = tcpw
                    To = Tw @ T_rel
                    world.set_obj(attached, To[:3, 3], Rotation.from_matrix(To[:3, :3]).as_quat()[[3, 0, 1, 2]])
            for c in ("top", "front"):
                world.set_world_cam(c, tracks[c][t])
            world.refresh()
            self._record(world, rec, t, cmd, grip[t], act, models, attached)
            for c in CAMS:
                rgb, vis, am_r, am_o = world.render(c, models[c])
                frames[c]["rgb"].append(rgb); frames[c]["vis"].append(vis)
                frames[c]["am_r"].append(am_r); frames[c]["am_o"].append(am_o)
        success = None
        if task is not None:
            op = world.data.xpos[world.obj_bodies[task["src"]]]
            hs = np.array(scene["objects"][task["src"]]["half"])
            if task["kind"] == "pick":
                success = bool(op[2] > L.table_h + hs[2] + 0.05 * L.R)
            else:
                pd = np.array(scene["objects"][task["dst"]]["pos"]); hd = np.array(scene["objects"][task["dst"]]["half"])
                success = bool(np.linalg.norm((op - pd)[:2]) < hd[:2].min() and abs(op[2] - (pd[2] + hd[2] + hs[2])) < 0.02)
            task["success"] = success
        meta = dict(robot=spec.name, mode=mode, fps=FPS, T=T, scene=scene, task=task,
                    cameras={c: dict(model=dict(width=models[c].width, height=models[c].height, fx=models[c].fx,
                                                fy=models[c].fy, cx=models[c].cx, cy=models[c].cy, kind=models[c].kind,
                                                dist=np.asarray(models[c].dist).tolist()), motion=cmodes[c]) for c in CAMS},
                    actuator=dict(delay_ms=act.delay * 1000 / (FPS * SUB), tau_s=act.tau, backlash_rad=act.backlash.tolist(),
                                  enc_noise_rad=act.noise),
                    arm_joints=spec.arm_joints, gripper_joints=spec.gripper_joints,
                    robot_bodies=[world.model.body(b).name for b in world.robot_bodies],
                    joint_names=[world.model.joint(j).name for j in world.joint_ids],
                    skeleton_names=self._skel_names, labels=dict(robot="1..N = robot_bodies index+1", object0=LABEL_OBJ0,
                                                                  table=LABEL_TABLE, table_leg=LABEL_LEG, floor=LABEL_FLOOR, background=0,
                                                                  distractor0=CL.LABEL_DIS0, clutter=CL.LABEL_CLUTTER),
                    gripper_width_table=dict(frac=self.grip_al.tolist(), width=self.grip_w.tolist()))
        arrays = {k: np.asarray(v, np.float32) for k, v in rec.items()}
        return meta, arrays, frames, world

    def _record(self, world: DataWorld, rec, t, cmd, g, act, models, attached):
        m, d = world.model, world.data
        Tb = np.linalg.inv(world.T_world_base())
        rec["q_cmd"].append(cmd[:self.n]); rec["grip_cmd"].append([g])
        rec["q_true"].append(act.q.copy()); rec["q_enc"].append(act.encoder())
        tp = Tb[:3, :3] @ d.site_xpos[world.tcp] + Tb[:3, 3]; tR = Tb[:3, :3] @ d.site_xmat[world.tcp].reshape(3, 3)
        rec["tcp_pose"].append(np.r_[tp, Rotation.from_matrix(tR).as_quat()])
        self.kin.set_joints(dict(zip(self.spec.arm_joints, cmd[:self.n])))
        sid = self.kin.model.site("tcp").id
        Tk = np.linalg.inv(self.kin.T_world_base())
        cp = Tk[:3, :3] @ self.kin.data.site_xpos[sid] + Tk[:3, 3]; cR = Tk[:3, :3] @ self.kin.data.site_xmat[sid].reshape(3, 3)
        rec["tcp_cmd_pose"].append(np.r_[cp, Rotation.from_matrix(cR).as_quat()])
        bp = []
        for b in world.robot_bodies + world.obj_bodies:
            p = Tb[:3, :3] @ d.xpos[b] + Tb[:3, 3]; R = Tb[:3, :3] @ d.xmat[b].reshape(3, 3)
            bp.append(np.r_[p, Rotation.from_matrix(R).as_quat()])
        rec["body_pose"].append(bp)
        rec["obj_pose"].append(bp[len(world.robot_bodies):])
        rec["joint_anchor"].append(world.to_base(d.xanchor[world.joint_ids]))
        rec["joint_axis"].append((Tb[:3, :3] @ d.xaxis[world.joint_ids].T).T)
        rec["attached"].append([-1 if attached is None else attached])
        # skeleton (same definition as the web demo) from the TRUE world state
        self.kin.set_joints({**dict(zip(self.spec.arm_joints, act.q[:self.n])), **dict(zip(self.spec.gripper_joints, act.q[self.n:]))})
        names, P, _ = skeleton_3d(self.kin, base_body="robot_mount", arm_joints=self.spec.arm_joints,
                                  ee_body=self.spec.ee_body, tcp_site="tcp", finger_joints=self.spec.gripper_joints or [], dedupe_m=0)
        self._skel_names = names
        Pw = P - self.kin.data.xpos[self.kin.model.body("robot_mount").id] + d.xpos[world.base_id]
        rec["skel3d"].append(world.to_base(Pw))
        objc = np.array([d.xpos[b] for b in world.obj_bodies])
        for c in CAMS:
            T = world.T_base_cam(c)
            rec[f"T_base_cam_{c}"].append(T)
            Tw = np.linalg.inv(world.T_world_base() @ T)
            for key, pts in ((f"skel2d_{c}", Pw), (f"obj2d_{c}", objc)):
                pc = pts @ Tw[:3, :3].T + Tw[:3, 3]
                uv, ok = models[c].project(pc)
                inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < models[c].width) & (uv[:, 1] >= 0) & (uv[:, 1] < models[c].height)
                rec[key].append(np.c_[uv, inside.astype(float)])


# =============================================================================== export
def robot_render_geoms(world: DataWorld):
    """Everything a stand-alone renderer needs to draw the robot exactly like MuJoCo does:
    per visual geom its body (index into robot_bodies), type, size, local pose, rgba, mesh."""
    m = world.model
    out = dict(body=[], type=[], size=[], pos=[], quat=[], rgba=[], mesh=[], is_attachment=[])
    meshes = {}
    for g in range(m.ngeom):
        if not world.is_robot[g] or m.geom_group[g] != 1:
            continue
        out["body"].append(world.robot_bodies.index(m.geom_bodyid[g]))
        out["type"].append(int(m.geom_type[g])); out["size"].append(m.geom_size[g].copy())
        out["pos"].append(m.geom_pos[g].copy()); out["quat"].append(m.geom_quat[g].copy())
        out["rgba"].append(m.geom_rgba[g].copy()); out["is_attachment"].append(m.geom(g).name.startswith("attach"))
        mid = int(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else -1
        out["mesh"].append(mid)
        if mid >= 0 and mid not in meshes:
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            meshes[mid] = (m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy())
    arr = {k: np.asarray(v) for k, v in out.items()}
    for mid, (v, f) in meshes.items():
        arr[f"mesh{mid}_v"] = v; arr[f"mesh{mid}_f"] = f
    arr["body_names"] = np.array([m.body(b).name for b in world.robot_bodies])
    return arr


def pack_images(imgs, kind):
    """List of images -> (concatenated encoded bytes, offsets). kind: jpg (rgb) | png (labels)."""
    bufs = []
    for im in imgs:
        if kind == "jpg":
            ok, b = cv2.imencode(".jpg", im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            ok, b = cv2.imencode(".png", im.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        bufs.append(b.tobytes())
    off = np.cumsum([0] + [len(b) for b in bufs]).astype(np.int64)
    return np.frombuffer(b"".join(bufs), np.uint8), off


def unpack_image(buf, off, i, kind):
    b = buf[off[i]:off[i + 1]]
    im = cv2.imdecode(b, cv2.IMREAD_COLOR if kind == "jpg" else cv2.IMREAD_UNCHANGED)
    return im[..., ::-1] if kind == "jpg" else im


def save_episode(out_dir, meta, arrays, frames):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1, default=lambda o: np.asarray(o).tolist())
    np.savez_compressed(os.path.join(out_dir, "arrays.npz"), **arrays)
    packed = {}
    for c, f in frames.items():
        for key, kind in (("rgb", "jpg"), ("vis", "png"), ("am_r", "png"), ("am_o", "png")):
            b, o = pack_images(f[key], kind)
            packed[f"{c}_{key}"] = b; packed[f"{c}_{key}_off"] = o
    np.savez(os.path.join(out_dir, "frames.npz"), **packed)
