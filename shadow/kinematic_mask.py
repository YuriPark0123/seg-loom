"""Kinematic robot mask renderer.

The mask is NOT estimated from the image. It is rendered from
  (robot kinematic model, joint encoder readings, camera K, camera extrinsics)
so the robot's identity is fixed by construction: the pixels returned belong to
*this* robot at *this* joint state seen from *this* (calibrated) camera.

A standalone MuJoCo scene containing only the robot (arm + gripper + mount) and a
camera is built; it never sees the task scene or its images.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

def apply_joint_couplings(m, d, explicit=()):
    """Set follower joints of joint equalities (URDF mimic -> <equality joint>) from their leaders:
    q1 - q1_0 = c0 + c1 (q2 - q2_0) + c2 (..)^2 + ...  Kinematic-only code (no mj_step) never enforces
    the constraint, so without this mimic fingers stay at 0. Joints named in `explicit` are left as set."""
    eqs = getattr(m, "_couplings", None)
    if eqs is None:
        eqs = []
        for e in range(m.neq):
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_obj2id[e] >= 0:
                j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
                eqs.append((m.joint(j1).name, m.jnt_qposadr[j1], m.jnt_qposadr[j2], m.eq_data[e][:5].copy()))
        try:
            m._couplings = eqs
        except AttributeError:
            pass
    for _ in range(3):                       # a few passes: follower of a follower
        for n1, a1, a2, c in eqs:
            if n1 in explicit:
                continue
            x = d.qpos[a2] - m.qpos0[a2]
            d.qpos[a1] = m.qpos0[a1] + c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4


# OpenCV camera (x right, y down, z forward) <-> MuJoCo camera (x right, y up, -z forward)
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def make_T(R=np.eye(3), t=np.zeros(3)):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def inv_T(T):
    return make_T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def ideal_K(fovy_deg, height, width):
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1.0]])


def scale_K(K, sx, sy):
    """Intrinsics after resizing the image by (sx, sy) (pixel-center convention)."""
    K2 = K.copy()
    K2[0, 0] *= sx; K2[1, 1] *= sy
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K2


@dataclass
class Camera:
    """Pinhole camera, OpenCV convention.

    Static camera : parent_body=None, T is T_base_cam (robot base frame -> camera)
    Mounted camera: parent_body='robot0_right_hand', T is T_parent_cam (hand-eye result)
    """
    K: np.ndarray
    T: np.ndarray
    height: int
    width: int
    parent_body: str | None = None
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def resized(self, height, width):
        return Camera(scale_K(self.K, width / self.width, height / self.height), self.T.copy(),
                      height, width, self.parent_body, self.dist.copy())

    def perturbed(self, trans_m=0.0, rot_deg=0.0, rng=None):
        """Extrinsic error of fixed magnitude in a random direction (calibration noise model)."""
        rng = np.random.default_rng() if rng is None else rng
        T = self.T.copy()
        if trans_m > 0:
            d = rng.normal(size=3); T[:3, 3] += trans_m * d / np.linalg.norm(d)
        if rot_deg > 0:
            a = rng.normal(size=3); a /= np.linalg.norm(a)
            T[:3, :3] = T[:3, :3] @ cv2.Rodrigues(a * np.deg2rad(rot_deg))[0]
        return Camera(self.K.copy(), T, self.height, self.width, self.parent_body, self.dist.copy())


def gt_camera_from_sim(sim, cam_name, height, width, base_body="robot0_base"):
    """Ground-truth camera from a robosuite sim. Used ONLY to score calibration estimates."""
    m, d = sim.model, sim.data
    cid = m.camera_name2id(cam_name)
    K = ideal_K(float(m.cam_fovy[cid]), height, width)
    parent = m.body_id2name(m.cam_bodyid[cid])
    if parent == "world":
        bid = m.body_name2id(base_body)
        T_w_base = make_T(d.body_xmat[bid].reshape(3, 3), d.body_xpos[bid])
        T_w_cam = make_T(d.cam_xmat[cid].reshape(3, 3) @ CV_TO_MJ, d.cam_xpos[cid])
        return Camera(K, inv_T(T_w_base) @ T_w_cam, height, width, None)
    q = m.cam_quat[cid]; R = np.zeros(9); mujoco.mju_quat2Mat(R, q)
    return Camera(K, make_T(R.reshape(3, 3) @ CV_TO_MJ, m.cam_pos[cid]), height, width, parent)


def build_robot_xml(robot_name, gripper="default", mount="default", base_pos=(0, 0, 0),
                    base_euler=(0, 0, 0), idn=0):
    """Robot-only MJCF (arm + gripper + mount) from robosuite's robot assets."""
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.mounts import mount_factory
    from robosuite.models.robots import create_robot
    from robosuite.models.world import MujocoWorldBase
    from robosuite.utils.mjcf_utils import new_element

    robot = create_robot(robot_name, idn=idn)
    if mount is not None:
        robot.add_mount(mount_factory(robot.default_mount if mount == "default" else mount, idn=idn))
    if gripper is not None:
        robot.add_gripper(gripper_factory(robot.default_gripper if gripper == "default" else gripper, idn=idn))
    robot.set_base_xpos(np.asarray(base_pos, float))
    robot.set_base_ori(np.asarray(base_euler, float))
    world = MujocoWorldBase()
    world.merge(robot)
    world.worldbody.append(new_element("camera", name="maskcam", pos="0 0 0", quat="1 0 0 0", fovy="45"))
    return world.get_xml()


def segmentation_selftest():
    """Render two geoms with the id-color path and check both ids come back. mujoco 2.3.7 with numpy >= 2 silently
    renders lit colors instead of ids, so every label map comes out empty (seen on the B200 server). Raises
    RuntimeError with the versions if the path is broken."""
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/><camera name="c" pos="0 0 2"/>'
        '<geom type="box" size=".3 .3 .1" pos="-.4 0 0"/><geom type="sphere" size=".25" pos=".4 0 0"/>'
        '</worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 60, 80)
    try:
        ids = set(np.unique(fast_geom_segmentation(r, d, 0, mujoco.MjvOption())).tolist())
    finally:
        if hasattr(r, "close"):
            r.close()
    if not {0, 1} <= ids:
        raise RuntimeError(f"segmentation render is broken (got ids {sorted(ids)}; mujoco {mujoco.__version__}, "
                           f"numpy {np.__version__}): use numpy<2 with mujoco 2.3.7")


def fast_geom_segmentation(renderer, data, camera, opt):
    """Per-pixel geom id (-1 = background / non-geom).

    Same result as Renderer.enable_segmentation_rendering(), but decodes the id-color image with
    vectorized numpy (mujoco 2.3.7's built-in decoding costs ~20 ms per 256x256 frame).
    """
    renderer.update_scene(data, camera=camera, scene_option=opt)
    scn = renderer.scene
    seg_f, id_f = mujoco.mjtRndFlag.mjRND_SEGMENT, mujoco.mjtRndFlag.mjRND_IDCOLOR
    old = scn.flags[seg_f], scn.flags[id_f]
    scn.flags[seg_f] = 1; scn.flags[id_f] = 1
    try:
        img = renderer.render()
    finally:
        scn.flags[seg_f], scn.flags[id_f] = old
    sid = img[..., 0].astype(np.int32) + img[..., 1].astype(np.int32) * 256 + img[..., 2].astype(np.int32) * 65536 - 1
    n = scn.ngeom
    lut = np.full(n + 1, -1, np.int32)        # index n (== -1 after wrap) -> background
    for i in range(n):
        g = scn.geoms[i]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
            lut[i] = g.objid
    sid = np.where((sid >= 0) & (sid < n), sid, n)
    return lut[sid]


class RobotKinematics:
    """Standalone kinematic model of one robot (FK only, no physics)."""

    def __init__(self, robot_name=None, *, xml=None, gripper="default", mount="default", base_pos=(0, 0, 0),
                 base_euler=(0, 0, 0), base_body="robot0_base"):
        """robot_name: a robosuite robot (legacy path); xml: any robot-only MJCF (e.g. from a URDF)
        that contains a camera named 'maskcam'."""
        self.robot_name = robot_name
        self.xml = xml if xml is not None else build_robot_xml(robot_name, gripper, mount, base_pos, base_euler)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.base_body = base_body
        self.joint_names = [self.model.joint(j).name for j in range(self.model.njnt)]
        self._jadr = {n: self.model.jnt_qposadr[j] for j, n in enumerate(self.joint_names)}
        mujoco.mj_kinematics(self.model, self.data)

    def set_joints(self, joints: dict):
        for n, v in joints.items():
            if n in self._jadr:
                self.data.qpos[self._jadr[n]] = v
        apply_joint_couplings(self.model, self.data, joints)
        mujoco.mj_kinematics(self.model, self.data)

    def joints_from_mj(self, model, data):
        """Read joint encoders (by name) from any MuJoCo model/data - proprioception only."""
        js = {}
        for n in self.joint_names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                js[n] = float(data.qpos[model.jnt_qposadr[j]])
        return js

    def joints_from_sim(self, sim, noise_rad=0.0, rng=None):
        """Read joint encoders (by name) from the robot's sim - proprioception only."""
        js = {}
        for n in self.joint_names:
            try:
                js[n] = float(sim.data.qpos[sim.model.get_joint_qpos_addr(n)])
            except Exception:
                continue
        if noise_rad > 0:
            rng = np.random.default_rng() if rng is None else rng
            js = {n: v + (rng.normal() * noise_rad if n.startswith("robot0_joint") or "robot0_" in n else 0.0)
                  for n, v in js.items()}
        return js

    def body_T(self, name):
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return make_T(self.data.xmat[b].reshape(3, 3), self.data.xpos[b])

    def T_world_base(self):
        return self.body_T(self.base_body)

    def T_base(self, body):
        return inv_T(self.T_world_base()) @ self.body_T(body)

    def T_world_cam(self, cam: Camera):
        if cam.parent_body is None:
            return self.T_world_base() @ cam.T
        return self.body_T(cam.parent_body) @ cam.T


class KinematicMaskRenderer(RobotKinematics):
    """Render a binary mask of this robot from joint state + camera geometry."""

    def __init__(self, robot_name=None, camera: Camera = None, *, include_mount=True, **kw):
        super().__init__(robot_name, **kw)
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "maskcam")
        self._renderer = None
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 0
        self._opt.geomgroup[1] = 1  # robosuite visual meshes live in group 1
        self._opt.sitegroup[:] = 0   # no debug sites in the twin view
        self._valid_geom = np.ones(self.model.ngeom, bool)
        if not include_mount:
            for g in range(self.model.ngeom):
                self._valid_geom[g] = not self.model.body(self.model.geom_bodyid[g]).name.startswith("mount")
        self.set_camera(camera)

    def set_camera(self, camera: Camera):
        self.camera = camera
        if self._renderer is None or (self._renderer.height, self._renderer.width) != (camera.height, camera.width):
            if self._renderer is not None:
                getattr(self._renderer, 'close', lambda: None)()
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, camera.width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, camera.height)
            self._renderer = mujoco.Renderer(self.model, camera.height, camera.width)
        # Render with an ideal (centered, square-pixel) camera whose fovy matches fy, then
        # warp to the real K with the pure-intrinsic homography H = K_real @ K_ideal^-1.
        fy = camera.K[1, 1]
        self._fovy = float(np.rad2deg(2 * np.arctan(0.5 * camera.height / fy)))
        K_r = ideal_K(self._fovy, camera.height, camera.width)
        self._H = camera.K @ np.linalg.inv(K_r)
        self._need_warp = not np.allclose(self._H, np.eye(3), atol=1e-6)

    def _place_camera(self):
        T = self.T_world_cam(self.camera)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, (T[:3, :3] @ CV_TO_MJ).flatten())
        self.model.cam_pos[self._cam_id] = T[:3, 3]
        self.model.cam_quat[self._cam_id] = q
        self.model.cam_fovy[self._cam_id] = self._fovy

    def render(self, joints: dict | None = None):
        if joints is not None:
            self.set_joints(joints)
        self._place_camera()
        mujoco.mj_camlight(self.model, self.data)
        gid = fast_geom_segmentation(self._renderer, self.data, self._cam_id, self._opt)
        mask = np.zeros(gid.shape, bool)
        hit = gid >= 0
        mask[hit] = self._valid_geom[gid[hit]]
        if self._need_warp:
            mask = cv2.warpPerspective(mask.astype(np.uint8), self._H, (self.camera.width, self.camera.height),
                                       flags=cv2.INTER_NEAREST) > 0
        return mask

    def render_depth(self):
        """Metric depth (m) of the twin through the current camera; +inf where there is no robot.
        Call after render() so the joints/camera are already placed."""
        r = self._renderer
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self._cam_id, scene_option=self._opt)
        d = r.render().copy()
        r.disable_depth_rendering()
        if self._need_warp:
            d = cv2.warpPerspective(d, self._H, (self.camera.width, self.camera.height), flags=cv2.INTER_NEAREST,
                                    borderValue=float("inf"))
        return d

    def render_rgb(self, joints: dict | None = None):
        """RGB view of the digital twin (robot only, black background) through the same camera."""
        if joints is not None:
            self.set_joints(joints)
        self._place_camera()
        mujoco.mj_camlight(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self._cam_id, scene_option=self._opt)
        img = self._renderer.render()
        if self._need_warp:
            img = cv2.warpPerspective(img, self._H, (self.camera.width, self.camera.height), flags=cv2.INTER_LINEAR)
        return img

    def close(self):
        if self._renderer is not None:
            getattr(self._renderer, 'close', lambda: None)(); self._renderer = None
