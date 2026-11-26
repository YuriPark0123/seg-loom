"""Interactive kinematic-twin demo for ANY robot given as URDF + configs/robots/<name>.yaml.

Real world : generic MuJoCo scene (table, cube, top/front/wrist cameras) with the URDF robot.
Control    : grab the EEF in the 3D view -> IK on the twin -> joint PD (computed torque).
Views      : top / front / wrist camera images overlaid with the twin (mask + skeleton) placed with
             the ESTIMATED checkerboard calibration; cameras can be moved and re-calibrated.
(The original robosuite/Panda version lives in server_robosuite.py.)

Run:  MUJOCO_GL=egl python webapp/server.py --robot so101 --port 8765
"""
import argparse
import asyncio
import base64
import glob
import json
import os
import queue
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2
import mujoco
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from calibrate_cams import load_camera
from urdf_check import calibrate_all, save_camera, scene_key
from shadow.calibration import calibrate, pose_error
from shadow.ik import IKSolver
from shadow.kinematic_mask import Camera, KinematicMaskRenderer, RobotKinematics, fast_geom_segmentation
from shadow.mj_world import CAMS, Layout, World, twin_xml
from shadow.skeleton import draw_skeleton, project, skeleton_3d
from shadow.urdf_robot import RobotSpec

MOVABLE = ["top", "front"]
TOP_DOWN = "top"                     # always looks straight down (position only)
CAM_COLOR = {"top": (0.35, 0.65, 1.0), "front": (0.35, 1.0, 0.55)}


def available_robots():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, "configs/robots/*.yaml")))


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def cam_angles(R):
    f = -R[:, 2]
    pan, tilt = np.arctan2(f[1], f[0]), np.arcsin(np.clip(f[2], -1, 1))
    right0 = np.cross(f, [0, 0, 1.0]); n = np.linalg.norm(right0)
    right0 = right0 / n if n > 1e-6 else R[:, 0]
    up0 = np.cross(right0, f)
    return np.rad2deg([pan, tilt, np.arctan2(R[:, 0] @ up0, R[:, 0] @ right0)])


def cam_R(pan, tilt, roll):
    p, t, r = np.deg2rad([pan, tilt, roll])
    f = np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)])
    right0 = np.cross(f, [0, 0, 1.0]); right0 /= np.linalg.norm(right0)
    up0 = np.cross(right0, f)
    return np.stack([np.cos(r) * right0 + np.sin(r) * up0, -np.sin(r) * right0 + np.cos(r) * up0, -f], 1)


class Sim:
    """Owns every MuJoCo/EGL object of one robot session; only touched from the worker thread."""

    def __init__(self, robot, view_res=256, main_res=(480, 640), fps=25, calib_res=512, status_cb=None):
        self.spec = RobotSpec.load(robot)
        self.layout = L = Layout(self.spec)
        sp = self.spec
        self.view_res, self.main_res, self.fps, self.calib_res = view_res, main_res, fps, calib_res

        # ---- calibration (estimated from checkerboard images; cached per robot)
        cdir = os.path.join(ROOT, "out/calib", sp.name)
        key = scene_key(sp, L)                     # recalibrate whenever the scene/cameras/robot change
        kf = os.path.join(cdir, "scene.key")
        if not (all(os.path.exists(f"{cdir}/{c}.npz") for c in CAMS) and os.path.exists(kf) and open(kf).read() == key):
            status_cb and status_cb(f"{sp.name}: calibrating cameras with a checkerboard (first run, ~30 s)…")
            os.makedirs(cdir, exist_ok=True)
            for c, (cam, _, _) in calibrate_all(sp, L, res=calib_res).items():
                save_camera(f"{cdir}/{c}.npz", cam)
            open(kf, "w").write(key)
        self.cams_file = {c: load_camera(f"{cdir}/{c}.npz") for c in CAMS}
        self.cams = {c: self.cams_file[c].resized(view_res, view_res) for c in CAMS}

        # ---- real world
        self.world = World(sp, L, res=(view_res, view_res))
        self.m, self.d = self.world.model, self.world.data
        self.r_view = self.world.renderer
        self.r_main = mujoco.Renderer(self.m, *main_res)
        self.opt = self.world.opt
        self.ctrl_joints = sp.arm_joints + sp.gripper_joints
        self.dofs = np.array([self.world.dadr[j] for j in self.ctrl_joints])
        self.qadr = np.array([self.world.jadr[j] for j in self.ctrl_joints])
        self.n_arm = len(sp.arm_joints)
        self.cube_adr = self.m.jnt_qposadr[self.m.joint("cube_free").id]

        # ---- twin (robot-only, ESTIMATED cameras) + IK model
        txml = twin_xml(sp, L)
        self.twin = {c: KinematicMaskRenderer(xml=txml, camera=self.cams[c], base_body="robot_mount") for c in CAMS}
        self.kin = RobotKinematics(xml=txml, base_body="robot_mount")
        self.ik = IKSolver(self.kin, site="tcp", joint_names=sp.arm_joints, rest_q=sp.rest_q, rot_weight=sp.rot_weight)
        self.skel_kw = dict(base_body="robot_mount", arm_joints=sp.arm_joints, ee_body=sp.ee_body, tcp_site="tcp",
                            finger_joints=sp.gripper_joints or [])

        # ---- UI state
        self.free_cam = mujoco.MjvCamera(); self.free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._preset("robot")
        self.kp, self.kd, self.vmax = 400.0, 40.0, 1.5
        self.overlay = dict(mask=True, occlusion=True, skeleton=True, target=True, iou=True)
        self.main_drag, self.grab_px = None, 28
        self.cam_id = {c: self.m.camera(c).id for c in CAMS}
        self.cam_home = {c: (self.m.cam_pos[self.cam_id[c]].copy(), self.m.cam_quat[self.cam_id[c]].copy()) for c in MOVABLE}
        self.T_gt_calib = {c: self.world.T_base_cam_gt(c) for c in MOVABLE}
        self.T_est_calib = {c: self.cams_file[c].T.copy() for c in MOVABLE}
        self.known_move = False
        self.calib_status = {c: "calibrated (initial)" for c in MOVABLE}
        self.calib_busy, self.recalib_queue = None, []
        self.cmd = queue.Queue()
        self.latest, self.seq = None, 0
        self._reset()

    # ------------------------------------------------------------------ helpers
    def config(self):
        L = self.layout
        return dict(robot=self.spec.name, robots=available_robots(), reach=L.R,
                    ws_lo=L.ws_box[0].round(3).tolist(), ws_hi=L.ws_box[1].round(3).tolist(),
                    cam_box={c: [L.cam_box[c][0].round(2).tolist(), L.cam_box[c][1].round(2).tolist()] for c in MOVABLE},
                    has_gripper=bool(self.spec.gripper), n_arm=self.n_arm)

    def _preset(self, name):
        L, fc = self.layout, self.free_cam
        if name == "overview":
            fc.azimuth, fc.elevation = 150, -8
            fc.lookat[:] = [0.4 * L.R, 0, L.table_h + 1.3 * L.R]; fc.distance = 3.4 * L.R + 0.6
        else:
            fc.azimuth, fc.elevation = 150, -25
            fc.lookat[:] = [0.35 * L.R, 0, L.table_h + 0.25 * L.R]; fc.distance = 2.4 * L.R + 0.35

    def q_now(self):
        return self.d.qpos[self.qadr].copy()

    def _reset(self):
        mujoco.mj_resetData(self.m, self.d)
        sp, L = self.spec, self.layout
        self.world.set_joints(dict(zip(sp.arm_joints, sp.rest_q)))
        if sp.gripper:
            self.world.set_joints(dict(zip(sp.gripper_joints, sp.gripper["open"])))
        rng = np.random.default_rng()
        a = self.cube_adr
        self.d.qpos[a:a + 3] = L.cube_pos + np.r_[rng.uniform(-0.08, 0.08, 2) * L.R, 0]
        self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)
        self.q_des = self.q_now()[:self.n_arm]; self.q_cmd = self.q_now()
        self.kin.set_joints(dict(zip(sp.arm_joints, self.q_des)))
        self.p0, self.R0 = self.ik.site_pose()
        self.target_p, self.rpy = self.p0.copy(), np.zeros(3)
        self.gripper_open = True
        self.ik_err = (0.0, 0.0)

    def _solve(self):
        cr, sr = np.cos(self.rpy), np.sin(self.rpy)
        Rx = np.array([[1, 0, 0], [0, cr[0], -sr[0]], [0, sr[0], cr[0]]])
        Ry = np.array([[cr[1], 0, sr[1]], [0, 1, 0], [-sr[1], 0, cr[1]]])
        Rz = np.array([[cr[2], -sr[2], 0], [sr[2], cr[2], 0], [0, 0, 1]])
        q, ep, er = self.ik.solve(self.q_des, self.target_p, Rz @ Ry @ Rx @ self.R0)
        self.q_des, self.ik_err = q, (ep, er)

    def _clip_target(self, p):
        lo, hi = self.layout.ws_box
        pad = 0.4 * self.layout.R
        return np.clip(p, [lo[0] - pad, lo[1] - pad, self.layout.table_h + 0.01], hi + pad)

    # ------------------------------------------------------------------ main view geometry
    def _main_cam(self):
        fc = self.free_cam
        az, el = np.deg2rad(fc.azimuth), np.deg2rad(fc.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pos = np.array(fc.lookat) - fc.distance * fwd
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        H, W = self.main_res
        f = 0.5 * H / np.tan(np.deg2rad(self.m.vis.global_.fovy) / 2)
        K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1.0]])
        return pos, np.stack([right, np.cross(fwd, right), fwd], 1), K

    def main_project(self, p):
        pos, R, K = self._main_cam()
        pc = R.T @ (p - pos)
        return None if pc[2] <= 1e-6 else (K @ pc)[:2] / pc[2]

    def _main_hit(self, u, v, n, anchor):
        pos, R, K = self._main_cam()
        d = R @ np.linalg.solve(K, [u, v, 1.0]); d /= np.linalg.norm(d)
        den = d @ n
        if abs(den) < 1e-6:
            return None
        s = ((anchor - pos) @ n) / den
        return pos + s * d if s > 0 else None

    # ------------------------------------------------------------------ cameras
    def _cam_world(self, c):
        i = self.cam_id[c]
        return self.d.cam_xpos[i].copy(), self.d.cam_xmat[i].reshape(3, 3).copy()

    def _set_cam_world(self, c, pos=None, R=None):
        i = self.cam_id[c]
        if pos is not None:
            self.m.cam_pos[i] = np.clip(pos, *self.layout.cam_box[c])
        if R is not None:
            q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten()); self.m.cam_quat[i] = q
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)   # cam_xpos is set by camlight
        self._update_twin_cam(c)

    def _update_twin_cam(self, c):
        T = self.T_est_calib[c]
        if self.known_move:
            T = self.world.T_base_cam_gt(c) @ np.linalg.inv(self.T_gt_calib[c]) @ T
        old = self.cams[c]
        self.cams[c] = Camera(old.K, T, old.height, old.width, None, old.dist)
        self.twin[c].set_camera(self.cams[c])

    def _handles(self):
        h = {"eef": self.target_p}
        for c in MOVABLE:
            h["cam:" + c] = self._cam_world(c)[0]
        return h

    def _pick(self, u, v):
        best, bd = None, self.grab_px
        for k, p in self._handles().items():
            uv = self.main_project(p)
            if uv is not None and np.hypot(*(uv - [u, v])) < bd:
                best, bd = k, np.hypot(*(uv - [u, v]))
        return best

    def _plane_drag(self, md, u, v, shift, get_p):
        n = np.array([0, 0, 1.0]) if shift else self._main_cam()[1][:, 2]
        if md["shift"] != shift:
            h = self._main_hit(*md["last"], n, get_p())
            md["offset"], md["shift"] = (get_p() - h if h is not None else np.zeros(3)), shift
        h = self._main_hit(u, v, n, get_p())
        return None if h is None else h + md["offset"]

    def _pointer(self, c):
        t, u, v, shift = c["type"], float(c.get("u", 0)), float(c.get("v", 0)), bool(c.get("shift", False))
        md = self.main_drag
        if t == "main_down":
            k, btn = self._pick(u, v), int(c.get("button", 0))
            if k == "eef" and btn == 0:
                mode = "eef"
            elif k and k.startswith("cam:"):
                mode = "cam_move" if (btn == 0 or k[4:] == TOP_DOWN) else "cam_rot"
            else:
                mode = "orbit"
            self.main_drag = dict(mode=mode, cam=k[4:] if k and k.startswith("cam:") else None, last=(u, v), shift=None)
        elif t == "main_up":
            self.main_drag = None
        elif t == "main_move" and md is not None:
            du, dv = u - md["last"][0], v - md["last"][1]
            if md["mode"] == "orbit":
                fc = self.free_cam
                fc.azimuth -= 0.4 * du; fc.elevation = float(np.clip(fc.elevation - 0.3 * dv, -89, 89))
            elif md["mode"] == "eef":
                p = self._plane_drag(md, u, v, shift, lambda: self.target_p)
                if p is not None:
                    self.target_p = self._clip_target(p); self._solve()
            elif md["mode"] == "cam_move":
                cam = md["cam"]
                p = self._plane_drag(md, u, v, shift or cam == TOP_DOWN, lambda: self._cam_world(cam)[0])
                if p is not None:
                    self._set_cam_world(cam, pos=p)
            elif md["mode"] == "cam_rot":
                cam = md["cam"]
                pan, tilt, roll = cam_angles(self._cam_world(cam)[1])
                self._set_cam_world(cam, R=cam_R(pan - 0.3 * du, np.clip(tilt - 0.3 * dv, -89, 30), roll))
            md["last"] = (u, v)

    def _wheel(self, c):
        k = self._pick(float(c.get("u", -1e9)), float(c.get("v", -1e9)))
        if k and k.startswith("cam:"):
            cam = k[4:]; p, R = self._cam_world(cam)
            self._set_cam_world(cam, pos=p + (-R[:, 2]) * (-0.06 if c["dy"] > 0 else 0.06) * self.layout.R)
        else:
            fc = self.free_cam
            fc.distance = float(np.clip(fc.distance * (1.08 if c["dy"] > 0 else 1 / 1.08), 0.2, 10.0))

    def _view_drag(self, cam, u, v):
        """Drag inside a calibrated camera image: ray through the ESTIMATED camera."""
        c = self.cams[cam]
        self.kin.set_joints(self.kin.joints_from_mj(self.m, self.d))
        T = self.kin.T_world_cam(c)
        d = T[:3, :3] @ np.linalg.solve(c.K, [u, v, 1.0]); d /= np.linalg.norm(d)
        if cam == "top":
            n = np.array([0, 0, 1.0])
        else:
            f = T[:3, 2].copy(); f[2] = 0; n = f / (np.linalg.norm(f) + 1e-9)
        den = d @ n
        if abs(den) < 1e-6:
            return
        s = ((self.target_p - T[:3, 3]) @ n) / den
        if s > 0:
            self.target_p = self._clip_target(T[:3, 3] + s * d); self._solve()

    # ------------------------------------------------------------------ re-calibration (extrinsics only)
    def _moved_since_calib(self, c):
        et, er = pose_error(self.world.T_base_cam_gt(c), self.T_gt_calib[c])
        return et > 1.0 or er > 0.1

    def _next_recalib(self):
        if not self.calib_busy and self.recalib_queue:
            self._start_recalib(self.recalib_queue.pop(0))

    def _start_recalib(self, c):
        i = self.cam_id[c]
        pos, quat, T_gt = self.m.cam_pos[i].copy(), self.m.cam_quat[i].copy(), self.world.T_base_cam_gt(c)
        cf, sp, L = self.cams_file[c], self.spec, self.layout
        self.kin.set_joints(self.kin.joints_from_mj(self.m, self.d))
        guess = self.kin.T_world_cam(self.cams[c])[:3, 3]      # stale estimate: rough hint where the camera is
        self.calib_busy, self.calib_status[c] = c, "re-calibrating… (robot runs checkerboard routine)"

        def job():
            try:
                scene = World(sp, L, board="hand", res=(cf.height, cf.width))
                j = scene.model.camera(c).id
                scene.model.cam_pos[j], scene.model.cam_quat[j] = pos, quat
                kin = RobotKinematics(xml=twin_xml(sp, L), base_body="robot_mount")
                cam, info = calibrate(scene, kin, c, "static", sampler="workspace", ws_box=tuple(zip(*L.ws_box)),
                                      K_fixed=cf.K, dist_fixed=cf.dist, arm_joints=sp.arm_joints, tcp_site="tcp",
                                      rot_weight=sp.rot_weight, hand_body=sp.ee_body, center=np.array(sp.rest_q),
                                      pos_tol=0.02 * sp.reach, n_try=200, n_keep=25, verbose=False,
                                      face_point=guess)
                if info["reproj_rms_px_joint"] > 2.0:          # quality gate: keep the previous estimate
                    raise RuntimeError(f"poor fit (reproj {info['reproj_rms_px_joint']:.1f}px)")
                self.cmd.put(dict(type="_calib_done", cam=c, T=cam.T, T_gt=T_gt, n=info["n_views"],
                                  rms=info["reproj_rms_px_joint"]))
            except Exception as e:
                self.cmd.put(dict(type="_calib_fail", cam=c, err=str(e)))

        threading.Thread(target=job, daemon=True).start()

    # ------------------------------------------------------------------ commands
    def handle(self, c):
        t = c.get("type")
        if t in ("main_down", "main_move", "main_up"):
            return self._pointer(c)
        if t == "wheel":
            return self._wheel(c)
        if t == "target":
            if "pos" in c:
                self.target_p = self._clip_target(np.array(c["pos"], float))
            if "rpy_deg" in c:
                self.rpy = np.deg2rad(np.array(c["rpy_deg"], float))
            self._solve()
        elif t == "drag" and c.get("view") in MOVABLE:
            self._view_drag(c["view"], float(c["u"]), float(c["v"]))
        elif t == "gripper":
            self.gripper_open = bool(c["open"])
        elif t == "reset":
            self._reset()
        elif t == "home":
            self.target_p, self.rpy = self.p0.copy(), np.zeros(3); self._solve()
        elif t == "view_preset":
            self._preset(c.get("name", "robot"))
        elif t == "overlay":
            self.overlay.update({k: bool(v) for k, v in c.items() if k in self.overlay})
        elif t == "known_move":
            self.known_move = bool(c["value"])
            for cam in MOVABLE:
                self._update_twin_cam(cam)
        elif t == "cam_set" and c.get("cam") in MOVABLE:
            cam = c["cam"]
            self._set_cam_world(cam, pos=np.array(c["pos"], float), R=None if cam == TOP_DOWN else cam_R(*c["ang"]))
        elif t == "recalib":
            cams = MOVABLE if c.get("cam", "all") == "all" else [c["cam"]]
            if c.get("cam", "all") == "all":
                cams = [k for k in cams if self._moved_since_calib(k)] or cams
            self.recalib_queue += [k for k in cams if k not in self.recalib_queue and k != self.calib_busy]
            for k in self.recalib_queue:
                self.calib_status[k] = "queued"
            self._next_recalib()
        elif t == "_calib_done":
            cam = c["cam"]
            self.T_est_calib[cam], self.T_gt_calib[cam] = c["T"], c["T_gt"]
            self._update_twin_cam(cam)
            self.calib_status[cam] = f"re-calibrated: {c['n']} views, reproj {c['rms']:.2f}px"
            self.calib_busy = None; self._next_recalib()
        elif t == "_calib_fail":
            self.calib_status[c["cam"]] = "re-calibration failed: " + c["err"][:80]
            self.calib_busy = None; self._next_recalib()
        elif t == "cam_home" and c.get("cam") in MOVABLE and not self.calib_busy:
            cam = c["cam"]; i = self.cam_id[cam]
            self.m.cam_pos[i], self.m.cam_quat[i] = self.cam_home[cam]
            mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)
            self.T_est_calib[cam] = self.cams_file[cam].T.copy()
            self.T_gt_calib[cam] = self.world.T_base_cam_gt(cam)
            self._update_twin_cam(cam)
            self.calib_status[cam] = "calibrated (initial)"

    # ------------------------------------------------------------------ physics
    def _physics_step(self):
        dt = self.m.opt.timestep
        des = self.q_des
        if self.spec.gripper:
            des = np.r_[des, self.spec.gripper["open" if self.gripper_open else "close"]]
        self.q_cmd += np.clip(des - self.q_cmd, -self.vmax * dt, self.vmax * dt)
        q, qd = self.d.qpos[self.qadr], self.d.qvel[self.dofs]
        M = np.zeros((self.m.nv, self.m.nv)); mujoco.mj_fullM(self.m, M, self.d.qM)
        tau = M[np.ix_(self.dofs, self.dofs)] @ (self.kp * (self.q_cmd - q) - self.kd * qd) + self.d.qfrc_bias[self.dofs]
        if self.spec.gripper:                       # gentle gripper: limited force so grasps don't explode
            tau[self.n_arm:] = np.clip(tau[self.n_arm:], -8.0, 8.0)
        self.d.qfrc_applied[:] = 0
        self.d.qfrc_applied[self.dofs] = tau
        mujoco.mj_step(self.m, self.d)

    # ------------------------------------------------------------------ rendering
    def _amodal(self, cam):
        saved = self.m.geom_group.copy()
        self.m.geom_group[~self.world.is_robot[:-1]] = 5
        gid = fast_geom_segmentation(self.r_view, self.d, cam, self.opt)
        self.m.geom_group[:] = saved
        return self.world.is_robot[gid]

    def _marker(self, img, cam, p, color, kin):
        uv, ok = project(p[None], cam, kin)
        if ok[0]:
            u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
            cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(img, (u, v), 7, color, 1, cv2.LINE_AA)

    def render(self):
        js = self.kin.joints_from_mj(self.m, self.d)          # encoders only
        self.kin.set_joints(js)
        _, P, edges = skeleton_3d(self.kin, **self.skel_kw)
        views, ious = {}, {}
        for c in CAMS:
            self.r_view.update_scene(self.d, camera=c, scene_option=self.opt)
            img = self.r_view.render().copy()
            tw = self.twin[c]
            mask = tw.render(js)
            vis = mask
            if self.overlay["occlusion"]:
                r = self.r_view
                r.enable_depth_rendering(); r.update_scene(self.d, camera=c, scene_option=self.opt)
                real_depth = r.render().copy(); r.disable_depth_rendering()
                # depth tolerance grows with distance (calibration depth error does too)
                vis = mask & (tw.render_depth() <= real_depth * 1.015 + 0.005)
            if self.overlay["iou"]:
                g_vis = self.world.is_robot[fast_geom_segmentation(self.r_view, self.d, c, self.opt)]
                ious[c] = dict(amodal=_iou(self._amodal(c), mask), visible=_iou(g_vis, vis))
            if self.overlay["mask"]:
                img[vis] = (0.45 * img[vis] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
                cnts, _ = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnts, -1, (0, 255, 255), 1, cv2.LINE_AA)
                if self.overlay["occlusion"]:
                    cnts, _ = cv2.findContours((mask & ~vis).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(img, cnts, -1, (255, 120, 255), 1, cv2.LINE_AA)
            if self.overlay["skeleton"]:
                uv, ok = project(P, tw.camera, tw)
                img = draw_skeleton(img, uv, ok, edges, radius=3)
            if self.overlay["target"]:
                self._marker(img, tw.camera, self.target_p, (255, 60, 60), tw)
            views[c] = img
        r = self.r_main
        r.update_scene(self.d, camera=self.free_cam, scene_option=self.opt)
        sc = r.scene
        if sc.ngeom < sc.maxgeom:
            active = bool(self.main_drag and self.main_drag["mode"] == "eef")
            rgba = np.array([1, 0.85, 0.1, 0.75] if active else [1, 0.2, 0.2, 0.6], np.float32)
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([max(0.012, 0.035 * self.layout.R), 0, 0]), self.target_p, np.eye(3).flatten(), rgba)
            sc.ngeom += 1
        self._draw_cam_gizmos(sc)
        views["main"] = r.render().copy()
        return views, ious

    def _draw_cam_gizmos(self, sc):
        md = self.main_drag
        sz = max(0.02, 0.045 * self.layout.R)
        for c in MOVABLE:
            if sc.ngeom + 10 > sc.maxgeom:
                return
            p, R = self._cam_world(c)
            col = np.array((1, 0.85, 0.1, 1) if (md and md.get("cam") == c) else (*CAM_COLOR[c], 1), np.float32)
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([sz, sz * 0.75, sz * 0.8]),
                                p, R.flatten(), col)
            sc.ngeom += 1
            t = np.tan(np.deg2rad(self.m.cam_fovy[self.cam_id[c]]) / 2)
            L = 0.5 * self.layout.R + 0.05
            corners = [p + R @ (np.array([sx * t, sy * t, -1.0]) * L) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            for a, b in [(p, q) for q in corners] + [(corners[k], corners[(k + 1) % 4]) for k in range(4)]:
                g = sc.geoms[sc.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), col)
                mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, *a, *b)
                sc.ngeom += 1

    def _cam_state(self):
        out = {}
        for c in MOVABLE:
            T_gt = self.world.T_base_cam_gt(c)
            et, er = pose_error(self.cams[c].T, T_gt)
            mt, mr = pose_error(T_gt, self.T_gt_calib[c])
            p, R = self._cam_world(c)
            out[c] = dict(err_mm=et, err_deg=er, moved_mm=mt, moved_deg=mr, status=self.calib_status[c],
                          pos=p.round(3).tolist(), ang=cam_angles(R).round(1).tolist(), busy=self.calib_busy == c)
        return out

    # ------------------------------------------------------------------ loop
    def run(self, stop: threading.Event, switch: dict):
        dt = self.m.opt.timestep
        t_sim, t0w = 0.0, time.perf_counter()
        next_frame, ft = 0.0, []
        tcp_site = self.m.site("tcp").id
        while not stop.is_set() and not switch.get("name"):
            while True:
                try:
                    self.handle(self.cmd.get_nowait())
                except queue.Empty:
                    break
            now = time.perf_counter() - t0w
            n = 0
            while t_sim < now and n < 100:
                self._physics_step(); t_sim += dt; n += 1
            if t_sim < now - 0.2:
                t_sim = now
            if now >= next_frame:
                t1 = time.perf_counter()
                views, ious = self.render()
                ft.append(time.perf_counter() - t1); ft = ft[-30:]
                enc = {k: base64.b64encode(cv2.imencode(".jpg", v[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
                       for k, v in views.items()}
                p = self.d.site_xpos[tcp_site].copy()
                self.seq += 1
                self.latest = json.dumps(dict(
                    seq=self.seq, views=enc, iou=ious, config=self.config(),
                    state=dict(target=self.target_p.round(4).tolist(), rpy_deg=np.rad2deg(self.rpy).round(1).tolist(),
                               eef=p.round(4).tolist(), track_err_mm=float(np.linalg.norm(p - self.target_p) * 1000),
                               ik_err_mm=self.ik_err[0] * 1000, ik_err_deg=float(np.rad2deg(self.ik_err[1])),
                               q=self.q_now()[:self.n_arm].round(3).tolist(), gripper_open=self.gripper_open,
                               render_ms=float(np.mean(ft) * 1000), overlay=self.overlay, main_res=list(self.main_res),
                               grab_px=self.grab_px,
                               handles={k: (None if (uv := self.main_project(v)) is None else uv.round(1).tolist())
                                        for k, v in self._handles().items()},
                               cams=self._cam_state(), known_move=self.known_move,
                               recalib_busy=self.calib_busy, recalib_queue=list(self.recalib_queue))))
                next_frame = now + 1.0 / self.fps
            else:
                time.sleep(0.001)


class Hub:
    """Holds the current Sim and lets the browser switch robots (rebuilds the Sim in the worker thread)."""

    def __init__(self, robot, fps):
        self.robot, self.fps = robot, fps
        self.sim, self.status, self.switch = None, "starting…", {}
        self.stop = threading.Event()

    def worker(self, ready: threading.Event):
        while not self.stop.is_set():
            name = self.switch.pop("name", None) or self.robot
            self.robot = name
            self.status = f"loading {name}…"
            try:
                sim = Sim(name, fps=self.fps, status_cb=lambda s: setattr(self, "status", s))
            except Exception as e:                    # bad URDF/spec: report and wait for another choice
                self.status = f"failed to load {name}: {e}"
                ready.set()
                while not self.switch.get("name") and not self.stop.is_set():
                    time.sleep(0.1)
                continue
            self.sim, self.status = sim, "ok"
            ready.set()
            sim.run(self.stop, self.switch)
            self.sim = None


def make_app(hub: Hub):
    app = FastAPI()
    here = os.path.dirname(os.path.abspath(__file__))

    @app.get("/")
    def index():
        return FileResponse(os.path.join(here, "index.html"))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()

        async def sender():
            last = (None, -1)
            while True:
                s = hub.sim
                if s is not None and s.latest is not None and (id(s), s.seq) != last:
                    last = (id(s), s.seq)
                    await sock.send_text(s.latest)
                elif s is None:
                    await sock.send_text(json.dumps(dict(loading=hub.status, robots=available_robots())))
                    await asyncio.sleep(0.3)
                await asyncio.sleep(0.01)

        task = asyncio.create_task(sender())
        try:
            while True:
                c = json.loads(await sock.receive_text())
                if c.get("type") == "switch_robot" and c.get("name") in available_robots():
                    hub.switch["name"] = c["name"]
                elif hub.sim is not None:
                    hub.sim.cmd.put(c)
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="so101", help=f"one of {available_robots()}")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    hub = Hub(args.robot, args.fps)
    ready = threading.Event()
    threading.Thread(target=hub.worker, args=(ready,), daemon=True).start()
    ready.wait()
    print(f"ready ({hub.status}): http://localhost:{args.port}", flush=True)
    try:
        uvicorn.run(make_app(hub), host=args.host, port=args.port, log_level="warning")
    finally:
        hub.stop.set()


if __name__ == "__main__":
    main()
