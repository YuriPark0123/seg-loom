"""Interactive web demo.

Real world  : robosuite Lift (Panda) physics, running in real time.
Control     : drag the EEF target in the browser -> IK on the twin -> joint PD torque control.
Views       : free orbit view + calibrated cameras (top view, front-top, wrist), each overlaid with
              the kinematic twin (mask + skeleton) placed with the ESTIMATED calibration.

Run:  MUJOCO_GL=egl python webapp/server.py --port 8765   then open http://localhost:8765
"""
import argparse
import asyncio
import base64
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
import robosuite as suite
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from calibrate_cams import load_camera
from shadow.calibration import RealWorldCalibScene, calibrate, pose_error
from shadow.ik import IKSolver
from shadow.kinematic_mask import (CV_TO_MJ, Camera, KinematicMaskRenderer, RobotKinematics,
                                   fast_geom_segmentation, gt_camera_from_sim)
from shadow.skeleton import draw_skeleton, project, skeleton_3d

CAMS = ["birdview", "agentview", "robot0_eye_in_hand"]
MOVABLE = ["birdview", "agentview"]           # cameras the user can relocate
CAM_COLOR = {"birdview": (0.35, 0.65, 1.0), "agentview": (0.35, 1.0, 0.55)}
LABEL = {"birdview": "top view", "agentview": "front-top", "robot0_eye_in_hand": "gripper cam"}


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def _rot(axis, ang):
    return cv2.Rodrigues(np.asarray(axis, float) * ang)[0]


TOP_DOWN = "birdview"                          # this camera always looks straight down (position only)
CAM_BOX = {"birdview": ([-1.2, -1.2, 1.3], [1.2, 1.2, 3.5]), "agentview": ([-0.6, -1.8, 0.85], [2.2, 1.8, 2.6])}


def cam_angles(R):
    """MuJoCo camera matrix (x right, y up, z backward) -> (pan, tilt, roll) in degrees."""
    f = -R[:, 2]
    pan, tilt = np.arctan2(f[1], f[0]), np.arcsin(np.clip(f[2], -1, 1))
    right0 = np.cross(f, [0, 0, 1.0]); n = np.linalg.norm(right0)
    right0 = right0 / n if n > 1e-6 else R[:, 0]
    up0 = np.cross(right0, f)
    roll = np.arctan2(R[:, 0] @ up0, R[:, 0] @ right0)
    return np.rad2deg([pan, tilt, roll])


def cam_R(pan, tilt, roll):
    """(pan, tilt, roll) degrees -> MuJoCo camera matrix. tilt<0 looks down; roll about the optical axis."""
    p, t, r = np.deg2rad([pan, tilt, roll])
    f = np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), np.sin(t)])
    right0 = np.cross(f, [0, 0, 1.0]); right0 /= np.linalg.norm(right0)
    up0 = np.cross(right0, f)
    right = np.cos(r) * right0 + np.sin(r) * up0
    up = -np.sin(r) * right0 + np.cos(r) * up0
    return np.stack([right, up, -f], 1)


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Sim:
    """Owns every MuJoCo/EGL object; must only be touched from its own thread."""

    def __init__(self, robot="Panda", calib_dir="out/calib", view_res=256, main_res=(480, 640), fps=15):
        self.env = suite.make("Lift", robots=robot, has_renderer=False, has_offscreen_renderer=False,
                              use_camera_obs=False, ignore_done=True, hard_reset=False)
        self.env.reset()
        self.m, self.d = self.env.sim.model._model, self.env.sim.data._data
        r = self.env.robots[0]
        self.arm_q, self.arm_v = np.array(r._ref_joint_pos_indexes), np.array(r._ref_joint_vel_indexes)
        self.arm_act = np.array(r._ref_joint_actuator_indexes)
        self.grip_act = np.array(r._ref_joint_gripper_actuator_indexes)
        self.tau_lim = self.m.actuator_ctrlrange[self.arm_act]
        base_pos = r.robot_model.base_xpos_offset["table"](self.env.table_full_size[0])

        # twin: IK model + one mask renderer per calibrated camera (ESTIMATED calibration)
        self.kin = RobotKinematics(robot, base_pos=base_pos)
        self.ik = IKSolver(self.kin, rest_q=r.init_qpos)
        self.cams = {c: load_camera(f"{calib_dir}/{c}.npz").resized(view_res, view_res) for c in CAMS}
        self.twin = {c: KinematicMaskRenderer(robot, self.cams[c], base_pos=base_pos) for c in CAMS}
        self.robot_name, self.base_pos, self.init_q = robot, base_pos, np.array(r.init_qpos)
        self.cams_file = {c: load_camera(f"{calib_dir}/{c}.npz") for c in CAMS}   # at calibration resolution

        # real-world renderers
        self.view_res, self.main_res = view_res, main_res
        for attr in ("offwidth", "offheight"):
            setattr(self.m.vis.global_, attr, max(getattr(self.m.vis.global_, attr), 640))
        self.r_view = mujoco.Renderer(self.m, view_res, view_res)
        self.r_main = mujoco.Renderer(self.m, *main_res)
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[0] = 0; self.opt.sitegroup[:] = 0
        bname = [self.m.body(self.m.geom_bodyid[g]).name for g in range(self.m.ngeom)]
        self.is_robot = np.array([n.startswith(("robot0_", "gripper0_", "mount0_")) for n in bname] + [False])
        self.free_cam = mujoco.MjvCamera()
        self.free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.free_cam.lookat[:] = [-0.15, 0, 0.95]; self.free_cam.distance = 2.0
        self.free_cam.azimuth, self.free_cam.elevation = 150, -25

        self.kp, self.kd, self.vmax = 400.0, 40.0, 1.5
        self.overlay = dict(mask=True, occlusion=True, skeleton=True, target=True, iou=True)
        self.cmd = queue.Queue()
        self.latest, self.seq, self.fps = None, 0, fps
        self.main_drag, self.grab_px = None, 28

        # camera relocation state. T_* are camera poses in the robot base frame (OpenCV convention).
        #   T_est_calib : what the last calibration estimated
        #   T_gt_calib  : where the real camera actually was during that calibration (scoring + "known move")
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)   # cam_xpos is set by camlight
        self.cam_id = {c: self.m.camera(c).id for c in CAMS}
        self.cam_home = {c: (self.m.cam_pos[self.cam_id[c]].copy(), self.m.cam_quat[self.cam_id[c]].copy()) for c in MOVABLE}
        self.T_gt_calib = {c: self._gt_T(c) for c in MOVABLE}
        self.T_est_calib = {c: self.cams_file[c].T.copy() for c in MOVABLE}
        self.known_move = False
        self.calib_status = {c: "calibrated (initial)" for c in MOVABLE}
        self.calib_busy, self.recalib_queue = None, []
        self._reset()

    # ------------------------------------------------------------------ state
    def arm_qpos(self):
        return self.d.qpos[self.arm_q].copy()

    def _reset(self):
        self.env.reset()
        self.q_des = self.arm_qpos(); self.q_cmd = self.q_des.copy()
        self.kin.set_joints(dict(zip(self.ik.names, self.q_des)))
        self.p0, self.R0 = self.ik.site_pose()
        self.target_p, self.rpy = self.p0.copy(), np.zeros(3)
        self.gripper_open = True
        self.ik_err = (0.0, 0.0)

    def _solve(self):
        R_t = rpy_to_R(*self.rpy) @ self.R0
        q, ep, er = self.ik.solve(self.q_des, self.target_p, R_t)
        self.q_des, self.ik_err = q, (ep, er)

    # ------------------------------------------------------------------ commands
    def _ray(self, cam_name, u, v):
        cam = self.cams[cam_name]
        self.kin.set_joints(dict(zip(self.ik.names, self.arm_qpos())))
        T = self.kin.T_world_cam(cam)
        d = T[:3, :3] @ np.linalg.solve(cam.K, [u, v, 1.0])
        return T[:3, 3], d / np.linalg.norm(d), T

    def _drag(self, cam_name, u, v):
        o, d, T = self._ray(cam_name, u, v)
        if cam_name == "birdview":          # move in the horizontal plane through the target
            n = np.array([0, 0, 1.0])
        else:                               # move in the vertical plane facing the camera
            f = T[:3, 2].copy(); f[2] = 0
            n = f / (np.linalg.norm(f) + 1e-9)
        denom = d @ n
        if abs(denom) < 1e-6:
            return
        s = ((self.target_p - o) @ n) / denom
        if s <= 0:
            return
        p = o + s * d
        self.target_p = np.clip(p, [-0.6, -0.6, 0.81], [0.6, 0.6, 1.6])
        self._solve()

    # ---- main (free orbit) view: pinhole model of MuJoCo's free camera
    def _main_cam(self):
        fc = self.free_cam
        az, el = np.deg2rad(fc.azimuth), np.deg2rad(fc.elevation)
        fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        pos = np.array(fc.lookat) - fc.distance * fwd
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        H, W = self.main_res
        f = 0.5 * H / np.tan(np.deg2rad(self.m.vis.global_.fovy) / 2)
        K = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1.0]])
        return pos, np.stack([right, down, fwd], 1), K          # R columns = OpenCV cam axes

    def main_project(self, p):
        pos, R, K = self._main_cam()
        pc = R.T @ (p - pos)
        if pc[2] <= 1e-6:
            return None
        return (K @ pc)[:2] / pc[2]

    def _main_hit(self, u, v, n, anchor):
        pos, R, K = self._main_cam()
        d = R @ np.linalg.solve(K, [u, v, 1.0]); d /= np.linalg.norm(d)
        den = d @ n
        if abs(den) < 1e-6:
            return None
        s = ((anchor - pos) @ n) / den
        return pos + s * d if s > 0 else None

    def _main_plane(self, shift):
        return np.array([0, 0, 1.0]) if shift else self._main_cam()[1][:, 2]

    # ---- relocatable cameras
    def _gt_T(self, c):
        return gt_camera_from_sim(self.env.sim, c, self.view_res, self.view_res).T

    def _cam_world(self, c):
        i = self.cam_id[c]
        return self.d.cam_xpos[i].copy(), self.d.cam_xmat[i].reshape(3, 3).copy()   # MuJoCo cam axes

    def _set_cam_world(self, c, pos=None, R=None):
        """Physically move the REAL camera (world-fixed camera in the real-world sim)."""
        i = self.cam_id[c]
        if pos is not None:
            self.m.cam_pos[i] = pos
        if R is not None:
            q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten()); self.m.cam_quat[i] = q
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)   # cam_xpos is set by camlight
        self._update_twin_cam(c)

    def _update_twin_cam(self, c):
        """What the twin BELIEVES the camera pose is.
        known move  : the move is measured (e.g. encoder pan-tilt / indexed mount) -> apply the same
                      rigid motion to the calibrated estimate
        unknown move: keep the last calibration (stale until re-calibrated)"""
        T = self.T_est_calib[c]
        if self.known_move:
            T = self._gt_T(c) @ np.linalg.inv(self.T_gt_calib[c]) @ T
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
        n = self._main_plane(shift)
        if md["shift"] != shift:                      # (re)anchor so the handle never jumps
            h = self._main_hit(*md["last"], n, get_p())
            md["offset"], md["shift"] = (get_p() - h if h is not None else np.zeros(3)), shift
        h = self._main_hit(u, v, n, get_p())
        return None if h is None else h + md["offset"]

    def _main_pointer(self, c):
        t, u, v, shift = c["type"], float(c.get("u", 0)), float(c.get("v", 0)), bool(c.get("shift", False))
        md = self.main_drag
        if t == "main_down":
            k = self._pick(u, v)
            btn = int(c.get("button", 0))
            if k == "eef" and btn == 0:
                mode = "eef"
            elif k and k.startswith("cam:"):
                mode = "cam_move" if (btn == 0 or k[4:] == TOP_DOWN) else "cam_rot"
            else:
                mode = "orbit"
            self.main_drag = dict(mode=mode, cam=k[4:] if k and k.startswith("cam:") else None,
                                  last=(u, v), shift=None)
        elif t == "main_up":
            self.main_drag = None
        elif t == "main_move" and md is not None:
            du, dv = u - md["last"][0], v - md["last"][1]
            if md["mode"] == "orbit":
                fc = self.free_cam
                fc.azimuth -= 0.4 * du
                fc.elevation = float(np.clip(fc.elevation - 0.3 * dv, -89, 89))
            elif md["mode"] == "eef":
                p = self._plane_drag(md, u, v, shift, lambda: self.target_p)
                if p is not None:
                    self.target_p = np.clip(p, [-0.6, -0.6, 0.81], [0.6, 0.6, 1.6])
                    self._solve()
            elif md["mode"] == "cam_move":
                cam = md["cam"]
                p = self._plane_drag(md, u, v, shift or cam == TOP_DOWN, lambda: self._cam_world(cam)[0])
                if p is not None:
                    self._set_cam_world(cam, pos=np.clip(p, *CAM_BOX[cam]))
            elif md["mode"] == "cam_rot":            # pan / tilt (roll kept)
                cam = md["cam"]
                pan, tilt, roll = cam_angles(self._cam_world(cam)[1])
                self._set_cam_world(cam, R=cam_R(pan - 0.3 * du, np.clip(tilt - 0.3 * dv, -89, 30), roll))
            md["last"] = (u, v)

    def _wheel(self, c):
        k = self._pick(float(c.get("u", -1e9)), float(c.get("v", -1e9)))
        if k and k.startswith("cam:"):              # dolly the camera along its optical axis
            cam = k[4:]; p, R = self._cam_world(cam)
            self._set_cam_world(cam, pos=np.clip(p + (-R[:, 2]) * (-0.03 if c["dy"] > 0 else 0.03), *CAM_BOX[cam]))
        else:
            fc = self.free_cam
            fc.distance = float(np.clip(fc.distance * (1.08 if c["dy"] > 0 else 1 / 1.08), 0.5, 8.0))

    # ---- extrinsic re-calibration (same camera -> intrinsics fixed), runs in a worker thread
    def _moved_since_calib(self, c):
        et, er = pose_error(self._gt_T(c), self.T_gt_calib[c])
        return et > 1.0 or er > 0.1

    def _next_recalib(self):
        if not self.calib_busy and self.recalib_queue:
            self._start_recalib(self.recalib_queue.pop(0))

    def _start_recalib(self, c):
        if self.calib_busy or c not in MOVABLE:
            return
        xml = self.env.model.get_xml()
        i = self.cam_id[c]
        pos, quat, T_gt = self.m.cam_pos[i].copy(), self.m.cam_quat[i].copy(), self._gt_T(c)
        K, dist = self.cams_file[c].K, self.cams_file[c].dist
        H, W = self.cams_file[c].height, self.cams_file[c].width
        self.calib_busy, self.calib_status[c] = c, "re-calibrating… (robot runs checkerboard routine)"

        def job():
            try:
                scene = RealWorldCalibScene(xml, "hand", H, W)
                j = scene.model.camera(c).id
                scene.model.cam_pos[j], scene.model.cam_quat[j] = pos, quat   # the camera where it is NOW
                kin = RobotKinematics(self.robot_name, base_pos=self.base_pos)
                spread = np.array([0.5, 0.35, 0.5, 0.35, 0.7, 0.35, 0.9])
                cam, info = calibrate(scene, kin, c, "static", center=self.init_q, spread=spread,
                                      K_fixed=K, dist_fixed=dist, verbose=False, sampler="workspace",
                                      n_try=160, n_keep=25)
                self.cmd.put(dict(type="_calib_done", cam=c, T=cam.T, T_gt=T_gt,
                                  n=info["n_views"], rms=info["reproj_rms_px_joint"]))
            except Exception as e:                     # e.g. board not visible from the new pose
                self.cmd.put(dict(type="_calib_fail", cam=c, err=str(e)))

        threading.Thread(target=job, daemon=True).start()

    def handle(self, c):
        t = c.get("type")
        if t in ("main_down", "main_move", "main_up"):
            return self._main_pointer(c)
        if t == "wheel":
            return self._wheel(c)
        if t == "known_move":
            self.known_move = bool(c["value"])
            for cam in MOVABLE:
                self._update_twin_cam(cam)
        elif t == "cam_set" and c.get("cam") in MOVABLE:   # absolute pose from the panel sliders
            cam = c["cam"]
            pos = np.clip(np.array(c["pos"], float), *CAM_BOX[cam])
            R = None if cam == TOP_DOWN else cam_R(*c["ang"])
            self._set_cam_world(cam, pos=pos, R=R)
        elif t == "recalib":                          # "restart": re-estimate the moved camera(s), then re-segment
            cams = MOVABLE if c.get("cam", "all") == "all" else [c["cam"]]
            if c.get("only_moved", True) and c.get("cam", "all") == "all":
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
            self.calib_busy = None
            self._next_recalib()
        elif t == "_calib_fail":
            self.calib_status[c["cam"]] = "re-calibration failed: " + c["err"][:80]
            self.calib_busy = None
            self._next_recalib()
        elif t == "cam_home":                          # put the camera back and restore the original calibration
            cam = c["cam"]
            if cam in MOVABLE and not self.calib_busy:
                pos, quat = self.cam_home[cam]
                i = self.cam_id[cam]; self.m.cam_pos[i], self.m.cam_quat[i] = pos, quat
                mujoco.mj_kinematics(self.m, self.d); mujoco.mj_camlight(self.m, self.d)   # cam_xpos is set by camlight
                self.T_est_calib[cam] = self.cams_file[cam].T.copy()
                self.T_gt_calib[cam] = self._gt_T(cam)
                self._update_twin_cam(cam)
                self.calib_status[cam] = "calibrated (initial)"
        if t == "target":
            if "pos" in c:
                self.target_p = np.clip(np.array(c["pos"], float), [-0.6, -0.6, 0.81], [0.6, 0.6, 1.6])
            if "rpy_deg" in c:
                self.rpy = np.deg2rad(np.array(c["rpy_deg"], float))
            self._solve()
        elif t == "drag" and c.get("view") in self.cams:
            self._drag(c["view"], float(c["u"]), float(c["v"]))
        elif t == "gripper":
            self.gripper_open = bool(c["open"])
        elif t == "reset":
            self._reset()
        elif t == "home":
            self.target_p, self.rpy = self.p0.copy(), np.zeros(3); self._solve()
        elif t == "view_preset":
            fc = self.free_cam
            if c.get("name") == "overview":             # shows the robot AND both relocatable cameras
                fc.azimuth, fc.distance, fc.elevation = 150, 3.0, -5; fc.lookat[:] = [-0.1, 0, 2.0]
            else:
                fc.azimuth, fc.distance, fc.elevation = 150, 2.0, -25; fc.lookat[:] = [-0.15, 0, 0.95]
        elif t == "orbit":
            fc = self.free_cam
            fc.azimuth += c.get("daz", 0); fc.elevation = float(np.clip(fc.elevation + c.get("del", 0), -89, 89))
            fc.distance = float(np.clip(fc.distance * c.get("zoom", 1.0), 0.5, 8.0))
        elif t == "overlay":
            self.overlay.update({k: bool(v) for k, v in c.items() if k in self.overlay})

    # ------------------------------------------------------------------ physics
    def _physics_step(self):
        dt = self.m.opt.timestep
        step = np.clip(self.q_des - self.q_cmd, -self.vmax * dt, self.vmax * dt)
        self.q_cmd += step
        q, qd = self.d.qpos[self.arm_q], self.d.qvel[self.arm_v]
        M = np.zeros((self.m.nv, self.m.nv)); mujoco.mj_fullM(self.m, M, self.d.qM)
        Ma = M[np.ix_(self.arm_v, self.arm_v)]
        tau = Ma @ (self.kp * (self.q_cmd - q) - self.kd * qd) + self.d.qfrc_bias[self.arm_v]
        self.d.ctrl[self.arm_act] = np.clip(tau, self.tau_lim[:, 0], self.tau_lim[:, 1])
        lo, hi = self.m.actuator_ctrlrange[self.grip_act].T
        self.d.ctrl[self.grip_act] = np.where(self.gripper_open, np.where(hi > 0, hi, lo), 0.0)
        mujoco.mj_step(self.m, self.d)

    # ------------------------------------------------------------------ rendering
    def _amodal(self, cam):
        saved = self.m.geom_group.copy()
        self.m.geom_group[~self.is_robot[:-1]] = 5
        gid = fast_geom_segmentation(self.r_view, self.d, cam, self.opt)
        self.m.geom_group[:] = saved
        return self.is_robot[gid]

    def _marker(self, img, cam_name, p, color, kin):
        uv, ok = project(p[None], self.cams[cam_name], kin)
        if ok[0]:
            u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
            cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(img, (u, v), 7, color, 1, cv2.LINE_AA)

    def render(self):
        js = self.kin.joints_from_sim(self.env.sim)        # encoders only
        self.kin.set_joints(js)
        _, P, edges = skeleton_3d(self.kin)
        views, ious = {}, {}
        for c in CAMS:
            self.r_view.update_scene(self.d, camera=c, scene_option=self.opt)
            img = self.r_view.render().copy()
            tw = self.twin[c]
            mask = tw.render(js)                      # amodal: full silhouette from kinematics
            if self.overlay["occlusion"]:
                # RGB-D occlusion test: a twin pixel is visible if it is not behind the measured depth
                r = self.r_view
                r.enable_depth_rendering(); r.update_scene(self.d, camera=c, scene_option=self.opt)
                real_depth = r.render().copy(); r.disable_depth_rendering()
                vis = mask & (tw.render_depth() <= real_depth + 0.01)
            else:
                vis = mask
            if self.overlay["iou"]:
                g_amodal = self._amodal(c)
                g_vis = self.is_robot[fast_geom_segmentation(self.r_view, self.d, c, self.opt)]
                ious[c] = dict(amodal=_iou(g_amodal, mask), visible=_iou(g_vis, vis))
            if self.overlay["mask"]:
                img[vis] = (0.45 * img[vis] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
                cnts, _ = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnts, -1, (0, 255, 255), 1, cv2.LINE_AA)
                if self.overlay["occlusion"]:      # occluded part: thin outline only
                    hidden = mask & ~vis
                    cnts, _ = cv2.findContours(hidden.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(img, cnts, -1, (255, 120, 255), 1, cv2.LINE_AA)
            if self.overlay["skeleton"]:
                uv, ok = project(P, tw.camera, tw)
                img = draw_skeleton(img, uv, ok, edges, radius=3)
            if self.overlay["target"]:
                self._marker(img, c, self.target_p, (255, 60, 60), tw)
            views[c] = img
        # free orbit view with a translucent target sphere
        r = self.r_main
        r.update_scene(self.d, camera=self.free_cam, scene_option=self.opt)
        sc = r.scene
        if sc.ngeom < sc.maxgeom:
            active = bool(self.main_drag and self.main_drag["mode"] == "eef")
            rgba = np.array([1, 0.85, 0.1, 0.75] if active else [1, 0.2, 0.2, 0.6], np.float32)
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                                self.target_p, np.eye(3).flatten(), rgba)
            sc.ngeom += 1
        self._draw_cam_gizmos(sc)
        views["main"] = r.render().copy()
        return views, ious

    def _draw_cam_gizmos(self, sc):
        """Real camera bodies + view frusta in the simulator view (grab to move them)."""
        md = self.main_drag
        for c in MOVABLE:
            if sc.ngeom + 10 > sc.maxgeom:
                return
            p, R = self._cam_world(c)
            active = md is not None and md.get("cam") == c
            col = np.array((1, 0.85, 0.1, 1) if active else (*CAM_COLOR[c], 1), np.float32)
            g = sc.geoms[sc.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.03, 0.022, 0.025]), p, R.flatten(), col)
            sc.ngeom += 1
            t = np.tan(np.deg2rad(self.m.cam_fovy[self.cam_id[c]]) / 2)
            L = 0.35
            corners = [p + R @ (np.array([sx * t, sy * t, -1.0]) * L) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
            segs = [(p, q) for q in corners] + [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
            for a, b in segs:
                g = sc.geoms[sc.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), col)
                mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, *a, *b)
                sc.ngeom += 1

    def _cam_state(self):
        out = {}
        for c in MOVABLE:
            T_gt = self._gt_T(c)
            et, er = pose_error(self.cams[c].T, T_gt)           # twin belief vs real camera
            mt, mr = pose_error(T_gt, self.T_gt_calib[c])       # how far the camera moved since calibration
            p, R = self._cam_world(c)
            out[c] = dict(err_mm=et, err_deg=er, moved_mm=mt, moved_deg=mr, status=self.calib_status[c],
                          pos=p.round(3).tolist(), ang=cam_angles(R).round(1).tolist(),
                          busy=self.calib_busy == c)
        return out

    # ------------------------------------------------------------------ main loop
    def run(self, stop: threading.Event):
        dt = self.m.opt.timestep
        t_sim, t_wall0 = 0.0, time.perf_counter()
        next_frame, frame_times = 0.0, []
        while not stop.is_set():
            while True:
                try:
                    self.handle(self.cmd.get_nowait())
                except queue.Empty:
                    break
            now = time.perf_counter() - t_wall0
            n = 0
            while t_sim < now and n < 100:            # keep physics in real time (cap catch-up)
                self._physics_step(); t_sim += dt; n += 1
            if t_sim < now - 0.2:
                t_sim = now                           # we fell behind: drop time instead of spiralling
            if now >= next_frame:
                t0 = time.perf_counter()
                views, ious = self.render()
                frame_times.append(time.perf_counter() - t0); frame_times = frame_times[-30:]
                enc = {k: base64.b64encode(cv2.imencode(".jpg", v[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
                       for k, v in views.items()}
                p = self.d.site_xpos[self.m.site("gripper0_grip_site").id].copy()   # real EEF
                main_uv = self.main_project(self.target_p)
                self.seq += 1
                self.latest = json.dumps(dict(
                    seq=self.seq, views=enc, iou=ious,
                    state=dict(target=self.target_p.round(4).tolist(), rpy_deg=np.rad2deg(self.rpy).round(1).tolist(),
                               eef=p.round(4).tolist(), track_err_mm=float(np.linalg.norm(p - self.target_p) * 1000),
                               ik_err_mm=self.ik_err[0] * 1000, ik_err_deg=float(np.rad2deg(self.ik_err[1])),
                               q=self.arm_qpos().round(3).tolist(), gripper_open=self.gripper_open,
                               render_ms=float(np.mean(frame_times) * 1000), overlay=self.overlay,
                               main_uv=None if main_uv is None else main_uv.round(1).tolist(),
                               main_res=list(self.main_res), grab_px=self.grab_px,
                               dragging=bool(self.main_drag and self.main_drag["mode"] == "eef"),
                               handles={k: (None if (uv := self.main_project(v)) is None else uv.round(1).tolist())
                                        for k, v in self._handles().items()},
                               cams=self._cam_state(), known_move=self.known_move,
                               recalib_busy=self.calib_busy, recalib_queue=list(self.recalib_queue))))
                next_frame = now + 1.0 / self.fps
            else:
                time.sleep(0.001)


def make_app(sim: Sim):
    app = FastAPI()
    here = os.path.dirname(os.path.abspath(__file__))

    @app.get("/")
    def index():
        return FileResponse(os.path.join(here, "index_robosuite.html"))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()

        async def sender():
            last = -1
            while True:
                if sim.latest is not None and sim.seq != last:
                    last = sim.seq
                    await sock.send_text(sim.latest)
                await asyncio.sleep(0.01)

        task = asyncio.create_task(sender())
        try:
            while True:
                sim.cmd.put(json.loads(await sock.receive_text()))
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--calib", default=os.path.join(ROOT, "out/calib"))
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    ready, holder, stop = threading.Event(), {}, threading.Event()

    def worker():
        holder["sim"] = Sim(calib_dir=args.calib, fps=args.fps)
        ready.set()
        holder["sim"].run(stop)

    th = threading.Thread(target=worker, daemon=True); th.start()
    ready.wait()
    print(f"ready: http://localhost:{args.port}", flush=True)
    try:
        uvicorn.run(make_app(holder["sim"]), host=args.host, port=args.port, log_level="warning")
    finally:
        stop.set()


if __name__ == "__main__":
    main()
