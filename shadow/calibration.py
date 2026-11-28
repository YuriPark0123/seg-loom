"""Camera calibration performed *inside* the "real world" simulator, the same way it is
done on a real robot:

  * intrinsics          : cv2.calibrateCamera from checkerboard views
  * static camera (eye-to-hand): checkerboard bolted to the robot hand, robot moved to N poses,
                          cv2.calibrateHandEye -> T_base_cam
  * wrist camera (eye-in-hand): checkerboard fixed on the table, robot moved to N poses,
                          cv2.calibrateHandEye -> T_hand_cam

The only things taken from the "real world" are camera images and joint encoder
readings. Hand poses come from forward kinematics of our own robot model
(RobotKinematics), never from the simulator's ground truth.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from .kinematic_mask import Camera, RobotKinematics, apply_joint_couplings, make_T


def checkerboard_body(name, pos, quat, cols=8, rows=6, square=0.02, margin=0.015, thick=0.003):
    """Checkerboard built from geoms (white plate + black squares). Normal = body +z."""
    body = ET.Element("body", name=name, pos=" ".join(map(str, pos)), quat=" ".join(map(str, quat)))
    W, H = cols * square + 2 * margin, rows * square + 2 * margin
    ET.SubElement(body, "geom", name=f"{name}_plate", type="box", size=f"{W/2} {H/2} {thick/2}",
                  rgba="1 1 1 1", contype="0", conaffinity="0", group="1")
    for i in range(cols):
        for j in range(rows):
            if (i + j) % 2 == 0:
                cx = (i + 0.5) * square - cols * square / 2
                cy = (j + 0.5) * square - rows * square / 2
                ET.SubElement(body, "geom", name=f"{name}_sq{i}_{j}", type="box",
                              size=f"{square/2} {square/2} 0.0005", pos=f"{cx} {cy} {thick/2 + 0.0005}",
                              rgba="0 0 0 1", contype="0", conaffinity="0", group="1")
    return body


def board_object_points(cols=8, rows=6, square=0.02):
    """Inner-corner coordinates in the board frame (z=0 on the top face, offset ignored)."""
    xs = (np.arange(1, cols) - cols / 2) * square
    ys = (np.arange(1, rows) - rows / 2) * square
    return np.array([[x, y, 0.0] for y in ys for x in xs], np.float32)


class RealWorldCalibScene:
    """The 'real world' sim scene (task env xml) + an added calibration board.

    Mimics a real calibration session: we can command joints and grab camera images.
    """

    def __init__(self, env_xml: str, mode: str, height=512, width=512, board=None):
        self.mode = mode
        # odd x even squares -> the board has no 180-degree symmetry, so corner order is unambiguous
        self.board = board or (dict(cols=9, rows=6, square=0.025) if mode == "hand" else dict(cols=9, rows=6, square=0.02))
        root = ET.fromstring(env_xml)
        wb = root.find("worldbody")
        b = self.board
        self.board_thick = 0.003
        if mode == "hand":   # board on a bracket at the hand, facing away from the flange
            hand = next(e for e in root.iter("body") if e.get("name") == "gripper0_eef")
            hand.append(checkerboard_body("calib_board", (0.0, 0.11, -0.06), (0, 1, 0, 0), **b))
        else:                # board flat on the table
            wb.append(checkerboard_body("calib_board", (0.0, 0.0, 0.802), (1, 0, 0, 0), **b))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0   # hide collision geoms, like robosuite's own renderer
        self.opt.sitegroup[:] = 0
        self.h, self.w = height, width
        self._jadr = {self.model.joint(j).name: self.model.jnt_qposadr[j] for j in range(self.model.njnt)}
        # move task objects out of the way
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                a = self.model.jnt_qposadr[j]
                self.data.qpos[a:a + 3] = [5.0, 5.0, 0.0]

    def set_joints(self, js: dict):
        for n, v in js.items():
            if n in self._jadr:
                self.data.qpos[self._jadr[n]] = v
        apply_joint_couplings(self.model, self.data, js)
        mujoco.mj_forward(self.model, self.data)

    def image(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name, scene_option=self.opt)
        return self.renderer.render()

    def encoders(self, joint_names):
        return {n: float(self.data.qpos[self._jadr[n]]) for n in joint_names if n in self._jadr}


def detect_corners(img, cols=9, rows=6):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    pat = (cols - 1, rows - 1)
    ok, c = cv2.findChessboardCornersSB(gray, pat, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return None
    return c.reshape(-1, 1, 2).astype(np.float32)


def _sample_joint_poses(kin: RobotKinematics, n, rng, center, spread, arm_joints=None):
    arm = arm_joints or [j for j in kin.joint_names if j.startswith("robot0_")]
    out = []
    for _ in range(n):
        out.append({j: c + rng.uniform(-s, s) for j, c, s in zip(arm, center, spread)})
    return out


def _sample_workspace_poses(kin: RobotKinematics, n, rng, q_rest, box=((-0.35, 0.25), (-0.35, 0.35), (0.92, 1.35)),
                            tilt=0.45, yaw=0.9, site="gripper0_grip_site", arm_joints=None, rot_weight=0.5,
                            pos_tol=0.01, rot_tol=0.05, face_point=None):
    """Random board poses over the workspace (board position + tilt), turned into joints by IK.
    For <6-DOF arms (small rot_weight) the orientation is only approximate and is not checked."""
    from .ik import IKSolver
    import cv2 as _cv2
    ik = IKSolver(kin, site=site, rest_q=q_rest, joint_names=arm_joints, rot_weight=rot_weight)
    arm = ik.names
    kin.set_joints(dict(zip(arm, q_rest)))
    _, R0 = ik.site_pose()
    out = []
    for _ in range(n * 3):
        p = np.array([rng.uniform(*box[0]), rng.uniform(*box[1]), rng.uniform(*box[2])])   # world frame
        if face_point is not None and rng.random() < 0.6:
            # board normal (= -approach = -z of the TCP frame) toward a rough guess of the camera position
            nrm = np.asarray(face_point) - p; nrm /= np.linalg.norm(nrm)
            nrm = _cv2.Rodrigues(rng.normal(size=3) * 0.25)[0] @ nrm
            z = -nrm; x = np.cross([0, 0, 1.0], z)
            x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
            R = np.stack([x, np.cross(z, x), z], 1) @ _cv2.Rodrigues(np.array([0, 0, rng.uniform(-np.pi, np.pi)]))[0]
        else:
            R = (_cv2.Rodrigues(np.array([0, 0, rng.uniform(-yaw, yaw)]))[0]
                 @ _cv2.Rodrigues(np.array([rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt), 0]))[0] @ R0)
        q, ep, er = ik.solve(q_rest, p, R, iters=80)
        if ep < pos_tol and (er < rot_tol or rot_weight < 0.2):
            out.append(dict(zip(arm, q)))
        if len(out) >= n:
            break
    return out


def calibrate(scene: RealWorldCalibScene, kin: RobotKinematics, cam_name, mode, *, n_try=150, n_keep=30,
              seed=0, center=None, spread=None, hand_body="robot0_right_hand", verbose=True,
              K_fixed=None, dist_fixed=None, sampler="joint", arm_joints=None, tcp_site="gripper0_grip_site",
              ws_box=None, rot_weight=0.5, pos_tol=0.01, face_point=None):
    """Return (Camera estimate, info dict). mode='static' (eye-to-hand) or 'wrist' (eye-in-hand).

    K_fixed: re-calibrate EXTRINSICS ONLY with known intrinsics (same physical camera, moved).
    """
    rng = np.random.default_rng(seed)
    b = scene.board
    objp = board_object_points(**b)  # z=0 plane; target-frame offset cancels in hand-eye
    obj_pts, img_pts, T_base_hand, used_imgs, hits = [], [], [], [], []

    def try_pose(js):
        scene.set_joints(js)
        img = scene.image(cam_name)
        c = detect_corners(img, b["cols"], b["rows"])
        if c is None:
            return False
        enc = scene.encoders(kin.joint_names)       # proprioception
        kin.set_joints(enc)
        obj_pts.append(objp); img_pts.append(c); used_imgs.append(img)
        T_base_hand.append(kin.T_base(hand_body))   # FK of OUR model, not sim GT
        hits.append(js)
        return True

    if sampler == "joint":
        for js in _sample_joint_poses(kin, n_try, rng, center, spread, arm_joints):
            try_pose(js)
            if len(obj_pts) >= n_keep:
                break
    else:
        # camera pose unknown -> sweep the board over the whole reachable workspace (via IK),
        # then densify around the poses where the board was actually seen
        kw = dict(site=tcp_site, arm_joints=arm_joints, rot_weight=rot_weight, pos_tol=pos_tol, face_point=face_point)
        if ws_box is not None:
            kw["box"] = ws_box
        for js in _sample_workspace_poses(kin, n_try, rng, center, **kw):
            try_pose(js)
            if len(obj_pts) >= n_keep:
                break
        arm = arm_joints or [j for j in kin.joint_names if j.startswith("robot0_")]
        budget = n_try
        while hits and len(obj_pts) < n_keep and budget > 0:
            base = hits[rng.integers(len(hits))]
            try_pose({j: base[j] + rng.normal() * 0.08 for j in arm}); budget -= 1
    if len(obj_pts) < 8:
        raise RuntimeError(f"only {len(obj_pts)} board detections for {cam_name}")

    h, w = scene.h, scene.w
    if K_fixed is None:
        flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None, flags=flags)
    else:
        K = np.asarray(K_fixed, np.float64)
        dist = np.zeros(5) if dist_fixed is None else np.asarray(dist_fixed, np.float64).reshape(-1)
        # per-view board poses with the intrinsics held fixed; uses the same homography-based init as the
        # full calibration (plain IPPE PnP can pick the flipped solution for small, far-away boards)
        fix = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT |
               cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
               cv2.CALIB_FIX_K3)
        rms, _, _, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), K.copy(), dist.copy(), flags=fix)

    R_t2c = [cv2.Rodrigues(r)[0] for r in rvecs]
    t_t2c = [t.reshape(3) for t in tvecs]
    if mode == "static":   # eye-to-hand: feed base->hand, get cam->base
        Tin = [np.linalg.inv(T) for T in T_base_hand]
    else:                  # eye-in-hand: feed hand->base, get cam->hand
        Tin = T_base_hand
    R_g, t_g = [T[:3, :3] for T in Tin], [T[:3, 3] for T in Tin]
    T_cam_board = [make_T(R, t) for R, t in zip(R_t2c, t_t2c)]
    best = None
    for meth in (cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_HORAUD,
                 cv2.CALIB_HAND_EYE_DANIILIDIS):
        try:
            R_x, t_x = cv2.calibrateHandEye(R_g, t_g, R_t2c, t_t2c, method=meth)
        except cv2.error:
            continue
        X0_ = make_T(R_x, t_x.reshape(3))   # T_base_cam (static) or T_hand_cam (wrist)
        res = refine_robot_camera(obj_pts, img_pts, T_base_hand, K, dist.reshape(-1)[:2],
                                  X0_, T_cam_board, mode, fix_intrinsics=K_fixed is not None)
        if best is None or res[4] < best[1][4]:
            best = (X0_, res, dict(refine_robot_camera.last_info))
        if res[4] < 0.5:
            break
    X0, (K1, dist1, X1, Y1, rms_ba), refine_robot_camera.last_info = best
    cam = Camera(K1, X1, h, w, parent_body=None if mode == "static" else hand_body,
                 dist=np.r_[dist1, 0, 0, 0])
    info = dict(**refine_robot_camera.last_info, n_views=len(obj_pts), reproj_rms_px=float(rms), reproj_rms_px_joint=float(rms_ba),
                sample_image=used_imgs[0], sample_corners=img_pts[0], handeye_only=Camera(K, X0, h, w,
                parent_body=None if mode == "static" else hand_body, dist=dist.reshape(-1)))
    if verbose:
        print(f"[calib {cam_name}] views={len(obj_pts)} intrinsic-only RMS={rms:.3f}px -> joint robot-camera "
              f"RMS={rms_ba:.3f}px (dropped {info['views_dropped']} views)  fx={K1[0,0]:.1f} fy={K1[1,1]:.1f} cx={K1[0,2]:.1f} cy={K1[1,2]:.1f}")
    return cam, info


def _vec(T):
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def _mat(v):
    return make_T(cv2.Rodrigues(v[:3])[0], v[3:6])


def refine_robot_camera(obj_pts, img_pts, T_base_hand, K, dist2, X0, T_cam_board, mode, fix_intrinsics=False):
    """Jointly refine intrinsics + hand-eye X + board pose Y by minimizing corner reprojection
    error through the robot's forward kinematics (standard robot-camera bundle adjustment).

    static: p_cam = X^-1 * T_base_hand_i * Y * p      (X = T_base_cam,  Y = T_hand_board)
    wrist : p_cam = X^-1 * T_base_hand_i^-1 * Y * p   (X = T_hand_cam,  Y = T_base_board)
    """
    from scipy.optimize import least_squares

    if mode == "static":
        Ys = [np.linalg.inv(Th) @ X0 @ Tcb for Th, Tcb in zip(T_base_hand, T_cam_board)]
    else:
        Ys = [Th @ X0 @ Tcb for Th, Tcb in zip(T_base_hand, T_cam_board)]
    Y0 = Ys[int(np.argmin([np.linalg.norm(Y[:3, 3] - np.median([y[:3, 3] for y in Ys], 0)) for Y in Ys]))]
    P = np.asarray(obj_pts[0], np.float64)
    uv = np.concatenate([c.reshape(-1, 2) for c in img_pts])

    k_fix = np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist2]

    def unpack(x):
        if fix_intrinsics:
            x = np.r_[k_fix, x]
        Kx = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1]])
        return Kx, np.r_[x[4:6], 0, 0, 0], _mat(x[6:12]), _mat(x[12:18])

    def resid(x):
        Kx, dx, X, Y = unpack(x)
        Xi = np.linalg.inv(X)
        out = []
        for Th in T_base_hand:
            T = Xi @ Th @ Y if mode == "static" else Xi @ np.linalg.inv(Th) @ Y
            pr, _ = cv2.projectPoints(P, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], Kx, dx)
            out.append(pr.reshape(-1, 2))
        return (np.concatenate(out) - uv).ravel()

    x0 = np.r_[_vec(X0), _vec(Y0)] if fix_intrinsics else np.r_[k_fix, _vec(X0), _vec(Y0)]
    n = len(P)
    keep = np.ones(len(T_base_hand), bool)
    for it in range(3):   # robust: re-solve after dropping views with gross residuals
        sel = np.repeat(keep, n)
        f = lambda x: resid(x).reshape(-1, 2)[np.concatenate([sel])].ravel()
        sol = least_squares(f, x0, loss="huber", f_scale=1.0, x_scale="jac", max_nfev=2000)
        x0 = sol.x
        per_view = np.sqrt((resid(sol.x).reshape(len(keep), n, 2) ** 2).sum(-1).mean(-1))
        new_keep = per_view < max(5 * np.median(per_view[keep]), 1.0)
        if (new_keep == keep).all():
            break
        keep = new_keep
    Kx, dx, X, Y = unpack(sol.x)
    r = sol.fun.reshape(-1, 2)
    refine_robot_camera.last_info = dict(views_used=int(keep.sum()), views_dropped=int((~keep).sum()))
    return Kx, dx[:2], X, Y, float(np.sqrt((r ** 2).sum(1).mean()))


def pose_error(T_est, T_gt):
    dT = np.linalg.inv(T_gt) @ T_est
    ang = np.rad2deg(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000), float(ang)
