"""Damped-least-squares IK on the robot-only kinematic model (the twin)."""
from __future__ import annotations

import mujoco
import numpy as np

from .kinematic_mask import RobotKinematics


def rot_err(R_target, R):
    """World-frame rotation vector taking R to R_target."""
    q = np.zeros(4); mujoco.mju_mat2Quat(q, (R_target @ R.T).flatten())
    v = np.zeros(3); mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class IKSolver:
    def __init__(self, kin: RobotKinematics, site="gripper0_grip_site", arm_prefix="robot0_joint",
                 rest_q=None, rot_weight=0.5, damping=0.05, null_gain=0.05, joint_names=None):
        self.kin, m = kin, kin.model
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        if joint_names is not None:
            self.jids = [m.joint(n).id for n in joint_names]
        else:
            self.jids = [j for j in range(m.njnt) if m.joint(j).name.startswith(arm_prefix)]
        self.names = [m.joint(j).name for j in self.jids]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jids])
        lim = m.jnt_limited[self.jids].astype(bool)
        self.lo = np.where(lim, m.jnt_range[self.jids, 0], -np.inf)   # continuous joints: no clamp
        self.hi = np.where(lim, m.jnt_range[self.jids, 1], np.inf)
        self.rest = np.zeros(len(self.jids)) if rest_q is None else np.asarray(rest_q)
        self.w, self.lam, self.kn = rot_weight, damping, null_gain

    def site_pose(self):
        d = self.kin.data
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, q_init, p_target, R_target, iters=100, tol_p=2e-4, tol_r=2e-3, max_step=0.15, relax=True):
        """Position has priority: if the requested orientation makes the position unreachable
        (joint limits / <6 DOF), retry with a progressively weaker orientation weight."""
        w0 = self.w
        best = None
        for w in ([w0, w0 * 0.2, 0.02] if relax else [w0]):
            self.w = w
            q, ep, er = self._solve(q_init, p_target, R_target, iters, tol_p, tol_r, max_step)
            if best is None or ep < best[1] - 1e-4:
                best = (q, ep, er)
            if ep < max(5 * tol_p, 2e-3):
                break
        self.w = w0
        q, ep, er = best
        d = self.kin.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.kin.model, d)
        return q, ep, er

    def _solve(self, q_init, p_target, R_target, iters, tol_p, tol_r, max_step):
        m, d = self.kin.model, self.kin.data
        q = np.clip(np.asarray(q_init, float).copy(), self.lo, self.hi)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for it in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            p, R = self.site_pose()
            ep, er = p_target - p, rot_err(R_target, R)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], self.w * jacr[:, self.dadr]])
            e = np.r_[ep, self.w * er]
            JJt = J @ J.T + self.lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(len(q)) - J.T @ np.linalg.solve(JJt, J)        # null-space projector
            dq += N @ (self.kn * (self.rest - q))
            n = np.abs(dq).max()
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, self.lo, self.hi)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        p, R = self.site_pose()
        return q, float(np.linalg.norm(p_target - p)), float(np.linalg.norm(rot_err(R_target, R)))
