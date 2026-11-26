"""Step 1: calibrate cameras in the "real world" sim, build a kinematic digital twin from the
estimated calibration + joint encoders, and measure how well the twin's projection of the robot
(mask + skeleton) overlaps the real camera images.

"real world" = robosuite env (its ground-truth camera poses are used ONLY for scoring)
"twin"       = robot-only MuJoCo model, placed with the ESTIMATED calibration, driven by encoders
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import h5py
import imageio
import numpy as np
import mujoco
import robosuite as suite

from shadow.calibration import RealWorldCalibScene, calibrate, pose_error
from shadow.kinematic_mask import KinematicMaskRenderer, gt_camera_from_sim
from shadow.skeleton import draw_skeleton, project, skeleton_3d


CALIB_MODE = {"birdview": "static", "agentview": "static", "frontview": "static", "sideview": "static",
              "robot0_eye_in_hand": "wrist"}
CAM_LABEL = {"birdview": "top view (birdview)", "agentview": "front-top (agentview)",
             "robot0_eye_in_hand": "wrist cam (eye-in-hand)"}


def iou(a, b, ignore=None):
    if ignore is not None:
        a, b = a & ~ignore, b & ~ignore
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def diff_vis(gt, pred):
    v = np.zeros(gt.shape + (3,), np.uint8)
    v[gt & pred] = (60, 200, 60)      # both
    v[gt & ~pred] = (230, 50, 50)     # real only (twin missed)
    v[~gt & pred] = (60, 110, 255)    # twin only (false)
    return v


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class RealWorld:
    """The 'real world': robosuite env state + camera images + GT segmentation (for scoring only).

    Images are rendered with mujoco.Renderer (one EGL stack in the process), hiding collision
    geoms (group 0) exactly like robosuite's own camera observations.
    """

    def __init__(self, task, robot, cams, res_list):
        self.env = suite.make(task, robots=robot, has_renderer=False, has_offscreen_renderer=False,
                              use_camera_obs=False, ignore_done=True, hard_reset=False)
        self.env.reset()
        self.cams, self.res_list = cams, res_list
        self._renderers = {}
        self._build_lookup()

    def _build_lookup(self):
        m = self.env.sim.model._model
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[0] = 0
        self.opt.sitegroup[:] = 0  # robosuite sites are invisible (alpha 0) in RGB but still occupy seg pixels
        bname = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        self.is_robot = np.array([n.startswith(("robot0_", "gripper0_", "mount0_")) for n in bname] + [False])
        free_roots = {m.body_rootid[m.jnt_bodyid[j]] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        self.is_obj = np.array([m.body_rootid[m.geom_bodyid[g]] in free_roots for g in range(m.ngeom)] + [False])
        for r in self._renderers.values():
            getattr(r, 'close', lambda: None)()
        self._renderers = {res: mujoco.Renderer(m, res, res) for res in self.res_list}

    def load_state(self, model_xml, state):
        self.env.reset_from_xml_string(self.env.edit_model_xml(model_xml))
        self.env.sim.reset()
        self._build_lookup()
        self.set_state(state)

    def set_state(self, state):
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

    def observe(self, res):
        r, d = self._renderers[res], self.env.sim.data._data
        out = {}
        for c in self.cams:
            r.update_scene(d, camera=c, scene_option=self.opt)
            rgb = r.render()
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=c, scene_option=self.opt)
            seg = r.render(); r.disable_segmentation_rendering()
            gid = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
            out[c] = dict(rgb=rgb, robot=self.is_robot[gid], other=self.is_obj[gid],
                          robot_amodal=self._amodal(r, d, c))
        return out

    def _amodal(self, r, d, cam):
        """GT robot mask with every non-robot geom hidden (scoring only): the full silhouette,
        including parts occluded by the table/objects. This is what a kinematic render predicts."""
        m = self.env.sim.model._model
        saved = m.geom_group.copy()
        m.geom_group[~self.is_robot[:-1]] = 5
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam, scene_option=self.opt)
        seg = r.render(); r.disable_segmentation_rendering()
        m.geom_group[:] = saved
        gid = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
        return self.is_robot[gid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Lift")
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--dataset", default="data/robomimic/lift_ph_low_dim_v141.hdf5")
    ap.add_argument("--cams", nargs="+", default=["birdview", "agentview", "robot0_eye_in_hand"])
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--policy_res", type=int, default=84)
    ap.add_argument("--calib_res", type=int, default=512)
    ap.add_argument("--n_demos", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="out/step1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rw = RealWorld(args.task, args.robot, args.cams, [args.res, args.policy_res])
    env = rw.env
    base_pos = env.robots[0].robot_model.base_xpos_offset["table"](env.table_full_size[0])

    # ------------------------------------------------------------ 1. calibration
    q0 = np.array(env.robots[0].init_qpos)
    spread = np.array([0.5, 0.35, 0.5, 0.35, 0.7, 0.35, 0.9])
    from shadow.kinematic_mask import RobotKinematics
    kin = RobotKinematics(args.robot, base_pos=base_pos)
    real_xml = env.model.get_xml()  # NOTE: sim.model.get_xml() returns the LAST loaded xml
    cams_est, report = {}, {}
    for c in args.cams:
        mode = CALIB_MODE[c]
        scene = RealWorldCalibScene(real_xml, "hand" if mode == "static" else "table",
                                    args.calib_res, args.calib_res)
        t0 = time.time()
        cam, info = calibrate(scene, kin, c, mode, center=q0, spread=spread, seed=args.seed)
        gt = gt_camera_from_sim(env.sim, c, args.calib_res, args.calib_res)
        et, er = pose_error(cam.T, gt.T)
        dK = np.abs(cam.K - gt.K)[[0, 1, 0, 1], [0, 1, 2, 2]]
        et0, er0 = pose_error(info["handeye_only"].T, gt.T)
        print(f"[{c}] hand-eye only (before joint refinement): {et0:.2f} mm, {er0:.3f} deg")
        report[c] = dict(mode=mode, n_views=info["n_views"], reproj_rms_px=info["reproj_rms_px"],
                         reproj_rms_px_joint=info["reproj_rms_px_joint"],
                         handeye_only_trans_err_mm=et0, handeye_only_rot_err_deg=er0,
                         trans_err_mm=et, rot_err_deg=er,
                         K_err_px_at_calib_res=dict(fx=dK[0], fy=dK[1], cx=dK[2], cy=dK[3]),
                         time_s=time.time() - t0)
        print(f"[{c}] pose error: {et:.2f} mm, {er:.3f} deg  | |dK| fx {dK[0]:.2f} fy {dK[1]:.2f} "
              f"cx {dK[2]:.2f} cy {dK[3]:.2f} px")
        img = info["sample_image"].copy()
        cv2.drawChessboardCorners(img, (8, 5), info["sample_corners"], True)
        imageio.imwrite(f"{args.out}/calib_view_{c}.png", img)
        cams_est[c] = cam
        del scene

    # ------------------------------------------------------------ 2. digital twins
    tw = {}  # (cam, res, est|gt) -> renderer
    for c in args.cams:
        for r in (args.res, args.policy_res):
            tw[(c, r, "est")] = KinematicMaskRenderer(args.robot, cams_est[c].resized(r, r), base_pos=base_pos)
            tw[(c, r, "gt")] = KinematicMaskRenderer(args.robot, gt_camera_from_sim(env.sim, c, r, r),
                                                     base_pos=base_pos)

    # ------------------------------------------------------------ 3. replay demos & compare
    f = h5py.File(args.dataset, "r")
    demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[: args.n_demos]
    stats = {c: {k: [] for k in ["iou_est", "iou_gt", "iou_est_modal", "iou_est_lo", "iou_gt_lo",
                                 "kp_err_px", "kp_err_px_lo"]} for c in args.cams}
    fk_err = []
    writer = imageio.get_writer(f"{args.out}/overlay_demo0.mp4", fps=10, macro_block_size=1)
    t_render = []
    for di, dk in enumerate(demos):
        states = f[f"data/{dk}/states"][()]
        xml = f[f"data/{dk}"].attrs["model_file"]
        rw.load_state(xml, states[0])
        for t in range(0, len(states), args.stride):
            rw.set_state(states[t])
            obs_hi, obs_lo = rw.observe(args.res), rw.observe(args.policy_res)
            js = kin.joints_from_sim(env.sim)          # encoders only
            kin.set_joints(js)
            fk_err.append(np.linalg.norm(kin.body_T("robot0_right_hand")[:3, 3]
                                         - env.sim.data.get_body_xpos("robot0_right_hand")))
            names, P, edges = skeleton_3d(kin)
            rows = []
            for c in args.cams:
                t0 = time.time()
                m_est = tw[(c, args.res, "est")].render(js)
                t_render.append(time.time() - t0)
                m_gt = tw[(c, args.res, "gt")].render(js)
                m_est_lo = tw[(c, args.policy_res, "est")].render(js)
                m_gt_lo = tw[(c, args.policy_res, "gt")].render(js)
                g, g_lo = obs_hi[c], obs_lo[c]
                S = stats[c]
                # amodal GT = full robot silhouette (what calibration+kinematics should reproduce)
                S["iou_est"].append(iou(g["robot_amodal"], m_est)); S["iou_gt"].append(iou(g["robot_amodal"], m_gt))
                S["iou_est_lo"].append(iou(g_lo["robot_amodal"], m_est_lo))
                S["iou_gt_lo"].append(iou(g_lo["robot_amodal"], m_gt_lo))
                # modal GT = visible robot pixels; the twin does not model table/object occlusion
                S["iou_est_modal"].append(iou(g["robot"], m_est))
                ce, cg = tw[(c, args.res, "est")], tw[(c, args.res, "gt")]
                uv_e, v_e = project(P, ce.camera, ce); uv_g, v_g = project(P, cg.camera, cg)
                ok = v_e & v_g & (uv_g[:, 0] > 0) & (uv_g[:, 0] < args.res) & (uv_g[:, 1] > 0) & (uv_g[:, 1] < args.res)
                if ok.any():
                    e = np.linalg.norm(uv_e[ok] - uv_g[ok], axis=1)
                    S["kp_err_px"].append(float(e.mean()))
                    S["kp_err_px_lo"].append(float(e.mean() * args.policy_res / args.res))
                if di == 0:
                    twin_rgb = ce.render_rgb(js)
                    blend = (0.5 * g["rgb"] + 0.5 * twin_rgb).astype(np.uint8)
                    dv = diff_vis(g["robot_amodal"], m_est)
                    dv = draw_skeleton(dv, uv_e, v_e, edges, radius=3)
                    real_sk = draw_skeleton(g["rgb"], uv_e, v_e, edges, radius=3)
                    rows.append(np.concatenate([
                        label(g["rgb"], f"real: {CAM_LABEL.get(c, c)}"),
                        label(twin_rgb, "twin (estimated calib)"),
                        label(blend, "overlay 50/50"),
                        label(dv, f"IoU {S['iou_est'][-1]:.3f}  G=both R=real B=twin"),
                        label(real_sk, "skeleton on real image")], 1))
            if di == 0:
                fr = np.concatenate(rows, 0)
                writer.append_data(fr)
                if t == (len(states) // 2) // args.stride * args.stride:
                    imageio.imwrite(f"{args.out}/overlay_demo0_mid.png", fr)
        print(f"demo {dk}: " + " | ".join(f"{c[:9]} IoU {np.mean(stats[c]['iou_est']):.3f}" for c in args.cams))
    writer.close()

    summary = {"calibration": report, "fk_err_mm_max": float(np.max(fk_err) * 1000),
               "render_ms_mean": float(np.mean(t_render) * 1000), "overlap": {}}
    for c in args.cams:
        summary["overlap"][c] = {k: (float(np.mean(v)), float(np.min(v)) if v else None) for k, v in stats[c].items()}
    print(json.dumps(summary, indent=1, default=float))
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1, default=float)

    # ------------------------------------------------------------ 4. sensitivity sweep
    sweep_frames = []
    for dk in demos[:3]:
        states = f[f"data/{dk}/states"][()]
        sweep_frames += [(f[f"data/{dk}"].attrs["model_file"], s) for s in states[::10]]
    rw.load_state(sweep_frames[0][0], sweep_frames[0][1])
    obs_cache = []
    for _, s in sweep_frames:
        rw.set_state(s); o = rw.observe(args.policy_res)
        obs_cache.append((kin.joints_from_sim(rw.env.sim), {c: o[c]["robot_amodal"] for c in args.cams}))
    trans = [0, 0.0025, 0.005, 0.01, 0.02, 0.04]
    rots = [0, 0.5, 1, 2, 5, 10]
    jnoise = [0, 0.5, 1, 2, 5]
    sweep = {}
    for c in args.cams:
        base = tw[(c, args.policy_res, "gt")]
        cam0 = base.camera
        sweep[c] = {"trans_m": [], "rot_deg": [], "joint_deg": []}
        for kind, vals in [("trans_m", trans), ("rot_deg", rots)]:
            for v in vals:
                ious = []
                for k, (js, gm) in enumerate(obs_cache):
                    cp = cam0.perturbed(trans_m=v if kind == "trans_m" else 0, rot_deg=v if kind == "rot_deg" else 0,
                                        rng=np.random.default_rng(k))
                    base.set_camera(cp); ious.append(iou(gm[c], base.render(js)))
                sweep[c][kind].append((v, float(np.mean(ious))))
        base.set_camera(cam0)
        for v in jnoise:
            ious = []
            for k, (js, gm) in enumerate(obs_cache):
                r = np.random.default_rng(k)
                jn = {n: x + (r.normal() * np.deg2rad(v) if n.startswith("robot0_joint") else 0) for n, x in js.items()}
                ious.append(iou(gm[c], base.render(jn)))
            sweep[c]["joint_deg"].append((v, float(np.mean(ious))))
    json.dump(sweep, open(f"{args.out}/sensitivity.json", "w"), indent=1)
    print(json.dumps(sweep))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for c in args.cams:
        for a, kind, xl, sc in [(ax[0], "trans_m", "extrinsic translation error (mm)", 1000),
                                (ax[1], "rot_deg", "extrinsic rotation error (deg)", 1),
                                (ax[2], "joint_deg", "joint encoder noise σ (deg)", 1)]:
            xs, ys = zip(*sweep[c][kind])
            a.plot(np.array(xs) * sc, ys, "o-", label=CAM_LABEL.get(c, c))
            a.set_xlabel(xl); a.set_ylabel(f"mask IoU @ {args.policy_res}px"); a.grid(alpha=.3)
    for c in args.cams:
        ax[0].axvline(report[c]["trans_err_mm"], ls=":", c="gray")
        ax[1].axvline(report[c]["rot_err_deg"], ls=":", c="gray")
    ax[0].legend(fontsize=8)
    fig.suptitle("Twin mask IoU vs calibration / proprioception error (dotted: our calibration's error)")
    fig.tight_layout(); fig.savefig(f"{args.out}/sensitivity.png", dpi=120)


if __name__ == "__main__":
    main()
