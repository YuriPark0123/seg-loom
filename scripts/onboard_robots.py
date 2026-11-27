"""Turn public robot descriptions (URDF / xacro) into training robots: flattened URDF + auto-derived spec.

  python scripts/onboard_robots.py                 # all robots in SOURCES
  python scripts/onboard_robots.py --only xarm6 wx250s --preview

For every robot:
  1. xacro -> URDF ($(find pkg) resolved from the package.xml files under assets/src/extra)
  2. arm chain = movable joints from the root until the gripper branch (small side branches such as a
     head pan joint are ignored, a bimanual robot keeps its first arm); gripper = movable joints below
  3. approach axis / TCP from the geometry below the flange; a fixed 'tcp_frame' link (z = approach) is
     appended so the rest of the pipeline (IK "gripper down", wrist camera) works unchanged
  4. gripper open/close from joint limits (mimic joints follow), oriented so that 'open' is wider
  5. reach from the chain length, rest pose / workspace check with procgen.validate
Output: assets/robots_train/<name>/<name>.urdf, configs/robots/<name>.yaml, report in
configs/robots/extra_report.json. The 4 evaluation robots (so101, ur7e, piper, panda) and their close
relatives (UR family, Franka, SO-ARM, Piper) are deliberately NOT in SOURCES.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import traceback
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import yaml

EXTRA = os.path.join(ROOT, "assets/src/extra")
OUT_URDF = os.path.join(ROOT, "assets/robots_train")
OUT_CFG = os.path.join(ROOT, "configs/robots")

# name: (source path relative to assets/src/extra, xacro args, repo, license)
SOURCES = {
    "edo": ("edo/robots/edo_sim.urdf", None, "ianathompson/eDO_description", "BSD-3-Clause"),
    "fanuc_m710ic70": ("fanuc_m710ic_description/urdf/m710ic70.urdf", None, "robot-descriptions/fanuc_m710ic_description", "BSD-3-Clause"),
    "fanuc_lrmate200ic": ("fanuc/fanuc_lrmate200ic_support/urdf/lrmate200ic.xacro", {}, "ros-industrial/fanuc", "BSD-3-Clause"),
    "fanuc_m10ia": ("fanuc/fanuc_m10ia_support/urdf/m10ia.xacro", {}, "ros-industrial/fanuc", "BSD-3-Clause"),
    "fanuc_cr7ia": ("fanuc/fanuc_cr7ia_support/urdf/cr7ia.xacro", {}, "ros-industrial/fanuc", "BSD-3-Clause"),
    "fanuc_crx10ial": ("fanuc/fanuc_crx10ia_support/urdf/crx10ial.xacro", {}, "ros-industrial/fanuc", "BSD-3-Clause"),
    "kinova_gen2": ("example-robot-data/robots/kinova_description/robots/kinova.urdf", None, "Gepetto/example-robot-data", "BSD-3-Clause"),
    # dropped: "kinova_gen3_2f85": ("ros2_kortex/kortex_description/robots/gen3_2f85.urdf", None, "Kinovarobotics/ros2_kortex", "BSD-3-Clause"),
    "kinova_gen3_lite": ("ros2_kortex/kortex_description/robots/gen3_lite.urdf", None, "Kinovarobotics/ros2_kortex", "BSD-3-Clause"),
    "jaco2_j2n6s300": ("kinova-ros/kinova_description/urdf/j2n6s300_standalone.xacro", {}, "Kinovarobotics/kinova-ros", "BSD-3-Clause"),
    # dropped: "jaco2_j2s7s300": ("kinova-ros/kinova_description/urdf/j2s7s300_standalone.xacro", {}, "Kinovarobotics/kinova-ros", "BSD-3-Clause"),
    "mico_m1n6s200": ("kinova-ros/kinova_description/urdf/m1n6s200_standalone.xacro", {}, "Kinovarobotics/kinova-ros", "BSD-3-Clause"),
    "iiwa14": ("drake/manipulation/models/iiwa_description/urdf/iiwa14_no_collision.urdf", None, "RobotLocomotion/drake", "BSD-3-Clause"),
    "iiwa7": ("differentiable-robot-model/diff_robot_data/kuka_iiwa/urdf/iiwa7.urdf", None, "facebookresearch/differentiable-robot-model", "MIT"),
    "rizon4": ("flexiv_description/urdf/flexiv.urdf.xacro", {"robot_type": "Rizon4"}, "flexivrobotics/flexiv_description", "Apache-2.0"),
    "rizon10": ("flexiv_description/urdf/flexiv.urdf.xacro", {"robot_type": "Rizon10"}, "flexivrobotics/flexiv_description", "Apache-2.0"),
    "xarm6": ("xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro", {"dof": "6", "add_gripper": "true"}, "xArm-Developer/xarm_ros2", "BSD-3-Clause"),
    "xarm7": ("xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro", {"dof": "7", "add_gripper": "true"}, "xArm-Developer/xarm_ros2", "BSD-3-Clause"),
    "xarm5": ("xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro", {"dof": "5", "add_gripper": "false"}, "xArm-Developer/xarm_ros2", "BSD-3-Clause"),
    "lite6": ("xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro", {"dof": "6", "robot_type": "lite", "add_gripper": "true"}, "xArm-Developer/xarm_ros2", "BSD-3-Clause"),
    "uf850": ("xarm_ros2/xarm_description/urdf/xarm_device.urdf.xacro", {"dof": "6", "robot_type": "uf850", "add_gripper": "true"}, "xArm-Developer/xarm_ros2", "BSD-3-Clause"),
    "omx_f": ("open_manipulator/open_manipulator_description/urdf/omx_f/omx_f.urdf", None, "ROBOTIS-GIT/open_manipulator", "Apache-2.0"),
    "omy_3m": ("open_manipulator/open_manipulator_description/urdf/omy_3m/omy_3m.urdf", None, "ROBOTIS-GIT/open_manipulator", "Apache-2.0"),
    "open_manipulator_x": ("open_manipulator/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf", None, "ROBOTIS-GIT/open_manipulator", "Apache-2.0"),
    # dropped: "openarm_v1": ("openarm_description/assets/robot/openarm_v1.0/urdf/openarm_v10.urdf.xacro", {}, "enactic/openarm_description", "Apache-2.0"),
    "yam": ("i2rt/i2rt/robot_models/yam/yam.urdf", None, "robot-descriptions/i2rt", "MIT"),
    # dropped: "arx_r5": ("i2rt/i2rt/robot_models/arx_r5", None, "robot-descriptions/i2rt", "MIT"),
    "sawyer": ("sawyer_robot/sawyer_description/urdf/sawyer.urdf.xacro", {"electric_gripper": "false"}, "RethinkRobotics/sawyer_robot", "Apache-2.0"),
    "px150": ("interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/px150.urdf.xacro", {}, "Interbotix/interbotix_ros_manipulators", "BSD-3-Clause"),
    "rx200": ("interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx200.urdf.xacro", {}, "Interbotix/interbotix_ros_manipulators", "BSD-3-Clause"),
    "wx250s": ("interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx250s.urdf.xacro", {}, "Interbotix/interbotix_ros_manipulators", "BSD-3-Clause"),
    "vx300s": ("interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro", {}, "Interbotix/interbotix_ros_manipulators", "BSD-3-Clause"),
}
FINGER_RE = re.compile(r"finger|gripper|jaw|knuckle|claw|drive|hand_joint|thumb|tip", re.I)


# ------------------------------------------------------------------------------------------- packages / xacro
def package_map():
    pk = {}
    for p in glob.glob(os.path.join(EXTRA, "**/package.xml"), recursive=True):
        try:
            n = ET.parse(p).getroot().findtext("name").strip()
            pk.setdefault(n, os.path.dirname(p))
        except Exception:
            pass
    return pk


def flatten(src, args, pk):
    if args is None:
        return open(src).read()
    import xacro
    import xacro.substitution_args as sa
    sa._eval_find = lambda pkg: pk[pkg]
    doc = xacro.process_file(src, mappings={k: str(v) for k, v in args.items()})
    return doc.toprettyxml(indent="  ")


# ------------------------------------------------------------------------------------------- tree analysis
def tree(root):
    joints = {j.get("name"): j for j in root.findall("joint")}
    children = {}
    for j in joints.values():
        children.setdefault(j.find("parent").get("link"), []).append(j)
    child_links = {j.find("child").get("link") for j in joints.values()}
    roots = [l.get("name") for l in root.findall("link") if l.get("name") not in child_links]
    return joints, children, roots[0]


def movable(j):
    return j.get("type") in ("revolute", "continuous", "prismatic") and j.find("mimic") is None


def subtree_joints(link, children):
    out = []
    for j in children.get(link, []):
        if j.get("type") in ("revolute", "continuous", "prismatic"):
            out.append(j)
        out += subtree_joints(j.find("child").get("link"), children)
    return out


def find_arm(root):
    """Return (arm joint names, flange link, gripper joint elements)."""
    joints, children, link = tree(root)
    arm = []
    while True:
        branches = []
        for j in children.get(link, []):
            sub = ([j] if j.get("type") in ("revolute", "continuous", "prismatic") else []) + subtree_joints(j.find("child").get("link"), children)
            branches.append((j, [s for s in sub if movable(s)], sub))
        live = [b for b in branches if b[1]]
        if not live:
            return arm, link, []
        big = [b for b in live if len(b[1]) >= 3]
        if len(live) == 1 and not (movable(live[0][0]) and FINGER_RE.search(live[0][0].get("name")) and len(live[0][1]) <= 3) \
                and not (len(live[0][1]) <= 2 and arm and live[0][0].get("type") == "prismatic"):
            j = live[0][0]
        elif big:
            j = big[0][0]                                  # main arm (ignore head pan, keep first arm of a bimanual robot)
        else:                                              # several small movable branches -> gripper
            return arm, link, [s for b in live for s in b[2]]
        if movable(j) and j.get("type") != "prismatic":
            arm.append(j.get("name"))
        elif movable(j) and j.get("type") == "prismatic" and len(subtree_joints(j.find("child").get("link"), children)) == 0:
            return arm, link, [s for b in live for s in b[2]]
        link = j.find("child").get("link")


# ------------------------------------------------------------------------------------------- derive spec
def fix_urdf(root, src_dir, repo_dir, pk):
    """Absolute mesh paths, unknown package:// names, machine-specific install paths, massless links."""
    for m in root.iter("mesh"):
        fn = m.get("filename")
        if fn.startswith("file://"):
            fn = fn[len("file://"):]
        mm = re.search(r"/share/([^/]+)/(.*)$", fn) if fn.startswith("/") and not os.path.exists(fn) else None
        if mm and mm.group(1) in pk:
            fn = os.path.join(pk[mm.group(1)], mm.group(2))
        elif fn.startswith("package://"):
            pkg, rel = fn[len("package://"):].split("/", 1)
            if pkg not in pk:                                       # package name != package.xml name: search the repo
                hit = glob.glob(os.path.join(repo_dir, "**", rel), recursive=True)
                if not hit:
                    raise RuntimeError(f"mesh not found: {fn}")
                pk[pkg] = hit[0][:-len(rel)].rstrip("/")
            fn = os.path.join(pk[pkg], rel)
        elif fn.startswith("file://"):
            fn = fn[len("file://"):]
        elif not os.path.isabs(fn):                                 # relative to the URDF dir or one of its parents
            d0, cand = src_dir, None
            while d0.startswith(repo_dir):
                if os.path.exists(os.path.join(d0, fn)):
                    cand = os.path.normpath(os.path.join(d0, fn)); break
                d0 = os.path.dirname(d0)
            fn = cand or os.path.normpath(os.path.join(src_dir, fn))
        if not os.path.exists(fn):
            raise RuntimeError(f"mesh not found: {fn}")
        fn = os.path.realpath(fn)
        if not fn.startswith(os.path.realpath(EXTRA) + "/"):
            raise RuntimeError(f"mesh outside assets/src/extra: {fn}")
        m.set("filename", "package://extra/" + os.path.relpath(fn, os.path.realpath(EXTRA)))   # portable
    for link in root.findall("link"):
        i = link.find("inertial")
        if i is None:                                               # massless link: give it a tiny inertia
            i = ET.SubElement(link, "inertial")
            ET.SubElement(i, "mass", value="0.01")
            ET.SubElement(i, "inertia", ixx="1e-5", iyy="1e-5", izz="1e-5", ixy="0", ixz="0", iyz="0")
        ms = i.find("mass")
        if ms is not None and float(ms.get("value", 0)) < 1e-3:
            ms.set("value", "0.01")
        it = i.find("inertia")
        if it is not None:
            for k in ("ixx", "iyy", "izz"):
                if float(it.get(k, 0)) < 1e-6:
                    it.set(k, "1e-5")


def derive(name, urdf_text, pk, rng, src_dir, repo_dir):
    from shadow.urdf_robot import RobotSpec, urdf_to_mjcf
    import mujoco
    d = os.path.join(OUT_URDF, name)
    os.makedirs(d, exist_ok=True)
    root = ET.fromstring(urdf_text)
    for t in ("gazebo", "transmission", "ros2_control"):
        for e in root.findall(t):
            root.remove(e)
    fix_urdf(root, src_dir, repo_dir, pk)
    arm, flange, grip = find_arm(root)
    if len(arm) < 4:
        raise RuntimeError(f"arm chain too short: {arm}")
    pkgs = {"extra": os.path.relpath(EXTRA, ROOT)}             # every mesh is package://extra/<repo>/...
    path0 = os.path.join(d, f"{name}_raw.urdf")
    ET.ElementTree(root).write(path0)
    # provisional spec: ee = flange, compile and measure geometry in the flange frame
    spec0 = RobotSpec(name=f"_{name}", urdf=path0, arm_joints=arm, ee_body=flange, rest_q=[0.0] * len(arm), reach=1.0,
                      package_dirs={k: os.path.join(ROOT, v) for k, v in pkgs.items()}, asset_dir=os.path.join(d, "mj0"))
    m = mujoco.MjModel.from_xml_string(urdf_to_mjcf(spec0, rebuild=True))
    da = mujoco.MjData(m)
    mujoco.mj_kinematics(m, da)
    fid = m.body(flange).id
    Rf, pf = da.xmat[fid].reshape(3, 3), da.xpos[fid]
    below = set()
    for b in range(m.nbody):                                   # bodies at/after the flange
        a = b
        while a > 0 and a != fid:
            a = m.body_parentid[a]
        if a == fid:
            below.add(b)
    finger_bodies = {m.joint(j.get("name")).bodyid[0] for j in grip if j.get("name") in [m.joint(i).name for i in range(m.njnt)]}
    pts, fpts = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in below or m.geom_group[g] != 1:
            continue
        b = m.geom_bodyid[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            s = m.geom_size[g]; t = m.geom_type[g]
            half = [s[0]] * 3 if t == 2 else ([s[0], s[0], s[1] + (s[0] if t == 3 else 0)] if t in (3, 5) else s[:3])
            v = np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T * np.asarray(half)
        w = v @ da.geom_xmat[g].reshape(3, 3).T + da.geom_xpos[g]
        loc = (w - pf) @ Rf                                       # into the flange frame
        pts.append(loc)
        anc = b
        while anc not in finger_bodies and anc > 0 and anc != fid:
            anc = m.body_parentid[anc]
        if anc in finger_bodies:
            fpts.append(loc)
    # approach axis: the last arm joint axis, signed towards the tool geometry
    lj = m.joint(arm[-1]).id
    ax_w = da.xaxis[lj]
    ax = Rf.T @ ax_w
    P = np.concatenate(pts) if pts else np.zeros((1, 3))
    cen = P.mean(0)
    dirn = cen / (np.linalg.norm(cen) + 1e-9)
    a = ax * np.sign(ax @ cen if abs(ax @ cen) > 1e-4 else 1.0)
    if np.linalg.norm(cen) > 0.02 and abs(dirn @ (ax / np.linalg.norm(ax))) < 0.7:
        a = dirn                                                  # offset tool: point at the tool instead
    a = a / np.linalg.norm(a)
    F = np.concatenate(fpts) if fpts else P
    proj = F @ a
    tip = proj.max()
    lat = F[proj > tip - 0.3 * (tip - proj.min() + 1e-6)].mean(0)
    lat = lat - (lat @ a) * a
    tcp = lat + a * (tip - (0.18 * (tip - F @ a).max() if fpts else 0.0))
    if not fpts:
        tcp = a * max(tip, 0.0)
    # finger slide / side axis: direction between finger groups (or any perpendicular)
    side = None
    if len(finger_bodies) >= 2:
        fb = sorted(finger_bodies)
        c = [((da.xpos[b] - pf) @ Rf) for b in fb]
        v = c[0] - c[-1]; v = v - (v @ a) * a
        if np.linalg.norm(v) > 1e-3:
            side = v / np.linalg.norm(v)
    if side is None:
        side = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        side /= np.linalg.norm(side)
    # tcp_frame: z = approach, y = side (fingers slide along y, as for the procedural robots)
    y = side - (side @ a) * a; y /= np.linalg.norm(y); x = np.cross(y, a)
    from scipy.spatial.transform import Rotation
    Rt = np.stack([x, y, a], 1)
    rpy = Rotation.from_matrix(Rt).as_euler("xyz")
    lk = ET.SubElement(root, "link", name="tcp_frame")
    jt = ET.SubElement(root, "joint", name="tcp_frame_joint", type="fixed")
    ET.SubElement(jt, "parent", link=flange); ET.SubElement(jt, "child", link="tcp_frame")
    ET.SubElement(jt, "origin", xyz=" ".join(f"{v:.6g}" for v in tcp), rpy=" ".join(f"{v:.6g}" for v in rpy))
    path = os.path.join(d, f"{name}.urdf")
    ET.ElementTree(root).write(path)
    # chain length (zero pose): first arm joint anchor -> ... -> tcp
    anchors = [da.xanchor[m.joint(j).id] for j in arm] + [pf + Rf @ tcp]
    chain = float(sum(np.linalg.norm(np.diff(np.array(anchors), axis=0), axis=1)))
    # gripper open / close from limits (mimic joints follow their driver automatically in MuJoCo)
    gj = [j for j in grip if movable(j)]
    gripper = None
    if gj:
        lo, hi = [], []
        for j in gj:
            lim = j.find("limit")
            l, h = (float(lim.get("lower", 0)), float(lim.get("upper", 0))) if lim is not None else (0.0, 0.0)
            lo.append(l); hi.append(h)
        gripper = dict(joints=[j.get("name") for j in gj], open=hi, close=lo)
    spec = dict(name=name, urdf=os.path.relpath(path, ROOT), package_dirs=pkgs, arm_joints=arm,
                ee_body="tcp_frame", tcp_offset=[0, 0, 0], rest_q=[0.0] * len(arm), reach=round(0.95 * chain, 3),
                rot_weight=0.5 if len(arm) >= 6 else 0.1, asset_dir=os.path.relpath(os.path.join(d, "mj"), ROOT),
                wrist_cam=dict(side_axis=[1, 0, 0]))
    if gripper:
        spec["gripper"] = gripper
    return spec


def finalize(spec, rng):
    """Orient the gripper (open = wider), find the rest pose; returns (spec, info) or raises."""
    from shadow.procgen import validate
    from shadow.urdf_robot import RobotSpec
    from shadow.datagen import gripper_width_table
    from shadow.mj_world import Layout
    p = os.path.join(OUT_CFG, f"{spec['name']}.yaml")
    yaml.safe_dump(spec, open(p, "w"), sort_keys=False)
    S = RobotSpec.load(p)
    info = {}
    if S.gripper:
        al, w = gripper_width_table(S, Layout(S))
        info["grip_width_open_close"] = [float(w[-1]), float(w[0])]
        if w[0] > w[-1] + 1e-3:                                   # 'open' was the narrow end -> swap
            spec["gripper"]["open"], spec["gripper"]["close"] = spec["gripper"]["close"], spec["gripper"]["open"]
            info["grip_swapped"] = True
        yaml.safe_dump(spec, open(p, "w"), sort_keys=False)
        S = RobotSpec.load(p)
    reach0 = spec["reach"]
    for k in range(4):                                            # chain length overestimates reach for offset arms
        rest, frac, vinfo = validate(S, rng)
        if rest is not None:
            break
        spec["reach"] = round(reach0 * (0.85 - 0.12 * k), 3)
        yaml.safe_dump(spec, open(p, "w"), sort_keys=False)
        S = RobotSpec.load(p)
    info.update(vinfo, workspace=frac, reach_scale=round(spec["reach"] / reach0, 3))
    if rest is None:
        raise RuntimeError(f"validate: {vinfo.get('reason')}")
    spec["rest_q"] = [round(float(v), 4) for v in rest]
    yaml.safe_dump(spec, open(p, "w"), sort_keys=False)
    spec["wrist_cam"], info["wrist_robot_frac"] = tune_wrist(p, spec, rng)
    yaml.safe_dump(spec, open(p, "w"), sort_keys=False)
    return spec, info


def _inside_robot(w, pt, margin):
    """Is pt inside (or within margin of) the convex hull of any robot visual geom?"""
    import mujoco
    from scipy.spatial import Delaunay
    m, d = w.model, w.data
    for g in range(m.ngeom):
        if not w.is_robot[g] or m.geom_group[g] != 1:
            continue
        if np.linalg.norm(d.geom_xpos[g] - pt) > m.geom_rbound[g] + margin:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            sz = m.geom_size[g]; t = m.geom_type[g]
            half = [sz[0]] * 3 if t == 2 else ([sz[0], sz[0], sz[1] + (sz[0] if t == 3 else 0)] if t in (3, 5) else sz[:3])
            v = np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T * np.asarray(half)
        c = v.mean(0)
        v = c + (v - c) * (1 + margin / max(np.linalg.norm(v - c, axis=1).max(), 1e-6))
        wv = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        try:
            if Delaunay(wv).find_simplex(pt) >= 0:
                return True
        except Exception:
            pass
    return False


def tune_wrist(p, spec, rng, max_frac=0.3):
    """Wrist camera: smallest side offset (4 directions) that is outside the robot, sees < max_frac robot and,
    preferably, a bit of the tool (fraction >= 0.01). Returns (wrist_cam, robot fraction)."""
    from shadow.camera_model import CameraModel
    from shadow.datagen import DataWorld, sample_scene
    from shadow.mj_world import Layout
    from shadow.urdf_robot import RobotSpec
    S = RobotSpec.load(p); L = Layout(S); R = L.R
    cam = CameraModel(160, 120, 80.0, 80.0, 80.0, 60.0)
    extras = dict(attachments=[], wrist_delta=dict(dpos=[0, 0, 0], drot=[0, 0, 0]), bodies=[])
    scene = sample_scene(S, L, np.random.default_rng(0), n_obj=0)
    fallback = None
    for k in (0.16, 0.25, 0.35, 0.5, 0.7):
        cands = []
        for sx in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
            wc = dict(side_axis=sx, pos=[round(float(v), 4) for v in np.array(sx) * k * R + np.r_[0, 0, -0.08 * R]],
                      look_at=[0, 0, round(0.35 * R, 4)], fovy=75)
            S.wrist_cam = wc
            w = DataWorld(S, L, scene, extras)
            w.set_q(S.rest_q, [float(v) for v in (S.gripper["open"] if S.gripper else [])]); w.refresh()
            if _inside_robot(w, w.data.cam_xpos[w.cam_ids["wrist"]].copy(), 0.01 * R):
                continue
            _, vis, _, _ = w.render("wrist", cam)
            frac = float(((vis >= 1) & (vis <= len(w.robot_bodies))).mean())
            if frac < max_frac:
                cands.append((frac < 0.01, abs(frac - 0.1), wc, frac))
        if cands:
            cands.sort(key=lambda c: (c[0], c[1]))
            if not cands[0][0]:
                return cands[0][2], cands[0][3]
            fallback = fallback or cands[0]
    if fallback is None:
        raise RuntimeError("no wrist camera position outside the robot")
    return fallback[2], fallback[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    os.makedirs(OUT_CFG, exist_ok=True)
    pk = package_map()
    rep_p = os.path.join(OUT_CFG, "extra_report.json")
    report = json.load(open(rep_p)) if os.path.exists(rep_p) else {}
    for name, (src, args, repo, lic) in SOURCES.items():
        if a.only and name not in a.only:
            continue
        rng = np.random.default_rng(abs(hash(name)) % 2 ** 32 if False else sum(map(ord, name)))
        try:
            s = os.path.join(EXTRA, src)
            if os.path.isdir(s):
                s = sorted(glob.glob(os.path.join(s, "*.urdf")))[0]
            repo_dir = os.path.join(EXTRA, src.split("/")[0])
            spec = derive(name, flatten(s, args, pk), pk, rng, os.path.dirname(s), repo_dir)
            spec, info = finalize(spec, rng)
            report[name] = dict(ok=True, repo=repo, license=lic, source=src, xacro_args=args, n_arm=len(spec["arm_joints"]),
                                arm=spec["arm_joints"], gripper=(spec.get("gripper") or {}).get("joints"), reach=spec["reach"], **info)
            print(f"OK   {name:20s} arm {len(spec['arm_joints'])} grip {len((spec.get('gripper') or {}).get('joints', []))} reach {spec['reach']:.2f} ws {info['workspace']:.2f}", flush=True)
        except Exception as e:
            report[name] = dict(ok=False, repo=repo, license=lic, source=src, error=f"{type(e).__name__}: {e}")
            print(f"FAIL {name:20s} {type(e).__name__}: {str(e)[:160]}", flush=True)
            if os.environ.get("DEBUG"):
                traceback.print_exc()
        json.dump(report, open(rep_p, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
