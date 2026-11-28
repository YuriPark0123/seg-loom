"""Convert a MuJoCo (MJCF) robot, e.g. from mujoco_menagerie, into a URDF + OBJ meshes for the onboarding path.

  python scripts/mjcf_to_urdf.py <model.xml> <out_dir> [--name NAME] [--root BODY]

What is kept: the kinematic tree (hinge -> revolute/continuous, slide -> prismatic, no joint -> fixed),
visual geoms (meshes as compiled by MuJoCo, primitives triangulated) with their material colours,
joint couplings written as equality/joint constraints (-> <mimic>). Not representable in URDF and dropped:
closed kinematic loops (equality/connect), tendons, textures, actuators. --root keeps only the subtree under
that body (e.g. one arm of a mobile manipulator); everything above it is cut off.
"""
import argparse
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _f(v):
    return " ".join(f"{float(x):.7g}" for x in v)


def _rpy(q_wxyz):
    return Rotation.from_quat(np.r_[q_wxyz[1:], q_wxyz[0]]).as_euler("xyz")


def _rot(q_wxyz):
    return Rotation.from_quat(np.r_[q_wxyz[1:], q_wxyz[0]]).as_matrix()


def convert(xml_path, out_dir, name=None, root_body=None):
    m = mujoco.MjModel.from_xml_path(xml_path)
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    name = name or os.path.splitext(os.path.basename(xml_path))[0]
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
    # visual geoms: menagerie puts them in group 2 (contype=conaffinity=0); otherwise take non-colliding ones, else all
    vis = [g for g in range(m.ngeom) if m.geom_group[g] == 2]
    if not vis:
        vis = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
    if not vis:
        vis = list(range(m.ngeom))
    vis = [g for g in vis if m.geom_bodyid[g] != 0 and m.geom_rgba[g][3] > 0]
    # subtree to keep
    keep = set(range(1, m.nbody))
    if root_body:
        r = m.body(root_body).id
        keep = {b for b in range(1, m.nbody) if _is_under(m, b, r)}
    # link frame = body frame shifted to its joint position (URDF rotates about the link origin)
    top = [b for b in sorted(keep) if m.body_parentid[b] not in keep]
    off = {0: np.zeros(3)}
    for b in range(1, m.nbody):
        nj = m.body_jntnum[b] if b not in top else 0             # joints of the cut-off root (floating base) dropped
        if nj > 1:
            raise RuntimeError(f"body {bname(b)} has {nj} joints (not supported)")
        j = m.body_jntadr[b]
        off[b] = m.jnt_pos[j].copy() if nj == 1 and m.jnt_type[j] in (2, 3) else np.zeros(3)
    root = ET.Element("robot", name=name)
    if len(top) != 1:
        ET.SubElement(root, "link", name="base_root")
    mesh_file = {}
    mats = {}
    for b in sorted(keep):
        link = ET.SubElement(root, "link", name=bname(b))
        mass = max(float(m.body_mass[b]), 1e-3)
        inl = ET.SubElement(link, "inertial")
        ET.SubElement(inl, "origin", xyz=_f(m.body_ipos[b] - off[b]), rpy="0 0 0")
        ET.SubElement(inl, "mass", value=f"{mass:.6g}")
        I = np.maximum(m.body_inertia[b], 1e-6)
        ET.SubElement(inl, "inertia", ixx=f"{I[0]:.6g}", iyy=f"{I[1]:.6g}", izz=f"{I[2]:.6g}", ixy="0", ixz="0", iyz="0")
        for g in vis:
            if m.geom_bodyid[g] != b:
                continue
            fn = _geom_mesh(m, g, out_dir, mesh_file)
            if fn is None:
                continue
            v = ET.SubElement(link, "visual")
            ET.SubElement(v, "origin", xyz=_f(m.geom_pos[g] - off[b]), rpy=_f(_rpy(m.geom_quat[g])))
            ET.SubElement(ET.SubElement(v, "geometry"), "mesh", filename=f"meshes/{fn}")
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            key = _f(rgba)
            mats.setdefault(key, f"mat{len(mats)}")
            mt = ET.SubElement(v, "material", name=mats[key])
            ET.SubElement(mt, "color", rgba=key)
    for b in sorted(keep):
        p = m.body_parentid[b]
        par = bname(p) if p in keep else ("base_root" if len(top) != 1 else None)
        if par is None:
            continue
        R = _rot(m.body_quat[b])
        xyz = m.body_pos[b] - off.get(p, 0) * (p in keep) + R @ off[b]
        nj = m.body_jntnum[b] if b not in top else 0
        jt = m.jnt_type[m.body_jntadr[b]] if nj else None
        if nj == 0 or jt == 0:                                   # fixed (free joints of the root are dropped)
            j = ET.SubElement(root, "joint", name=f"{bname(b)}_fixed", type="fixed")
        else:
            jid = m.body_jntadr[b]
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint{jid}"
            if jt == 1:
                raise RuntimeError(f"ball joint {jn} not supported")
            typ = "prismatic" if jt == 2 else ("revolute" if m.jnt_limited[jid] else "continuous")
            j = ET.SubElement(root, "joint", name=jn, type=typ)
            ET.SubElement(j, "axis", xyz=_f(m.jnt_axis[jid]))
            lo, hi = m.jnt_range[jid] if m.jnt_limited[jid] else (-np.pi, np.pi)
            ET.SubElement(j, "limit", lower=f"{lo:.6g}", upper=f"{hi:.6g}", effort="100", velocity="2")
        ET.SubElement(j, "parent", link=par); ET.SubElement(j, "child", link=bname(b))
        ET.SubElement(j, "origin", xyz=_f(xyz), rpy=_f(_rpy(m.body_quat[b])))
    # joint couplings -> mimic (joint1 = c0 + c1 * joint2)
    jel = {j.get("name"): j for j in root.findall("joint")}
    for e in range(m.neq):
        if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        j1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[e])
        j2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj2id[e])
        if j1 in jel and j2 in jel and m.eq_obj2id[e] >= 0:
            c = m.eq_data[e]
            ET.SubElement(jel[j1], "mimic", joint=j2, multiplier=f"{c[1]:.6g}", offset=f"{c[0]:.6g}")
    fit = fit_couplings(m, set(jel))
    for jn, (lead, k, c0, res) in fit.items():
        mim = jel[lead].find("mimic")
        if mim is not None:                     # leader is itself coupled (equality): refer to the free joint instead
            k2, c2 = float(mim.get("multiplier")), float(mim.get("offset"))
            lead, k, c0 = mim.get("joint"), k * k2, c0 + k * c2
        if jel[jn].find("mimic") is None and jn != lead:
            ET.SubElement(jel[jn], "mimic", joint=lead, multiplier=f"{k:.6g}", offset=f"{c0:.6g}")
            print(f"  coupling {jn} = {c0:+.4f} + {k:+.4f} * {lead}  (fit residual {res:.2g})")
    path = os.path.join(out_dir, f"{name}.urdf")
    ET.indent(root)
    ET.ElementTree(root).write(path)
    return path


def fit_couplings(m, names, n=15):
    """Gripper joints driven through a tendon or a closed loop (equality/connect) cannot be written as URDF.
    Sweep each such actuator in simulation (no gravity, other actuators holding) and fit every joint that moves
    linearly to the one that moves most: joint = c0 + k * leader. Returns {joint: (leader, k, c0, max residual)}."""
    tendon_acts = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON]
    loops = any(m.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT for e in range(m.neq))
    if not tendon_acts and not loops:
        return {}
    acts = tendon_acts or [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                           and m.jnt_type[m.actuator_trnid[a][0]] in (2, 3) and _is_gripperish(m, m.actuator_trnid[a][0])]
    out = {}
    m.opt.gravity[:] = 0
    for a in acts:
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        lo, hi = m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else (0, 1)
        Q = []
        for c in np.linspace(lo, hi, n):
            d.ctrl[a] = c
            for _ in range(1500):
                mujoco.mj_step(m, d)
            Q.append(d.qpos.copy())
        Q = np.array(Q)
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] in (2, 3)]
        mov = [j for j in hinge if np.ptp(Q[:, m.jnt_qposadr[j]]) > 1e-3]
        if len(mov) < 2:
            continue
        lead = max(mov, key=lambda j: np.ptp(Q[:, m.jnt_qposadr[j]]))
        x = Q[:, m.jnt_qposadr[lead]]
        ln = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, lead)
        for j in mov:
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jn not in names or j == lead:
                continue
            y = Q[:, m.jnt_qposadr[j]]
            k, c0 = np.polyfit(x, y, 1)
            out[jn] = (ln, float(k), float(c0), float(np.abs(np.polyval([k, c0], x) - y).max()))
    return out


def _is_gripperish(m, j):
    n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").lower()
    return any(t in n for t in ("finger", "gripper", "driver", "jaw", "knuckle"))


def _is_under(m, b, r):
    while b > 0:
        if b == r:
            return True
        b = m.body_parentid[b]
    return False


def _geom_mesh(m, g, out_dir, cache):
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        key = ("mesh", mid)
        if key not in cache:
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn_ = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            tm = trimesh.Trimesh(m.mesh_vert[va:va + vn], m.mesh_face[fa:fa + fn_], process=False)
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh{mid}"
            cache[key] = f"{nm}.obj"
            tm.export(os.path.join(out_dir, "meshes", cache[key]))
        return cache[key]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        tm = trimesh.creation.box(extents=2 * s[:3])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        tm = trimesh.creation.icosphere(subdivisions=2, radius=s[0])
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        tm = trimesh.creation.cylinder(radius=s[0], height=2 * s[1], sections=24)
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        tm = trimesh.creation.capsule(radius=s[0], height=2 * s[1], count=[16, 16])
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        tm = trimesh.creation.icosphere(subdivisions=2, radius=1.0); tm.apply_scale(s[:3])
    else:
        return None
    key = ("prim", g)
    cache[key] = f"prim_{g}.obj"
    tm.export(os.path.join(out_dir, "meshes", cache[key]))
    return cache[key]


def menagerie_list():
    """MENAGERIE table of scripts/onboard_robots.py (read without importing it: this runs in a MuJoCo>=3 env)."""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "onboard_robots.py")).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "MENAGERIE")
    return ast.literal_eval(node.value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--name"); ap.add_argument("--root")
    ap.add_argument("--menagerie", metavar="EXTRA_DIR",
                    help="convert every MENAGERIE entry of onboard_robots.py: EXTRA_DIR/mujoco_menagerie -> EXTRA_DIR/menagerie_urdf")
    a = ap.parse_args()
    if a.menagerie:
        for n, (x, r, _) in menagerie_list().items():
            print(n, convert(os.path.join(a.menagerie, "mujoco_menagerie", x), os.path.join(a.menagerie, "menagerie_urdf", n), n, r))
    else:
        print(convert(a.xml, a.out, a.name, a.root))
