"""Procedural robot arms: random kinematic structure + primitive geometry + gripper, written as a
URDF plus a RobotSpec yaml, so they go through exactly the same pipeline as real robots.

Structure (all links point along +z of their own frame at q = 0):
  base  -> j1 (yaw, sometimes pitch) -> shoulder offset -> j2 pitch -> upper arm -> j3 pitch
        -> forearm -> wrist joints (random yaw / pitch) -> flange -> hand (palm) -> fingers
Random: joint count 3..7, link lengths and radii, lateral offsets (UR-like), fixed frame
re-orientations, joint limits, link style (cylinder / box), colors, gripper type (parallel prismatic
or rotary two-finger, second finger = URDF mimic).

  generate_robot(rng, name, out_root) -> path to spec yaml (or None if it failed validation)
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PALETTE = {
    "white": (0.92, 0.92, 0.9), "light_gray": (0.7, 0.71, 0.73), "dark_gray": (0.28, 0.29, 0.31),
    "black": (0.08, 0.08, 0.09), "orange": (0.95, 0.45, 0.1), "blue": (0.15, 0.3, 0.7),
    "red": (0.75, 0.12, 0.1), "yellow": (0.95, 0.8, 0.15), "teal": (0.1, 0.55, 0.55), "silver": (0.8, 0.8, 0.82),
}


def _f(v):
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _inertial(link, mass, dims, kind):
    """Solid cylinder (r, L along z) or box (x, y, z) inertia."""
    ine = ET.SubElement(link, "inertial")
    ET.SubElement(ine, "origin", xyz=_f([0, 0, dims[-1] / 2 if kind != "sphere" else 0]), rpy="0 0 0")
    ET.SubElement(ine, "mass", value=f"{mass:.5g}")
    if kind == "cyl":
        r, L = dims
        ixx = mass * (3 * r * r + L * L) / 12; izz = mass * r * r / 2
        I = (ixx, ixx, izz)
    else:
        x, y, z = dims
        I = (mass * (y * y + z * z) / 12, mass * (x * x + z * z) / 12, mass * (x * x + y * y) / 12)
    I = [max(i, 1e-6) for i in I]
    ET.SubElement(ine, "inertia", ixx=f"{I[0]:.6g}", iyy=f"{I[1]:.6g}", izz=f"{I[2]:.6g}", ixy="0", ixz="0", iyz="0")


def _shape(link, kind, dims, xyz, rpy, rgba, name):
    """Add identical visual + collision elements."""
    for tag in ("visual", "collision"):
        el = ET.SubElement(link, tag, name=f"{name}_{tag[:3]}")
        ET.SubElement(el, "origin", xyz=_f(xyz), rpy=_f(rpy))
        g = ET.SubElement(el, "geometry")
        if kind == "cyl":
            ET.SubElement(g, "cylinder", radius=f"{dims[0]:.5g}", length=f"{dims[1]:.5g}")
        elif kind == "sphere":
            ET.SubElement(g, "sphere", radius=f"{dims[0]:.5g}")
        else:
            ET.SubElement(g, "box", size=_f(dims))
        if tag == "visual":
            m = ET.SubElement(el, "material", name="")
            ET.SubElement(m, "color", rgba=_f(rgba))


def _axis_rpy(axis):
    """rpy that turns +z onto a principal axis (for joint housings drawn along the joint axis)."""
    a = np.abs(np.asarray(axis))
    if a[0] > 0.5:
        return [0, np.pi / 2, 0]
    if a[1] > 0.5:
        return [np.pi / 2, 0, 0]
    return [0, 0, 0]


def sample_design(rng: np.random.Generator):
    """Random structural parameters (in units of the overall size S, scaled at the end)."""
    n = int(rng.choice([3, 4, 5, 6, 6, 7]))
    S = float(rng.uniform(0.3, 1.0))                                   # upper arm + forearm ~ S
    types = ["yaw" if rng.random() < 0.85 else "pitch", "pitch"]
    if n >= 3:
        types.append("pitch")
    while len(types) < n:
        prev = types[-1]
        types.append("pitch" if prev == "yaw" else ("yaw" if rng.random() < 0.6 else "pitch"))
    lens = [rng.uniform(0.0, 0.1) * S]                                  # j1 -> j2 (shoulder offset)
    ua, fa = rng.uniform(0.42, 0.58), rng.uniform(0.35, 0.5)
    lens += [ua * S]
    if n >= 3:
        lens += [fa * S]
    while len(lens) < n:
        lens.append(rng.uniform(0.04, 0.12) * S)
    flange = rng.uniform(0.02, 0.06) * S
    lat = [0.0] * n                                                     # lateral (y) offsets of joint origins
    if rng.random() < 0.35 and n >= 3:
        k = int(rng.integers(1, min(n, 4)))
        lat[k] = float(rng.choice([-1, 1]) * rng.uniform(0.06, 0.14) * S)
    reorient = [0.0] * n                                                # fixed rotation about x at joint origin
    for k in range(3, n):
        if rng.random() < 0.15:
            reorient[k] = float(rng.choice([-1, 1]) * np.pi / 2)
    r0 = rng.uniform(0.04, 0.065) * S
    radii = [max(r0 * rng.uniform(0.8, 0.92) ** k, 0.012) for k in range(n + 1)]
    arm_col, acc_col = rng.choice(list(PALETTE), 2, replace=False)
    return dict(n=n, S=S, types=types, lens=lens, flange=flange, lat=lat, reorient=reorient, radii=radii,
                base_h=float(rng.uniform(0.08, 0.2) * S), style="box" if rng.random() < 0.35 else "cyl",
                arm_color=str(arm_col), accent_color=str(acc_col),
                pitch_axis=[0, 1, 0] if rng.random() < 0.7 else [1, 0, 0],
                gripper=str(rng.choice(["parallel", "parallel", "rotary"])),
                limits=[float(np.deg2rad(rng.uniform(150, 180) if t == "yaw" else rng.uniform(110, 160))) for t in types],
                palm=dict(w=float(rng.uniform(0.08, 0.15)), t=float(rng.uniform(0.03, 0.06)), h=float(rng.uniform(0.03, 0.06))),
                finger=dict(l=float(rng.uniform(0.05, 0.1)), w=float(rng.uniform(0.012, 0.022)),
                            open=float(rng.uniform(0.032, 0.045)), rot_open=float(rng.uniform(0.45, 0.8))),
                finger_color=str(rng.choice(["black", "dark_gray", "silver", "light_gray"])))


def build_urdf(name: str, D: dict) -> tuple[str, dict]:
    """Design -> (URDF xml string, info for the spec)."""
    S, n = D["S"], D["n"]
    rob = ET.Element("robot", name=name)
    arm = (*PALETTE[D["arm_color"]], 1.0)
    acc = (*PALETTE[D["accent_color"]], 1.0)
    fcol = (*PALETTE[D["finger_color"]], 1.0)
    # base
    base = ET.SubElement(rob, "link", name="base_link")
    rb = D["radii"][0] * 1.35
    if D["style"] == "box":
        _shape(base, "box", [2 * rb, 2 * rb, D["base_h"]], [0, 0, D["base_h"] / 2], [0, 0, 0], acc, "base")
    else:
        _shape(base, "cyl", [rb, D["base_h"]], [0, 0, D["base_h"] / 2], [0, 0, 0], acc, "base")
    _inertial(base, 3.0, [rb, D["base_h"]], "cyl")
    parent, parent_len = "base_link", D["base_h"]
    joints = []
    for k in range(n):
        lk = f"link{k + 1}"
        r = D["radii"][k + 1]
        L = D["lens"][k] if k + 1 < n else D["lens"][k]
        link = ET.SubElement(rob, "link", name=lk)
        axis = [0, 0, 1] if D["types"][k] == "yaw" else D["pitch_axis"]
        # joint housing (along the joint axis) + link body (along +z)
        hl = 2.3 * r
        if D["style"] == "box":
            _shape(link, "box", [hl if axis[0] else 2 * r, hl if axis[1] else 2 * r, hl if axis[2] else 2 * r],
                   [0, 0, 0], [0, 0, 0], acc, f"{lk}_hous")
            if L > 1e-3:
                _shape(link, "box", [1.6 * r, 1.6 * r, L], [0, 0, L / 2], [0, 0, 0], arm, f"{lk}_body")
        else:
            _shape(link, "cyl", [1.15 * r, hl], [0, 0, 0], _axis_rpy(axis), acc, f"{lk}_hous")
            if L > 1e-3:
                _shape(link, "cyl", [0.85 * r, L], [0, 0, L / 2], [0, 0, 0], arm, f"{lk}_body")
        _inertial(link, max(0.2, 4.0 * S * (0.75 ** k)), [r, max(L, 0.02)], "cyl")
        j = ET.SubElement(rob, "joint", name=f"joint{k + 1}", type="revolute")
        ET.SubElement(j, "parent", link=parent); ET.SubElement(j, "child", link=lk)
        ET.SubElement(j, "origin", xyz=_f([0, D["lat"][k], parent_len]), rpy=_f([D["reorient"][k], 0, 0]))
        ET.SubElement(j, "axis", xyz=_f(axis))
        lim = D["limits"][k]
        ET.SubElement(j, "limit", lower=f"{-lim:.5g}", upper=f"{lim:.5g}", effort="100", velocity="2.5")
        joints.append(f"joint{k + 1}")
        parent, parent_len = lk, L
    # hand (palm) on a fixed flange joint
    P, F = D["palm"], D["finger"]
    pw, pt, ph = P["w"] * S, P["t"] * S, P["h"] * S
    fl, fw = np.clip(F["l"] * S, 0.03, 0.085), np.clip(F["w"] * S, 0.008, 0.016)
    g0 = 0.004
    qmax = F["open"]
    pw = max(pw, g0 + 2 * qmax + 2 * fw + 0.01)
    pt = max(pt, 0.02)
    hand = ET.SubElement(rob, "link", name="hand")
    _shape(hand, "box", [pt, pw, ph], [0, 0, D["flange"] + ph / 2], [0, 0, 0], arm, "palm")
    _shape(hand, "cyl", [0.7 * D["radii"][-1], D["flange"]], [0, 0, D["flange"] / 2], [0, 0, 0], acc, "flange")
    _inertial(hand, 0.5, [pt, pw, ph], "box")
    j = ET.SubElement(rob, "joint", name="flange_joint", type="fixed")
    ET.SubElement(j, "parent", link=parent); ET.SubElement(j, "child", link="hand")
    ET.SubElement(j, "origin", xyz=_f([0, 0, parent_len]), rpy="0 0 0")
    top = D["flange"] + ph
    fingers = []
    if D["gripper"] == "parallel":
        for side, nm in ((1, "finger_l"), (-1, "finger_r")):
            fk = ET.SubElement(rob, "link", name=nm)
            _shape(fk, "box", [0.8 * pt, fw, fl], [0, side * fw / 2, fl / 2], [0, 0, 0], fcol, nm)
            _inertial(fk, 0.05, [pt, fw, fl], "box")
            j = ET.SubElement(rob, "joint", name=f"{nm}_joint", type="prismatic")
            ET.SubElement(j, "parent", link="hand"); ET.SubElement(j, "child", link=nm)
            ET.SubElement(j, "origin", xyz=_f([0, side * g0 / 2, top]), rpy="0 0 0")
            ET.SubElement(j, "axis", xyz=_f([0, side, 0]))
            ET.SubElement(j, "limit", lower="0", upper=f"{qmax:.5g}", effort="50", velocity="0.2")
            if side < 0:
                ET.SubElement(j, "mimic", joint="finger_l_joint", multiplier="1", offset="0")
            fingers.append(f"{nm}_joint")
        open_q, close_q = [qmax, qmax], [0.0, 0.0]
        grip = dict(type="parallel", g0=g0, finger_len=float(fl), open=float(qmax))
    else:
        th = D["finger"]["rot_open"]
        for side, nm in ((1, "finger_l"), (-1, "finger_r")):
            fk = ET.SubElement(rob, "link", name=nm)
            _shape(fk, "box", [0.8 * pt, fw, fl], [0, side * fw / 2, fl / 2], [0, 0, 0], fcol, nm)
            _shape(fk, "cyl", [fw * 0.6, 0.85 * pt], [0, side * fw / 2, 0], [0, np.pi / 2, 0], acc, f"{nm}_pin")
            _inertial(fk, 0.05, [pt, fw, fl], "box")
            j = ET.SubElement(rob, "joint", name=f"{nm}_joint", type="revolute")
            ET.SubElement(j, "parent", link="hand"); ET.SubElement(j, "child", link=nm)
            ET.SubElement(j, "origin", xyz=_f([0, side * g0 / 2, top]), rpy="0 0 0")
            ET.SubElement(j, "axis", xyz=_f([-side, 0, 0]))                # positive = open (tips move outward)
            ET.SubElement(j, "limit", lower="0", upper=f"{th:.5g}", effort="50", velocity="1.5")
            if side < 0:
                ET.SubElement(j, "mimic", joint="finger_l_joint", multiplier="1", offset="0")
            fingers.append(f"{nm}_joint")
        open_q, close_q = [th, th], [0.0, 0.0]
        grip = dict(type="rotary", g0=g0, finger_len=float(fl), open=float(th))
    tcp = [0.0, 0.0, float(top + (0.75 if D["gripper"] == "parallel" else 0.85) * fl)]
    chain = D["base_h"] + sum(D["lens"][1:]) + D["flange"] + tcp[2] - D["flange"]
    info = dict(arm_joints=joints, gripper=dict(joints=fingers, open=open_q, close=close_q), tcp_offset=tcp,
                chain_len=float(sum(D["lens"][1:]) + tcp[2]), gripper_model=grip)
    xml = ET.tostring(rob, encoding="unicode")
    return xml, info


def _down_R(flip):
    return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]) if not flip else np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]])


def validate(spec, rng, n_pts=12):
    """Find a 'gripper down' rest pose and check that the layout workspace is reachable.
    Returns (rest_q, reach_frac, info) or (None, ...) when the robot is unusable."""
    from .ik import IKSolver
    from .kinematic_mask import RobotKinematics
    from .mj_world import Layout, twin_xml
    L = Layout(spec)
    kin = RobotKinematics(xml=twin_xml(spec, L), base_body="robot_mount")
    ik = IKSolver(kin, site="tcp", joint_names=spec.arm_joints, rest_q=np.zeros(len(spec.arm_joints)),
                  rot_weight=spec.rot_weight)
    base = L.base_pos
    target = base + np.array([0.5 * L.R, 0.0, 0.3 * L.R])
    best = None
    for k in range(10):
        q0 = rng.uniform(ik.lo * 0.5, ik.hi * 0.5) if k else np.zeros(len(ik.lo))
        q0 = np.where(np.isfinite(q0), q0, 0)
        for flip in (False, True):
            q, ep, er = ik.solve(q0, target, _down_R(flip), iters=200)
            score = ep + 0.02 * er
            if best is None or score < best[0]:
                best = (score, q, ep, er)
    _, rest, ep, er = best
    if ep > 0.01 * L.R:
        return None, 0.0, dict(reason=f"rest unreachable ({ep * 1000:.0f} mm)")
    # links must stay above the table at rest
    kin.set_joints(dict(zip(spec.arm_joints, rest)))
    m, d = kin.model, kin.data
    if (d.xpos[1:, 2] < L.table_h - 0.01).any():
        return None, 0.0, dict(reason="rest pose goes under the table")
    ik.rest = rest
    lo, hi = L.ws_box
    ok = 0
    for _ in range(n_pts):
        p = rng.uniform(lo, hi)
        _, e, _ = ik.solve(rest, p, _down_R(False), iters=150)
        ok += e < 0.02 * L.R
    frac = ok / n_pts
    return (rest if frac >= 0.75 else None), frac, dict(rest_err_mm=ep * 1000, rest_rot_err_deg=float(np.rad2deg(er)),
                                                        reason="" if frac >= 0.75 else f"workspace {frac:.2f}")


def generate_robot(rng: np.random.Generator, name: str, out_root: str, max_tries=8):
    """Sample designs until one validates; write <out_root>/<name>/{robot.urdf, spec.yaml}."""
    from .urdf_robot import RobotSpec
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)
    for t in range(max_tries):
        D = sample_design(rng)
        xml, info = build_urdf(name, D)
        urdf = os.path.join(d, "robot.urdf")
        open(urdf, "w").write(xml)
        reach = 0.95 * info["chain_len"]
        spec = RobotSpec(name=name, urdf=urdf, arm_joints=info["arm_joints"], ee_body="hand",
                         rest_q=[0.0] * D["n"], reach=round(float(reach), 4), tcp_offset=info["tcp_offset"],
                         gripper=info["gripper"], rot_weight=0.5 if D["n"] >= 6 else 0.1,
                         wrist_cam=dict(side_axis=[1, 0, 0], fovy=float(rng.uniform(60, 90))),
                         default_rgba=[0.7, 0.7, 0.7, 1], asset_dir=os.path.join(d, "mj"))
        try:
            rest, frac, vinfo = validate(spec, rng)
        except Exception as e:                                     # malformed design -> resample
            rest, frac, vinfo = None, 0.0, dict(reason=f"error {e}")
        if rest is None:
            continue
        spec.rest_q = [round(float(x), 5) for x in rest]
        spec.meta = dict(procedural=True, design=D, gripper_model=info["gripper_model"], reach_frac=frac, tries=t + 1,
                         **vinfo)
        y = {k: v for k, v in spec.__dict__.items() if not k.startswith("_")}
        y["urdf"] = os.path.relpath(urdf, ROOT)
        y["asset_dir"] = os.path.relpath(spec.asset_dir, ROOT)
        yaml.safe_dump(_plain(y), open(os.path.join(d, "spec.yaml"), "w"), sort_keys=False)
        return os.path.join(d, "spec.yaml")
    return None


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x
