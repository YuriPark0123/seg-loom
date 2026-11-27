"""URDF -> MuJoCo robot model, driven by a small per-robot YAML spec.

Spec (configs/robots/<name>.yaml):
  name, urdf, package_dirs{pkg: dir}
  arm_joints [..]                       joints the IK moves
  gripper: {joints: [..], open: [..], close: [..]}   (optional)
  ee_body, tcp_offset [x,y,z]           grasp point (TCP) in the ee body frame
  rest_q [..]                           home pose of arm_joints
  reach                                 rough reach (m): scales cameras / table / board / IK box
  wrist_cam: {body, pos, euler_deg, fovy} (optional; OpenCV-style camera looks along +z of that frame)
  base_pos_rel [x,y] (optional)         base placement relative to the table's robot-side edge
  rot_weight                            IK orientation weight (small for <6-DOF arms)

Everything downstream (scene, twin, IK, skeleton, calibration, web UI) reads this spec, so a new
robot needs only its URDF + one YAML file.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE_MESH = {".stl", ".obj", ".msh"}


@dataclass
class RobotSpec:
    name: str
    urdf: str
    arm_joints: list
    ee_body: str
    rest_q: list
    reach: float
    tcp_offset: list = field(default_factory=lambda: [0, 0, 0])
    package_dirs: dict = field(default_factory=dict)
    gripper: dict | None = None
    wrist_cam: dict | None = None
    base_pos_rel: list = field(default_factory=lambda: [0.0, 0.0])
    rot_weight: float = 0.5
    default_rgba: list = field(default_factory=lambda: [0.75, 0.75, 0.78, 1.0])
    asset_dir: str | None = None                 # converted-model cache dir (default assets/robots/<name>)
    meta: dict | None = None                     # free-form info (procedural generator params, ...)

    @staticmethod
    def load(path_or_name: str) -> "RobotSpec":
        p = path_or_name if path_or_name.endswith(".yaml") else os.path.join(ROOT, "configs/robots", f"{path_or_name}.yaml")
        d = yaml.safe_load(open(p))
        base = os.path.dirname(os.path.abspath(p))
        res = lambda x: x if os.path.isabs(x) else os.path.normpath(os.path.join(ROOT, x))
        d["urdf"] = res(d["urdf"])
        d["package_dirs"] = {k: res(v) for k, v in (d.get("package_dirs") or {}).items()}
        if d.get("asset_dir"):
            d["asset_dir"] = res(d["asset_dir"])
        return RobotSpec(**d)

    @property
    def gripper_joints(self):
        return list(self.gripper["joints"]) if self.gripper else []

    @property
    def out_dir(self):
        return self.asset_dir or os.path.join(ROOT, "assets/robots", self.name)


def _resolve_mesh(fn: str, urdf_dir: str, package_dirs: dict) -> str:
    if fn.startswith("package://"):
        pkg, rel = fn[len("package://"):].split("/", 1)
        if pkg not in package_dirs:
            raise KeyError(f"package '{pkg}' not in spec.package_dirs (mesh {fn})")
        return os.path.join(package_dirs[pkg], rel)
    if fn.startswith("file://"):
        return fn[len("file://"):]
    return fn if os.path.isabs(fn) else os.path.join(urdf_dir, fn)


def _convert_mesh(src: str, dst_noext: str) -> str:
    ext = os.path.splitext(src)[1].lower()
    if ext in NATIVE_MESH:
        dst = dst_noext + ext
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        return dst
    import trimesh                                         # .dae / .ply / .glb ... -> .obj
    dst = dst_noext + ".obj"
    if not os.path.exists(dst):
        m = trimesh.load(src, force="mesh")
        m.export(dst)
    return dst


def urdf_to_mjcf(spec: RobotSpec, rebuild=False) -> str:
    """Convert the URDF once (meshes copied/converted into assets/robots/<name>/meshes) and
    return the robot-only MJCF xml string (absolute meshdir)."""
    out = spec.out_dir
    cache = os.path.join(out, "robot.xml")
    key = hashlib.md5((open(spec.urdf, "rb").read() + repr(spec.package_dirs).encode())).hexdigest()
    keyf = os.path.join(out, "robot.key")
    if not rebuild and os.path.exists(cache) and os.path.exists(keyf) and open(keyf).read() == key:
        return open(cache).read()

    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)
    root = ET.parse(spec.urdf).getroot()
    urdf_dir = os.path.dirname(spec.urdf)
    for link in root.findall("link"):
        for kind in ("visual", "collision"):
            for k, el in enumerate(link.findall(kind)):
                m = el.find("geometry/mesh")
                if m is None:
                    continue
                src = _resolve_mesh(m.get("filename"), urdf_dir, spec.package_dirs)
                dst = _convert_mesh(src, os.path.join(out, "meshes", f"{kind[:3]}_{link.get('name').replace('/', '_')}_{k}"))
                m.set("filename", os.path.basename(dst))
    # named materials defined once and reused by name only -> inline the colors
    named = {}
    for mat in root.iter("material"):
        c = mat.find("color")
        if mat.get("name") and c is not None:
            named[mat.get("name")] = c.get("rgba")
    for vis in root.iter("visual"):
        mat = vis.find("material")
        if mat is None:
            mat = ET.SubElement(vis, "material", name="")
        if mat.find("color") is None:
            ET.SubElement(mat, "color", rgba=named.get(mat.get("name"), " ".join(map(str, spec.default_rgba))))
    # MuJoCo compiler options: keep fixed links as bodies (tool frames!), keep visual geoms
    for old in root.findall("mujoco"):
        root.remove(old)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", meshdir=os.path.join(out, "meshes"), balanceinertia="true",
                  discardvisual="false", fusestatic="false", strippath="false")
    tmp = os.path.join(out, "robot_mj.urdf")
    ET.ElementTree(root).write(tmp)
    model = mujoco.MjModel.from_xml_path(tmp)
    raw = os.path.join(out, "robot_raw.xml")
    mujoco.mj_saveLastXML(raw, model)

    # ---- post-process the MJCF
    x = ET.parse(raw).getroot()
    x.find("compiler").set("meshdir", os.path.join(out, "meshes"))
    # URDF visual geoms come in as contype=conaffinity=0 (non-colliding); put them in group 1, collision in 0
    for g in x.iter("geom"):
        is_visual = g.get("contype") == "0" and g.get("conaffinity") == "0"
        g.set("group", "1" if is_visual else "0")
    # mimic joints -> equality constraints
    eq = None
    for j in root.findall("joint"):
        mim = j.find("mimic")
        if mim is not None:
            eq = eq if eq is not None else (x.find("equality") or ET.SubElement(x, "equality"))
            ET.SubElement(eq, "joint", joint1=j.get("name"), joint2=mim.get("joint"),
                          polycoef=f"{mim.get('offset', '0')} {mim.get('multiplier', '1')} 0 0 0")
    # TCP site on the ee body
    ee = next(b for b in x.iter("body") if b.get("name") == spec.ee_body)
    ET.SubElement(ee, "site", name="tcp", pos=" ".join(map(str, spec.tcp_offset)), size="0.005", group="5")
    xml = ET.tostring(x, encoding="unicode")
    open(cache, "w").write(xml)
    open(keyf, "w").write(key)
    return xml


def robot_root_children(robot_xml: str):
    """(assets, worldbody children, equality, contact, default) of a robot MJCF, for merging."""
    x = ET.fromstring(robot_xml)
    return x
