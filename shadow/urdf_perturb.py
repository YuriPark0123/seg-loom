"""'Approximate URDF' generator: the world keeps the true robot, the model is given a perturbed copy.

Perturbations (each recorded so the error is known for evaluation):
  length   : every joint origin translation scaled by (1 + e), |e| <= len_frac      (CAD vs real)
  zero     : origin rotated about the joint axis by d, |d| <= zero_deg            (encoder zero offset)
  shape    : primitive sizes scaled / cylinder<->box swapped; meshes replaced by their bounding box
  drop     : visuals of some links removed (e.g. fingers missing in the URDF)
Meshes referenced by the result are copied next to it (relative paths), so the output directory is
self-contained and can be moved to another machine.
"""
from __future__ import annotations

import copy
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .urdf_robot import _resolve_mesh


def _vec(s, n=3):
    return np.array([float(x) for x in s.split()]) if s else np.zeros(n)


def _f(v):
    return " ".join(f"{float(x):.7g}" for x in v)


def localize_meshes(root: ET.Element, urdf_dir: str, package_dirs: dict, out_dir: str):
    """Copy every mesh into out_dir/meshes and rewrite filenames to relative paths."""
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    seen = {}
    for m in root.iter("mesh"):
        src = _resolve_mesh(m.get("filename"), urdf_dir, package_dirs)
        if src not in seen:
            base = os.path.basename(src)
            dst = os.path.join("meshes", f"{len(seen):03d}_{base}")
            try:                                   # hardlink: many instances of one robot share the mesh bytes (tar stores them once)
                os.link(src, os.path.join(out_dir, dst))
            except OSError:
                shutil.copy(src, os.path.join(out_dir, dst))
            seen[src] = dst
        m.set("filename", seen[src])


def _mesh_box(mesh_el, out_dir):
    """Bounding box (center, size) of a mesh in its own frame (scale applied)."""
    import trimesh
    mesh = trimesh.load(os.path.join(out_dir, mesh_el.get("filename")), force="mesh")
    sc = _vec(mesh_el.get("scale"), 3) if mesh_el.get("scale") else np.ones(3)
    lo, hi = mesh.bounds[0] * sc, mesh.bounds[1] * sc
    return (lo + hi) / 2, np.maximum(hi - lo, 1e-3)


def perturb_urdf(true_root: ET.Element, out_dir: str, rng: np.random.Generator, level=1.0,
                 len_frac=0.10, zero_deg=5.0, shape_frac=0.15, p_mesh_box=0.3, p_swap=0.2, p_drop=0.1,
                 protect=()):
    """true_root: parsed URDF whose mesh filenames are relative to out_dir (see localize_meshes).
    Writes out_dir/approx.urdf, returns the record of applied errors."""
    root = copy.deepcopy(true_root)
    rec = dict(length={}, zero_deg={}, shape={}, dropped=[], level=level, len_frac=len_frac)
    for j in root.findall("joint"):
        o = j.find("origin")
        if o is None:
            o = ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
        xyz, rpy = _vec(o.get("xyz")), _vec(o.get("rpy"))
        if np.linalg.norm(xyz) > 1e-6:
            e = float(np.clip(rng.normal(0, len_frac / 2), -len_frac, len_frac) * level)
            xyz = xyz * (1 + e)
            rec["length"][j.get("name")] = e
        if j.get("type") in ("revolute", "continuous") and j.find("mimic") is None:
            ax = _vec(j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0")
            d = float(rng.uniform(-zero_deg, zero_deg) * level)
            R = Rotation.from_euler("xyz", rpy) * Rotation.from_rotvec(np.deg2rad(d) * ax / np.linalg.norm(ax))
            rpy = R.as_euler("xyz")
            rec["zero_deg"][j.get("name")] = d
        o.set("xyz", _f(xyz)); o.set("rpy", _f(rpy))
    links = [l for l in root.findall("link") if l.find("visual") is not None]
    for link in links:
        nm = link.get("name")
        if nm not in protect and rng.random() < p_drop * level:
            for v in link.findall("visual"):
                link.remove(v)
            rec["dropped"].append(nm)
            continue
        for k, v in enumerate(link.findall("visual")):
            g = v.find("geometry")
            shp = list(g)[0]
            key = f"{nm}/{k}"
            if shp.tag == "mesh":
                if rng.random() < p_mesh_box * level:
                    c, size = _mesh_box(shp, out_dir)
                    o = v.find("origin")
                    if o is None:
                        o = ET.SubElement(v, "origin", xyz="0 0 0", rpy="0 0 0")
                    R = Rotation.from_euler("xyz", _vec(o.get("rpy")))
                    o.set("xyz", _f(_vec(o.get("xyz")) + R.apply(c)))
                    g.remove(shp); ET.SubElement(g, "box", size=_f(size))
                    rec["shape"][key] = "mesh->box"
                continue
            s = 1 + float(np.clip(rng.normal(0, shape_frac / 2), -shape_frac, shape_frac) * level)
            if shp.tag == "cylinder" and rng.random() < p_swap * level:
                r, L = float(shp.get("radius")), float(shp.get("length"))
                g.remove(shp); ET.SubElement(g, "box", size=_f([2 * r * s, 2 * r * s, L]))
                rec["shape"][key] = f"cyl->box x{s:.3f}"
            elif shp.tag == "box" and rng.random() < p_swap * level:
                sz = _vec(shp.get("size"))
                g.remove(shp); ET.SubElement(g, "cylinder", radius=f"{max(sz[0], sz[1]) / 2 * s:.5g}", length=f"{sz[2]:.5g}")
                rec["shape"][key] = f"box->cyl x{s:.3f}"
            elif shp.tag == "cylinder":
                shp.set("radius", f"{float(shp.get('radius')) * s:.5g}")
                rec["shape"][key] = f"radius x{s:.3f}"
            elif shp.tag == "box":
                shp.set("size", _f(_vec(shp.get("size")) * s))
                rec["shape"][key] = f"box x{s:.3f}"
            elif shp.tag == "sphere":
                shp.set("radius", f"{float(shp.get('radius')) * s:.5g}")
                rec["shape"][key] = f"sphere x{s:.3f}"
    for c in root.findall("mujoco"):
        root.remove(c)
    ET.ElementTree(root).write(os.path.join(out_dir, "approx.urdf"))
    return rec


def export_robot(spec, out_dir: str, rng: np.random.Generator, level=1.0, len_frac=0.10):
    """Self-contained robot folder for the dataset: true.urdf + approx.urdf + meshes + record."""
    os.makedirs(out_dir, exist_ok=True)
    root = ET.parse(spec.urdf).getroot()
    for c in root.findall("mujoco"):
        root.remove(c)
    localize_meshes(root, os.path.dirname(spec.urdf), spec.package_dirs, out_dir)
    ET.ElementTree(root).write(os.path.join(out_dir, "true.urdf"))
    protect = set()                                    # never drop the base: something must anchor the robot
    links = root.findall("link")
    if links:
        protect.add(links[0].get("name"))
    rec = perturb_urdf(root, out_dir, rng, level=level, protect=protect, len_frac=len_frac)
    return rec
