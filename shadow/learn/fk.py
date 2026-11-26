"""Differentiable, batched forward kinematics straight from a URDF.

    kin = URDFKinematics.from_file("robots/so101/approx.urdf")
    T = kin.fk(q, joint_names=meta["joint_names"])        # (B, L, 4, 4) link frames in the base frame

Joint types: revolute, continuous, prismatic, fixed (+ <mimic>). Link frames are the URDF link frames, which is
also what MuJoCo calls the body frame (the dataset keeps every link as a body), so the result is directly
comparable with arrays.npz["body_pose"][:, :Nb] (ordered by meta["robot_bodies"]).

Learnable corrections used by the registration refiner (M3) can be passed to fk():
  dq       (B, J)     joint zero offsets, added to every movable joint
  dorigin  (B, Jall, 3) translation added to each joint origin (link length error)
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import axis_angle_to_R, make_T, rpy_to_R

MOVABLE = ("revolute", "continuous", "prismatic")


def _vec(s, default):
    return np.array([float(x) for x in s.split()]) if s else np.array(default, float)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    mimic: tuple | None          # (master joint, multiplier, offset)
    lower: float
    upper: float


class URDFKinematics:
    def __init__(self, root: ET.Element, path: str | None = None):
        self.path = path
        self.name = root.get("name")
        self.links = [l.get("name") for l in root.findall("link")]
        self.link_el = {l.get("name"): l for l in root.findall("link")}
        joints = []
        for j in root.findall("joint"):
            o = j.find("origin")
            a = j.find("axis")
            m = j.find("mimic")
            lim = j.find("limit")
            joints.append(Joint(
                j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                _vec(o.get("xyz") if o is not None else None, [0, 0, 0]),
                _vec(o.get("rpy") if o is not None else None, [0, 0, 0]),
                _vec(a.get("xyz") if a is not None else None, [1, 0, 0]),
                (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0))) if m is not None else None,
                float(lim.get("lower", 0)) if lim is not None else 0.0,
                float(lim.get("upper", 0)) if lim is not None else 0.0))
        children = {j.child for j in joints}
        roots = [l for l in self.links if l not in children]
        assert len(roots) == 1, f"URDF must have one root link, got {roots}"
        self.root = roots[0]
        # topological order (parents before children)
        by_parent = {}
        for j in joints:
            by_parent.setdefault(j.parent, []).append(j)
        order, stack = [], [self.root]
        while stack:
            p = stack.pop()
            for j in by_parent.get(p, []):
                order.append(j)
                stack.append(j.child)
        self.joints = order
        self.joint_index = {j.name: i for i, j in enumerate(order)}
        self.movable = [j.name for j in order if j.type in MOVABLE]
        self.link_index = {n: i for i, n in enumerate(self.links)}
        axes = np.stack([j.axis / max(np.linalg.norm(j.axis), 1e-12) for j in order]) if order else np.zeros((0, 3))
        self._xyz = torch.tensor(np.stack([j.xyz for j in order]), dtype=torch.float64)
        self._rpy = torch.tensor(np.stack([j.rpy for j in order]), dtype=torch.float64)
        self._axis = torch.tensor(axes, dtype=torch.float64)

    @classmethod
    def from_file(cls, path: str) -> "URDFKinematics":
        return cls(ET.parse(path).getroot(), path)

    @property
    def urdf_dir(self):
        return os.path.dirname(self.path) if self.path else "."

    def limits(self, names):
        jj = {j.name: j for j in self.joints}
        return np.array([[jj[n].lower, jj[n].upper] for n in names])

    def fk(self, q: torch.Tensor, joint_names: list[str] | None = None, dq: torch.Tensor | None = None,
           dorigin: torch.Tensor | None = None) -> torch.Tensor:
        """q (B, J) values of `joint_names` (default self.movable). Movable joints missing from joint_names
        get 0, or their mimic master's value. Returns (B, L, 4, 4) in the order of self.links."""
        names = joint_names or self.movable
        B = q.shape[0]
        dev, dt = q.device, q.dtype
        col = {n: i for i, n in enumerate(names)}
        xyz = self._xyz.to(dev, dt).expand(B, -1, -1)
        if dorigin is not None:
            xyz = xyz + dorigin
        Rorig = rpy_to_R(self._rpy.to(dev, dt))                     # (Jall, 3, 3)
        axis = self._axis.to(dev, dt)

        def qval(jn):
            j = self.joints[self.joint_index[jn]]
            if jn in col:
                v = q[:, col[jn]]
            elif j.mimic is not None and j.mimic[0] in col:
                v = q[:, col[j.mimic[0]]] * j.mimic[1] + j.mimic[2]
            else:
                v = torch.zeros(B, device=dev, dtype=dt)
            if dq is not None and jn in col:
                v = v + dq[:, col[jn]]
            return v

        T = {self.root: torch.eye(4, device=dev, dtype=dt).expand(B, 4, 4)}
        for i, j in enumerate(self.joints):
            To = make_T(Rorig[i].expand(B, 3, 3), xyz[:, i])
            if j.type in ("revolute", "continuous"):
                Tm = make_T(axis_angle_to_R(axis[i].expand(B, 3), qval(j.name)), torch.zeros(B, 3, device=dev, dtype=dt))
                To = To @ Tm
            elif j.type == "prismatic":
                Tm = make_T(torch.eye(3, device=dev, dtype=dt).expand(B, 3, 3), axis[i] * qval(j.name)[:, None])
                To = To @ Tm
            T[j.child] = T[j.parent] @ To
        return torch.stack([T[n] for n in self.links], 1)
