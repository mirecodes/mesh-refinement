# -*- coding: utf-8 -*-
"""
Lightweight URDF parser + FK (no external URDF libs).
- xml.etree.ElementTree 로 파싱
- rpy(origin), axis, limit(lower/upper) 지원
- joint types: fixed, revolute, continuous, prismatic
- FK: 링크별 월드변환 4x4 반환
- 미지원(간단화): mimic, calibration, dynamics, safety_controller 등

Author: you
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import xml.etree.ElementTree as ET


# ----------------------------- math helpers ----------------------------------


def _inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy -> R. Compose as Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cx, cy, cz = math.cos(roll), math.cos(pitch), math.cos(yaw)
    sx, sy, sz = math.sin(roll), math.sin(pitch), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _T_from_xyz_rpy(xyz: Tuple[float, float, float],
                    rpy: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_to_R(*rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def _axis_angle_R(axis: np.ndarray, theta: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s, C = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
    ], dtype=np.float64)


def _T_rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _axis_angle_R(axis, theta)
    return T


def _T_trans_axis(axis: np.ndarray, d: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = axis / (np.linalg.norm(axis) + 1e-12) * d
    return T


# ----------------------------- data models -----------------------------------


@dataclass
class JointSpec:
    name: str
    joint_type: str                      # fixed | revolute | prismatic | continuous
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: np.ndarray                     # (3,)
    limit_lower: Optional[float] = None  # radians / meters
    limit_upper: Optional[float] = None  # radians / meters


@dataclass
class URDFModel:
    links: List[str]
    joints: Dict[str, JointSpec]
    # graph helpers
    parent_to_children: Dict[str, List[str]]   # parent link -> list(child link)
    child_to_joint: Dict[str, str]             # child link -> joint name
    roots: List[str]                           # links with no parent joint


# ------------------------------- parser --------------------------------------

class URDFParser:
    """
    Usage:
        parser = URDFParser.from_file("path/to/object.urdf")
        T_link = parser.link_transforms(q={"joint_a": 0.1, ...})  # {link: 4x4}
    """

    def __init__(self, model: URDFModel):
        self.model = model

    # ---- construction ----

    @classmethod
    def from_file(cls, urdf_path: str) -> "URDFParser":
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # links
        links = [el.attrib["name"] for el in root.findall("link") if "name" in el.attrib]

        # joints
        joints: Dict[str, JointSpec] = {}
        parent_to_children: Dict[str, List[str]] = {}
        child_to_joint: Dict[str, str] = {}

        for j in root.findall("joint"):
            name = j.attrib["name"]
            jtype = j.attrib.get("type", "fixed").lower()

            parent_el = j.find("parent")
            child_el = j.find("child")
            if parent_el is None or child_el is None:
                raise ValueError(f"joint '{name}' missing parent/child")

            parent = parent_el.attrib["link"]
            child = child_el.attrib["link"]

            # origin
            xyz, rpy = _parse_origin(j.find("origin"))

            # axis
            axis = _parse_axis(j)

            # limits
            lo, hi = _parse_limit(j)  # None when absent

            spec = JointSpec(
                name=name,
                joint_type=jtype,
                parent=parent,
                child=child,
                origin_xyz=xyz,
                origin_rpy=rpy,
                axis=axis,
                limit_lower=lo,
                limit_upper=hi,
            )
            joints[name] = spec

            parent_to_children.setdefault(parent, []).append(child)
            child_to_joint[child] = name

        # roots = links with no parent joint
        parents = set(child_to_joint.keys())
        roots = [L for L in links if L not in parents]

        model = URDFModel(
            links=links,
            joints=joints,
            parent_to_children=parent_to_children,
            child_to_joint=child_to_joint,
            roots=roots,
        )
        parser = cls(model)
        # store XML root for later visual.origin parsing
        parser._xml_root = root
        # precompute link -> visual origin transforms
        parser._visual_T = parser._parse_visual_origins()
        return parser

    # ---- FK ----

    def link_transforms(self, q: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Compute T_world_link(4x4) for each link given q (joint->value).
        - revolute/continuous: radians
        - prismatic: meters
        - fixed: ignored
        Missing q entries default to 0.0.
        Limits are enforced for revolute/prismatic when present (continuous is unclamped).
        """
        T_world: Dict[str, np.ndarray] = {root: np.eye(4, dtype=np.float64) for root in self.model.roots}

        # BFS/DFS from roots through child links
        stack = list(self.model.roots)
        visited = set()

        while stack:
            parent = stack.pop()
            visited.add(parent)
            # children of this parent
            for child in self.model.parent_to_children.get(parent, []):
                jname = self.model.child_to_joint[child]
                js = _joint_by_child(self.model.joints, child, jname)

                # T_parent_jointOrigin
                Tpo = _T_from_xyz_rpy(js.origin_xyz, js.origin_rpy)

                # motion by q
                qval = float(q.get(js.name, 0.0))
                if js.joint_type == "revolute":
                    qval = _clamp(qval, js.limit_lower, js.limit_upper)
                    Tmot = _T_rot_axis(js.axis, qval)
                elif js.joint_type == "continuous":
                    Tmot = _T_rot_axis(js.axis, qval)  # no clamp
                elif js.joint_type == "prismatic":
                    qval = _clamp(qval, js.limit_lower, js.limit_upper)
                    Tmot = _T_trans_axis(js.axis, qval)
                elif js.joint_type == "fixed":
                    Tmot = np.eye(4, dtype=np.float64)
                else:
                    raise ValueError(f"unsupported joint type: {js.joint_type}")

                # compose
                T_world[child] = T_world[parent] @ Tpo @ Tmot
                if child not in visited:
                    stack.append(child)

        return T_world


    def parent_link_of(self, child_link: str) -> Optional[str]:
        """child_link의 부모 링크명 반환 (루트면 None)."""
        jname = self.model.child_to_joint.get(child_link)
        if not jname:
            return None
        return self.model.joints[jname].parent

    def relative_transform(self, parent_link: str, child_link: str, q: Dict[str, float]) -> np.ndarray:
        """T_parent->child (FK에서 얻어 계산)."""
        T_world = self.link_transforms(q)
        if parent_link not in T_world or child_link not in T_world:
            raise KeyError(f"relative_transform: missing link(s): {parent_link}, {child_link}")
        return _inv(T_world[parent_link]) @ T_world[child_link]

    def joint_path_to(self, link_name: str):
        """root(해당 링크의 뿌리) → link_name 으로 내려오는 JointSpec 리스트를 반환."""
        path = []
        cur = link_name
        while cur in self.model.child_to_joint:
            jname = self.model.child_to_joint[cur]
            js = self.model.joints[jname]
            path.append(js)
            cur = js.parent
        path.reverse()  # root→...→link
        return path

    def chain_T_to_link(self, link_name: str, q: Dict[str, float], *, clamp_limits: bool = True) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        for js in self.joint_path_to(link_name):
            Tpo = _T_from_xyz_rpy(js.origin_xyz, js.origin_rpy)
            qval_req = float(q.get(js.name, 0.0))

            # motion
            if js.joint_type in ("revolute", "continuous"):
                qval = qval_req
                if js.joint_type == "revolute" and clamp_limits:
                    qval = _clamp(qval, js.limit_lower, js.limit_upper)
                Tmot = _T_rot_axis(js.axis, qval)
            elif js.joint_type == "prismatic":
                qval = qval_req
                if clamp_limits:
                    qval = _clamp(qval, js.limit_lower, js.limit_upper)
                Tmot = _T_trans_axis(js.axis, qval)
            else:  # fixed
                Tmot = np.eye(4)

            T = T @ Tpo @ Tmot
        return T

    def _parse_visual_origins(self) -> Dict[str, np.ndarray]:
        """
        Build a map: link_name -> 4x4 transform for the first <visual><origin .../> on that link.
        If a link has no visual or origin, use identity.
        """
        visual_T: Dict[str, np.ndarray] = {}
        # Walk all <link> elements from the stored XML root
        for link_el in self._xml_root.findall("link"):
            lname = link_el.attrib.get("name")
            if not lname:
                continue
            vis_el = link_el.find("visual")
            if vis_el is None:
                continue
            org_el = vis_el.find("origin")
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if org_el is not None:
                xyz_str = org_el.attrib.get("xyz", "0 0 0")
                rpy_str = org_el.attrib.get("rpy", "0 0 0")
                xyz = tuple(float(x) for x in xyz_str.split())
                rpy = tuple(float(x) for x in rpy_str.split())
            visual_T[lname] = _T_from_xyz_rpy(xyz, rpy)
        return visual_T

    def visual_T(self, link_name: str) -> np.ndarray:
        """Return the 4x4 transform from link frame to visual(frame) for this link. Identity if absent."""
        return self._visual_T.get(link_name, np.eye(4, dtype=np.float64))


# --------------------------- internal utils ----------------------------------

def _parse_origin(origin_elem: Optional[ET.Element]) -> Tuple[Tuple[float, float, float],
                                                              Tuple[float, float, float]]:
    if origin_elem is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = tuple(float(x) for x in origin_elem.attrib.get("xyz", "0 0 0").split())
    rpy = tuple(float(x) for x in origin_elem.attrib.get("rpy", "0 0 0").split())
    return xyz, rpy


def _parse_axis(joint_elem: ET.Element) -> np.ndarray:
    axis_el = joint_elem.find("axis")
    if axis_el is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([float(x) for x in axis_el.attrib.get("xyz", "1 0 0").split()],
                 dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _parse_limit(joint_elem: ET.Element) -> Tuple[Optional[float], Optional[float]]:
    limit_el = joint_elem.find("limit")
    if limit_el is None:
        return None, None
    lo = limit_el.attrib.get("lower")
    hi = limit_el.attrib.get("upper")
    return (float(lo) if lo is not None else None,
            float(hi) if hi is not None else None)


def _clamp(x: float, lo: Optional[float], hi: Optional[float]) -> float:
    if (lo is not None) and (x < lo):
        return lo
    if (hi is not None) and (x > hi):
        return hi
    return x


def _joint_by_child(joints: Dict[str, JointSpec], child: str, jname: str) -> JointSpec:
    js = joints.get(jname)
    if js is None or js.child != child:
        raise KeyError(f"joint not found or mismatched for child '{child}' (joint='{jname}')")
    return js
