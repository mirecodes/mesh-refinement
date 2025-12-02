import os
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np
import pymeshlab
import trimesh

from json_handler import JsonHandler

import vedo

# ================================
# Small math / format helpers
# ================================
def _norm(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _to_str_xyz(v):
    v = np.asarray(v, dtype=float).ravel()
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"

def _mapping_get(mapping, key):
    """dict-like 안전 접근 (None 또는 키 없음 시 None)."""
    if mapping is None:
        return None
    try:
        return mapping.get(key)
    except Exception:
        try:
            return mapping[key]
        except Exception:
            return None


# ================================
# URDF helpers
# ================================
def _export_meshes(meshes: Dict[int, trimesh.Trimesh], out_dir: str, fmt: str = "stl") -> Dict[int, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for lid, mesh in meshes.items():
        path = os.path.join(out_dir, f"link_{lid}.{fmt}")
        mesh.export(path)
        paths[lid] = path
    return paths

def _compute_mass_inertia_at_com(mesh: trimesh.Trimesh, density: float):
    """Returns: mass, com(3,), inertia_3x3 at COM."""
    com = np.asarray(mesh.center_mass, dtype=float)
    try:
        vol = float(mesh.volume)
    except Exception:
        vol = 0.0
    mass = float(density) * vol

    T = np.eye(4); T[:3, 3] = -com
    moved = mesh.copy()
    try:
        moved.apply_transform(T)
    except Exception:
        pass

    I_attr = getattr(moved, "moment_inertia", None)
    if callable(I_attr):
        try:
            I_com = I_attr(density=density)
        except TypeError:
            I_com = I_attr(mass=mass)
    elif isinstance(I_attr, np.ndarray):
        I_com = np.asarray(I_attr, dtype=float) * float(density)
    else:
        mp = getattr(moved, "mass_properties", None)
        I_com = np.asarray(mp["inertia"], dtype=float) if isinstance(mp, dict) and "inertia" in mp else np.zeros((3, 3), float)
    return mass, com, I_com

def _urdf_inertia_dict(I):
    return {
        "ixx": float(I[0, 0]), "ixy": float(I[0, 1]), "ixz": float(I[0, 2]),
        "iyy": float(I[1, 1]), "iyz": float(I[1, 2]), "izz": float(I[2, 2]),
    }

def _build_tree(joints):
    """child당 하나의 parent만 사용한다고 가정."""
    parent_of, joint_by_child, children_of = {}, {}, {}
    for j in joints:
        p = int(j["parent"]); c = int(j["child"])
        if c in parent_of:
            continue
        parent_of[c] = p
        joint_by_child[c] = j
        children_of.setdefault(p, []).append(c)
    return parent_of, joint_by_child, children_of

def _gather_link_frame_origins(joints, base_origin=np.zeros(3)):
    """link_id -> world에서의 link 프레임 원점(=해당 조인트 origin). base(0)는 base_origin."""
    base_origin = np.asarray(base_origin, float)
    link_o = {0: base_origin}
    for j in joints:
        c = int(j["child"])
        o = np.asarray(j["axis"]["origin"], dtype=float)
        link_o[c] = o
        p = int(j["parent"])
        link_o.setdefault(p, base_origin)
    return link_o

def generate_urdf(
    closed_parts: Dict[int, trimesh.Trimesh],
    joints: List[dict],
    *,
    out_dir: str,
    urdf_name: str = "robot",
    density: float = 1000.0,
    mesh_fmt: str = "stl",
    base_origin=(0.0, 0.0, 0.0),
):
    os.makedirs(out_dir, exist_ok=True)
    mesh_paths = _export_meshes(closed_parts, os.path.join(out_dir, "meshes"), fmt=mesh_fmt)

    parent_of, joint_by_child, children_of = _build_tree(joints)
    link_ids = sorted(closed_parts.keys())
    link_frame_origin = _gather_link_frame_origins(joints, base_origin=np.asarray(base_origin, float))

    robot = ET.Element("robot", name=urdf_name)

    # base link (mass 0)
    base_link = ET.SubElement(robot, "link", name="base")
    inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0.0")
    ET.SubElement(inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")

    # links
    for lid in link_ids:
        link = ET.SubElement(robot, "link", name=f"link_{lid}")
        mesh = closed_parts[lid]

        mass, com_world, I_com = _compute_mass_inertia_at_com(mesh, density=density)
        link_o_world = link_frame_origin.get(lid, np.zeros(3))

        # visual / collision: 메쉬는 world 좌표라고 가정 → link frame(origin=joint origin)으로 평행이동만
        rel_xyz = _to_str_xyz(np.asarray([0, 0, 0], float) - link_o_world)

        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=rel_xyz, rpy="0 0 0")
        geom_v = ET.SubElement(visual, "geometry")
        ET.SubElement(geom_v, "mesh", filename=os.path.relpath(mesh_paths[lid], out_dir))

        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=rel_xyz, rpy="0 0 0")
        geom_c = ET.SubElement(collision, "geometry")
        ET.SubElement(geom_c, "mesh", filename=os.path.relpath(mesh_paths[lid], out_dir))

        inertial = ET.SubElement(link, "inertial")
        com_in_link = com_world - link_o_world
        ET.SubElement(inertial, "origin", xyz=_to_str_xyz(com_in_link), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
        I = _urdf_inertia_dict(I_com)
        ET.SubElement(inertial, "inertia",
                      ixx=f"{I['ixx']:.9g}", ixy=f"{I['ixy']:.9g}", ixz=f"{I['ixz']:.9g}",
                      iyy=f"{I['iyy']:.9g}", iyz=f"{I['iyz']:.9g}", izz=f"{I['izz']:.9g}")

    # joints
    for child, j in joint_by_child.items():
        parent = int(j["parent"])
        parent_name = "base" if parent == 0 else f"link_{parent}"
        child_name = f"link_{child}"

        # joint origin (parent 프레임 기준)
        joint_world_o = np.asarray(j["axis"]["origin"], dtype=float)
        parent_world_o = link_frame_origin.get(parent, np.zeros(3))
        joint_in_parent = joint_world_o - parent_world_o

        # ----- 타입 매핑 -----
        jtype_raw = str(_mapping_get(j, "type") or "revolute").strip().lower()
        if jtype_raw.startswith("rev"):
            urdf_type = "revolute"
        elif jtype_raw.startswith("pri"):
            urdf_type = "prismatic"
        elif jtype_raw.startswith("pla"):
            urdf_type = "planar"  # URDF에 존재 (면의 법선 = axis)
        elif jtype_raw.startswith("sph"):
            urdf_type = "floating"  # URDF에는 spherical 없음 → floating으로 대체
        else:
            urdf_type = "revolute"  # 폴백

        # 공통 joint 노드
        j_el = ET.SubElement(robot, "joint", name=f"joint_{parent}_{child}", type=urdf_type)
        ET.SubElement(j_el, "parent", link=parent_name)
        ET.SubElement(j_el, "child", link=child_name)
        ET.SubElement(j_el, "origin", xyz=_to_str_xyz(joint_in_parent), rpy="0 0 0")

        # ----- axis / limit 처리 -----
        # revolute / prismatic / planar 에서는 axis 사용
        if urdf_type in ("revolute", "prismatic", "planar"):
            n_raw = _mapping_get(j, "axis")
            n_vec = _mapping_get(n_raw, "n") if isinstance(n_raw, dict) else None
            axis = _norm(n_vec if n_vec is not None else [0, 0, 1])
            ET.SubElement(j_el, "axis", xyz=_to_str_xyz(axis))

        # limit은 revolute / prismatic만 지정
        if urdf_type in ("revolute", "prismatic"):
            # 필요 시 값 튜닝
            ET.SubElement(j_el, "limit", lower="-1.57", upper="1.57", effort="10.0", velocity="1.0")

    # save
    tree = ET.ElementTree(robot)
    urdf_path = os.path.join(out_dir, f"{urdf_name}.urdf")
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path


# ================================
# Stage: articulate (minimal)
# ================================
def stage_articulate(cfgs):
    # Load states
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)

    # Load meshes
    dir_mesh = states.segment.dirs.mesh
    links: Dict[int, trimesh.Trimesh] = {}
    for k, path in dir_mesh.items():
        lid = int(k)
        try:
            mesh = trimesh.load(path, process=False)
        except Exception as e:
            print(f"[error]: Failed to load mesh for link {lid}: {e}")
            continue
        links[lid] = mesh

    if not links:
        print("[error]: No link meshes found in states.segment.dirs.mesh")
        return

    # Load Joints
    joints: List[dict] = states.segment.joints

    # Generate URDF File
    urdf_dir = generate_urdf(
        closed_parts=links,
        joints=joints,
        out_dir=cfgs.urdf_out_dir,
        urdf_name="object",
        density=1000.0,   # 필요시 변경
        mesh_fmt="stl",
    )



    print(f"[info]: the urdf file has been generated in: {urdf_dir}")

    # Vedo viewer of the articulated object
    actors = [vedo.Mesh([m.vertices, m.faces]).c('lightblue').alpha(0.5) for _, m in sorted(links.items())]
    vedo.show(actors, "Loaded link meshes", axes=1).close()