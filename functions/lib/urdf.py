import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
from collections import defaultdict
from functions.lib.geometry import normalize

# ================================
# Small math / format helpers
# ================================
def to_str_xyz(v):
    v = np.asarray(v, dtype=float).ravel()
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"

def mapping_get(mapping, key):
    """dict-like safe access."""
    if mapping is None:
        return None
    try:
        return mapping.get(key)
    except Exception:
        try:
            return mapping[key]
        except Exception:
            return None


# =============================================================================
# Joint building (from process steps)
# =============================================================================
def build_joints_from_rlps(rlps: List[dict]) -> List[dict]:
    """
    Group selected vectors by (link pair, origin) and synthesize joints:
      - mixed types at same origin -> error
      - >=2 Revolute -> Spherical
      - >=2 Prismatic -> Planar
      - else pass-through
    """
    picked = []
    for rlp in rlps:
        p = int(rlp.get("a", {}).get("parent", rlp.get("parent", -1)))
        c = int(rlp.get("a", {}).get("child",  rlp.get("child",  -1)))
        if p < 0 or c < 0 or p == c:
            continue
        for v in (rlp.get("vectors") or []):
            st = v.get("state")
            if st not in ("Revolute", "Prismatic"):
                continue
            o = np.asarray(v.get("center", rlp.get("center", [0,0,0])), dtype=float).ravel()
            n = np.asarray(v.get("n") if v.get("n") is not None else v.get("n_poly"), dtype=float).ravel()
            if o.size != 3 or n.size != 3:
                continue
            n = normalize(n)
            if not np.isfinite(o).all() or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                continue
            i, j = (p, c) if p < c else (c, p)
            picked.append({"pair": (i, j), "origin": o, "n": n, "type": st})

    if not picked:
        return []

    O = np.vstack([x["origin"] for x in picked])
    bb = O.max(axis=0) - O.min(axis=0)
    diag = float(np.linalg.norm(bb))
    eps = max(diag * 1e-4, 1e-8)

    def _cluster_origins(items):
        groups = []
        centers = []
        for k, it in enumerate(items):
            o = it["origin"]
            assigned = False
            for gi, cen in enumerate(centers):
                if np.linalg.norm(o - cen) <= eps:
                    groups[gi].append(k)
                    centers[gi] = centers[gi] + (o - centers[gi]) / float(len(groups[gi]))
                    assigned = True
                    break
            if not assigned:
                groups.append([k])
                centers.append(o.copy())
        return groups, np.vstack(centers) if centers else np.zeros((0,3))

    by_pair: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for idx, it in enumerate(picked):
        by_pair[it["pair"]].append(idx)

    joints = []
    for pair, idxs in by_pair.items():
        sub = [picked[k] for k in idxs]
        groups, group_centers = _cluster_origins(sub)
        for g_idx, g in enumerate(groups):
            items = [sub[k] for k in g]
            origin = group_centers[g_idx]
            types = {it["type"] for it in items}
            if len(types) > 1:
                raise ValueError(f"Mixed joint types at same origin for link {pair}: {sorted(list(types))}")
            jtype = next(iter(types))
            dirs = np.vstack([it["n"] for it in items])

            if jtype == "Revolute" and len(items) >= 2:
                n_rep = dirs[0]
                axes = [d.tolist() for d in dirs[:2]]
                joints.append({
                    "parent": pair[0], "child": pair[1],
                    "type": "Spherical",
                    "axis": {"origin": origin.astype(float).tolist(),
                             "n": n_rep.astype(float).tolist()},
                    "axes": axes
                })
                continue

            if jtype == "Prismatic" and len(items) >= 2:
                _, _, Vt = np.linalg.svd(dirs, full_matrices=False)
                n_plane = normalize(Vt[-1])
                u0 = dirs[0] - np.dot(dirs[0], n_plane) * n_plane
                if np.linalg.norm(u0) < 1e-12:
                    for k in range(1, dirs.shape[0]):
                        u0 = dirs[k] - np.dot(dirs[k], n_plane) * n_plane
                        if np.linalg.norm(u0) >= 1e-12:
                            break
                u = normalize(u0)
                v = normalize(np.cross(n_plane, u))
                joints.append({
                    "parent": pair[0], "child": pair[1],
                    "type": "Planar",
                    "axis": {"origin": origin.astype(float).tolist(),
                             "n": n_plane.astype(float).tolist()},
                    "plane": {"n": n_plane.astype(float).tolist(),
                              "u": u.astype(float).tolist(),
                              "v": v.astype(float).tolist()}
                })
                continue

            n = dirs[0]
            joints.append({
                "parent": pair[0], "child": pair[1],
                "axis": {"origin": origin.astype(float).tolist(),
                         "n": n.astype(float).tolist()},
                "type": jtype
            })

    return joints


# ================================
# URDF helpers
# ================================
def export_meshes(meshes: Dict[int, trimesh.Trimesh], out_dir: str, fmt: str = "stl") -> Dict[int, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for lid, mesh in meshes.items():
        path = os.path.join(out_dir, f"link_{lid}.{fmt}")
        mesh.export(path)
        paths[lid] = path
    return paths

def compute_mass_inertia_at_com(mesh: trimesh.Trimesh, density: float):
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

def urdf_inertia_dict(I):
    return {
        "ixx": float(I[0, 0]), "ixy": float(I[0, 1]), "ixz": float(I[0, 2]),
        "iyy": float(I[1, 1]), "iyz": float(I[1, 2]), "izz": float(I[2, 2]),
    }

def build_tree(joints):
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

def gather_link_frame_origins(joints, base_origin=np.zeros(3), transform_matrix=None):
    """link_id -> world에서의 link 프레임 원점(=해당 조인트 origin). base(0)는 base_origin."""
    base_origin = np.asarray(base_origin, float)
    
    # Apply transform to base_origin if needed
    if transform_matrix is not None:
        # Transform point: p' = T * p
        # Homogeneous coords
        p_h = np.append(base_origin, 1.0)
        p_trans = np.dot(transform_matrix, p_h)
        base_origin = p_trans[:3] / p_trans[3]

    link_o = {0: base_origin}
    for j in joints:
        c = int(j["child"])
        o = np.asarray(j["axis"]["origin"], dtype=float)
        
        # Apply transform to joint origin if needed
        if transform_matrix is not None:
            p_h = np.append(o, 1.0)
            p_trans = np.dot(transform_matrix, p_h)
            o = p_trans[:3] / p_trans[3]
            
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
    transform_matrix: np.ndarray = None, # Added parameter
):
    os.makedirs(out_dir, exist_ok=True)
    
    # Export meshes (WITHOUT applying transform, as requested)
    mesh_paths = export_meshes(closed_parts, os.path.join(out_dir, "meshes"), fmt=mesh_fmt)

    parent_of, joint_by_child, children_of = build_tree(joints)
    link_ids = sorted(closed_parts.keys())
    
    # Gather link origins (applying transform if provided)
    link_frame_origin = gather_link_frame_origins(joints, base_origin=np.asarray(base_origin, float), transform_matrix=transform_matrix)

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
        
        # Compute mass/inertia on the transformed mesh if needed
        # Or compute on original and rotate inertia? 
        # Easier to transform mesh temporarily for calculation if transform is rigid.
        mesh_for_calc = mesh.copy()
        if transform_matrix is not None:
            mesh_for_calc.apply_transform(transform_matrix)

        mass, com_world, I_com = compute_mass_inertia_at_com(mesh_for_calc, density=density)
        link_o_world = link_frame_origin.get(lid, np.zeros(3))

        # visual / collision: 메쉬는 world 좌표라고 가정 → link frame(origin=joint origin)으로 평행이동만
        # If transform_matrix is applied, mesh_paths point to transformed meshes, and link_o_world is transformed.
        # So relative position calculation remains valid in the new frame.
        rel_xyz = to_str_xyz(np.asarray([0, 0, 0], float) - link_o_world)

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
        ET.SubElement(inertial, "origin", xyz=to_str_xyz(com_in_link), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
        I = urdf_inertia_dict(I_com)
        ET.SubElement(inertial, "inertia",
                      ixx=f"{I['ixx']:.9g}", ixy=f"{I['ixy']:.9g}", ixz=f"{I['ixz']:.9g}",
                      iyy=f"{I['iyy']:.9g}", iyz=f"{I['iyz']:.9g}", izz=f"{I['izz']:.9g}")

    # joints
    for child, j in joint_by_child.items():
        parent = int(j["parent"])
        parent_name = "base" if parent == 0 else f"link_{parent}"
        child_name = f"link_{child}"

        # joint origin (parent 프레임 기준)
        # j["axis"]["origin"] is in original world frame (before transform_matrix application inside this function)
        # We need to transform it if transform_matrix is present.
        joint_world_o = np.asarray(j["axis"]["origin"], dtype=float)
        if transform_matrix is not None:
            p_h = np.append(joint_world_o, 1.0)
            p_trans = np.dot(transform_matrix, p_h)
            joint_world_o = p_trans[:3] / p_trans[3]
            
        parent_world_o = link_frame_origin.get(parent, np.zeros(3))
        joint_in_parent = joint_world_o - parent_world_o

        # ----- 타입 매핑 -----
        jtype_raw = str(mapping_get(j, "type") or "revolute").strip().lower()
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
        ET.SubElement(j_el, "origin", xyz=to_str_xyz(joint_in_parent), rpy="0 0 0")

        # ----- axis / limit 처리 -----
        # revolute / prismatic / planar 에서는 axis 사용
        if urdf_type in ("revolute", "prismatic", "planar"):
            n_raw = mapping_get(j, "axis")
            n_vec = mapping_get(n_raw, "n") if isinstance(n_raw, dict) else None
            axis = normalize(n_vec if n_vec is not None else [0, 0, 1])
            
            # Apply rotation to axis if transform_matrix is present
            if transform_matrix is not None:
                # Transform vector: v' = R * v
                R = transform_matrix[:3, :3]
                axis = normalize(np.dot(R, axis))

            ET.SubElement(j_el, "axis", xyz=to_str_xyz(axis))

        # limit은 revolute / prismatic만 지정
        if urdf_type in ("revolute", "prismatic"):
            # 필요 시 값 튜닝
            ET.SubElement(j_el, "limit", lower="-1.57", upper="1.57", effort="10.0", velocity="1.0")

    # Add base_joint (fixed) connecting base to link_1
    # This ensures the robot is rooted at the base link
    if 1 in link_ids:
        base_joint = ET.SubElement(robot, "joint", name="base_joint", type="fixed")
        ET.SubElement(base_joint, "parent", link="base")
        ET.SubElement(base_joint, "child", link="link_1")
        
        # Origin of link_1 in world frame (which is base frame)
        link1_origin = link_frame_origin.get(1, np.zeros(3))
        ET.SubElement(base_joint, "origin", xyz=to_str_xyz(link1_origin), rpy="0 0 0")

    # save
    tree = ET.ElementTree(robot)
    urdf_path = os.path.join(out_dir, f"{urdf_name}.urdf")
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path
