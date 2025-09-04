import numpy as np
import pymeshlab
import trimesh
from json_handler import JsonHandler


# ---------- 1) open submesh 추출 (원본 faces 필터링 + 인덱스 리맵) ----------
def extract_open_submesh(verts: np.ndarray, faces: np.ndarray, link_vertices: list[int] | np.ndarray):
    """
    원본 (verts, faces)에서 link_vertices(정점 인덱스 집합)에 완전히 포함되는 faces만 골라
    open submesh로 리턴. (sub_verts, sub_faces, global_idx_of_sub_verts)
    """
    link_vertices = np.asarray(link_vertices, dtype=int).ravel()
    if link_vertices.size == 0:
        return np.zeros((0,3), float), np.zeros((0,3), int), np.zeros((0,), int)

    in_set = np.isin(faces, link_vertices)
    face_mask = np.all(in_set, axis=1)
    if not np.any(face_mask):
        return np.zeros((0,3), float), np.zeros((0,3), int), np.zeros((0,), int)

    sub_faces_glob = faces[face_mask]                          # (F',3) 글로벌 정점 인덱스
    sub_verts_idx, inv = np.unique(sub_faces_glob.ravel(), return_inverse=True)
    sub_faces = inv.reshape(-1, 3).astype(np.int64)            # 리맵된 faces
    sub_verts = verts[sub_verts_idx].astype(np.float64)        # 서브메쉬 정점 좌표
    return sub_verts, sub_faces, sub_verts_idx


# ---------- 2) voxelization + marching cubes로 watertight 리메싱 ----------
def remesh_watertight_trimesh(sub_verts: np.ndarray,
                               sub_faces: np.ndarray,
                               *,
                               pitch: float | None = None,
                               pitch_rel: float = 0.01,
                               smooth_iters: int = 0,
                               extra_points: np.ndarray | list | None = None) -> trimesh.Trimesh:
    """
    open mesh라도 voxelization → marching_cubes로 닫힌 watertight mesh를 생성.
    - pitch가 None이면 bbox 대각선 * pitch_rel 로 자동 설정.
    - smooth_iters > 0 이면 약간의 스무딩 수행(선택).
    - extra_points가 있으면 voxel 점군과 합쳐서 보정된 표면 추출 시도.
    """
    if sub_verts.size == 0 or sub_faces.size == 0:
        return trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=int), process=False)

    m = trimesh.Trimesh(vertices=sub_verts, faces=sub_faces, process=False)
    # bbox 기반 자동 해상도
    if pitch is None:
        bb = m.bounds
        diag = float(np.linalg.norm(bb[1] - bb[0]))
        if diag <= 0:
            diag = 1.0
        pitch = max(diag * pitch_rel, 1e-5)

    # voxelize and solid-fill so the volume is closed before marching cubes
    vox = m.voxelized(pitch=pitch)
    try:
        # ensure a solid occupancy (not just a surface shell)
        vox = vox.fill()
    except Exception:
        pass
    # small morphological closing to seal tiny leaks
    try:
        vox = vox.dilation(1).erosion(1)
    except Exception:
        pass

    # collect base occupied voxel centers as points
    try:
        base_pts = vox.points.copy()
    except Exception:
        base_pts = None

    # fuse with extra_points (e.g., extrapolated cap points)
    all_pts = None
    if base_pts is not None:
        all_pts = base_pts
    if extra_points is not None:
        ep = np.asarray(extra_points, dtype=float).reshape(-1, 3)
        if ep.size:
            all_pts = ep if all_pts is None else np.vstack([all_pts, ep])

    # try reconstructing from union point cloud (more robust sealing)
    closed = None
    try:
        from trimesh.voxel import ops as vops
        if all_pts is not None and all_pts.size:
            closed = vops.points_to_marching_cubes(all_pts, pitch=pitch)
    except Exception:
        closed = None

    # fallback: classic marching cubes on voxel grid
    if closed is None:
        closed = vox.marching_cubes

    # if any residual holes remain, try hole-filling as a safeguard
    try:
        if not closed.is_watertight:
            closed = closed.fill_holes()
    except Exception:
        pass

    # 선택적 스무딩
    if smooth_iters > 0 and len(closed.vertices) > 0:
        try:
            trimesh.smoothing.filter_taubin(closed, lamb=0.5, nu=-0.53, iterations=smooth_iters)
        except Exception:
            pass
    return closed


def _mapping_get(mapping, key):
    """Safe getter that works with dict-like objects (e.g., DictHandler) and plain dicts.
    Returns None if key not present or any access error occurs.
    """
    if mapping is None:
        return None
    # Try mapping.get(key) (DictHandler.get may only accept 1 arg)
    try:
        return mapping.get(key)
    except TypeError:
        # Some custom get signatures: retry without defaults already
        try:
            return mapping.get(key)
        except Exception:
            pass
    except AttributeError:
        # No get; try __getitem__
        try:
            return mapping[key]
        except Exception:
            return None
    except Exception:
        pass
    # Fallback: try __getitem__
    try:
        return mapping[key]
    except Exception:
        return None


# ---------- 3) links를 받아 각 링크별 watertight mesh 생성 ----------
def build_closed_meshes_from_links(verts: np.ndarray,
                                   faces: np.ndarray,
                                   links,
                                   *,
                                   pitch: float | None = None,
                                   pitch_rel: float = 0.01,
                                   smooth_iters: int = 0,
                                   extrapoints_map: dict | None = None):
    """
    links 포맷을 유연하게 처리:
      - list[list[int]]: index가 카테고리 id, 값이 vertex index 리스트 (보통 0은 background)
      - list[{'id': int, 'vertices': [...] }]
      - dict[int -> list[int]]
    반환: dict[int -> trimesh.Trimesh] (watertight meshes)
    """
    def _iter_links(_links):
        # list of lists
        if isinstance(_links, list) and (len(_links) == 0 or isinstance(_links[0], list)):
            for lid, lst in enumerate(_links):
                if lid == 0:        # 0은 보통 background로 스킵
                    continue
                if lst:
                    yield int(lid), lst
        # list of dicts
        elif isinstance(_links, list) and len(_links) > 0 and isinstance(_links[0], dict):
            for d in _links:
                lid = int(d.get("id", -1))
                lst = d.get("vertices", [])
                if lid >= 0 and lst:
                    yield lid, lst
        # dict
        elif isinstance(_links, dict):
            for lid, lst in _links.items():
                lid = int(lid)
                if lid == 0:
                    continue
                if lst:
                    yield lid, lst
        else:
            raise TypeError("Unsupported links format")

    out = {}
    for link_id, vlist in _iter_links(links):
        sub_v, sub_f, _ = extract_open_submesh(verts, faces, vlist)
        # gather extra points for this link if provided
        extra_pts = None
        if extrapoints_map is not None:
            # keys may be int or str (DictHandler.get often only accepts a single argument)
            extra_pts = _mapping_get(extrapoints_map, link_id)
            if extra_pts is None:
                extra_pts = _mapping_get(extrapoints_map, str(link_id))
        closed = remesh_watertight_trimesh(
            sub_v, sub_f,
            pitch=pitch, pitch_rel=pitch_rel, smooth_iters=smooth_iters,
            extra_points=extra_pts
        )
        out[int(link_id)] = closed
    return out


import os
import math
import numpy as np
import xml.etree.ElementTree as ET
import trimesh


def _norm(v):
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _to_str_xyz(v):
    v = np.asarray(v, dtype=float).ravel()
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"


def _export_meshes(closed_parts: dict[int, trimesh.Trimesh], out_dir: str, fmt: str = "stl") -> dict[int, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for lid, mesh in closed_parts.items():
        path = os.path.join(out_dir, f"link_{lid}.{fmt}")
        mesh.export(path)
        paths[lid] = path
    return paths


def _compute_mass_inertia_at_com(mesh: trimesh.Trimesh, density: float):
    """
    Returns: mass, com(3,), inertia_3x3 at COM
    """
    # center of mass (density-independent for uniform density)
    com = np.asarray(mesh.center_mass, dtype=float)

    # mass from volume and density
    try:
        vol = float(mesh.volume)
    except Exception:
        vol = 0.0
    mass = float(density) * vol

    # inertia about COM: translate mesh to COM, then compute moment_inertia
    T = np.eye(4)
    T[:3, 3] = -com
    moved = mesh.copy()
    try:
        moved.apply_transform(T)
    except Exception:
        pass
    # moment_inertia at the (translated) origin == inertia at COM
    I_attr = getattr(moved, "moment_inertia", None)
    I_com = None
    if callable(I_attr):
        # API where moment_inertia is a method
        try:
            I_com = I_attr(density=density)
        except TypeError:
            # some versions use mass instead of density
            I_com = I_attr(mass=mass)
    elif isinstance(I_attr, np.ndarray):
        # API where moment_inertia is already a 3x3 ndarray (usually unit density)
        I_com = np.asarray(I_attr, dtype=float) * float(density)
    else:
        # last resort: try mass_properties dict-like
        mp = getattr(moved, "mass_properties", None)
        if isinstance(mp, dict) and "inertia" in mp:
            I_com = np.asarray(mp["inertia"], dtype=float)
        else:
            I_com = np.zeros((3, 3), dtype=float)

    return mass, com, I_com


def _urdf_inertia_dict(I):
    # URDF inertia ordering
    return {
        "ixx": float(I[0, 0]), "ixy": float(I[0, 1]), "ixz": float(I[0, 2]),
        "iyy": float(I[1, 1]), "iyz": float(I[1, 2]),
        "izz": float(I[2, 2]),
    }


def _build_tree(joints):
    """
    joints: [{'parent': p, 'child': c, 'axis': {'origin': [..], 'n': [..]}, 'type': 'Linear'|'Revolute'}]
    Returns:
      parent_of: {child_id -> parent_id}
      joint_by_child: {child_id -> joint_entry}
      children_of: {parent_id -> [child_ids]}
    * 각 child는 하나의 부모만 갖는다고 가정(입력에서 첫 것 사용)
    """
    parent_of = {}
    joint_by_child = {}
    children_of = {}
    for j in joints:
        p = int(j["parent"]); c = int(j["child"])
        if c in parent_of:   # 이미 부모가 있으면 첫 것을 유지(필요시 해제/검증 가능)
            continue
        parent_of[c] = p
        joint_by_child[c] = j
        children_of.setdefault(p, []).append(c)
    return parent_of, joint_by_child, children_of


def _gather_link_frame_origins(joints, base_origin=np.zeros(3)):
    """
    Returns link_frame_origin: {link_id -> 3D origin (world frame)}
    - base(0) at base_origin
    - each child at its joint['axis']['origin']
    """
    link_frame_origin = {0: np.asarray(base_origin, float)}
    for j in joints:
        c = int(j["child"])
        o = np.asarray(j["axis"]["origin"], dtype=float)
        link_frame_origin[c] = o
        # 부모도 없으면 임시로 추가(부모가 조인트가 없을 수도 있으므로)
        p = int(j["parent"])
        link_frame_origin.setdefault(p, np.asarray(base_origin, float))
    return link_frame_origin


def generate_urdf(
    closed_parts: dict[int, trimesh.Trimesh],
    joints: list[dict],
    *,
    out_dir: str,
    urdf_name: str = "robot",
    density: float = 1000.0,  # kg/m^3 (물=1000), 필요시 조정
    mesh_fmt: str = "stl",
    base_origin=(0.0, 0.0, 0.0),
):
    """
    closed_parts: {link_id: trimesh.Trimesh (world coords)}
    joints: [{'parent': int, 'child': int, 'axis': {'origin': [x,y,z], 'n': [nx,ny,nz]}, 'type': 'Linear'|'Revolute'}]
    """
    os.makedirs(out_dir, exist_ok=True)
    mesh_paths = _export_meshes(closed_parts, os.path.join(out_dir, "meshes"), fmt=mesh_fmt)

    # 트리 정보
    parent_of, joint_by_child, children_of = _build_tree(joints)
    link_ids = sorted(closed_parts.keys())

    # 각 링크 프레임의 월드 원점 = 연결 조인트의 origin (base=0은 base_origin)
    link_frame_origin = _gather_link_frame_origins(joints, base_origin=np.asarray(base_origin, float))

    # XML 루트
    robot = ET.Element("robot", name=urdf_name)

    # 0) BASE LINK (질량 0)
    base_link = ET.SubElement(robot, "link", name="base")
    inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0.0")
    ET.SubElement(inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")

    # 1) 각 링크 정의
    for lid in link_ids:
        link = ET.SubElement(robot, "link", name=f"link_{lid}")

        mesh = closed_parts[lid]

        # 질량/관성 (COM 기준)
        mass, com_world, I_com = _compute_mass_inertia_at_com(mesh, density=density)

        # link 프레임 원점(=부모와 연결되는 조인트 origin) in world
        link_o_world = link_frame_origin.get(lid, np.zeros(3))

        # visual/collision의 origin (mesh 좌표가 world 기준임을 가정)
        # -> 링크 프레임(=joint origin)로의 상대 이동만 적용 (회전은 0 가정)
        v_origin = com_world * 0.0  # 시각화는 메쉬 자체를 link frame에 배치하려면:
        # visual/collision은 보통 메쉬 원점(여기서는 world)에서 link frame으로 이동시켜야 하므로:
        vis_xyz = _to_str_xyz(np.asarray([0,0,0], float) - link_o_world)
        col_xyz = vis_xyz

        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=vis_xyz, rpy="0 0 0")
        geom_v = ET.SubElement(visual, "geometry")
        ET.SubElement(geom_v, "mesh", filename=os.path.relpath(mesh_paths[lid], out_dir))

        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=col_xyz, rpy="0 0 0")
        geom_c = ET.SubElement(collision, "geometry")
        ET.SubElement(geom_c, "mesh", filename=os.path.relpath(mesh_paths[lid], out_dir))

        # inertial: COM에 배치 (inertial.origin 은 link frame 기준 좌표)
        inertial = ET.SubElement(link, "inertial")
        com_in_link = com_world - link_o_world
        ET.SubElement(inertial, "origin", xyz=_to_str_xyz(com_in_link), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
        I = _urdf_inertia_dict(I_com)
        ET.SubElement(inertial, "inertia",
                      ixx=f"{I['ixx']:.9g}", ixy=f"{I['ixy']:.9g}", ixz=f"{I['ixz']:.9g}",
                      iyy=f"{I['iyy']:.9g}", iyz=f"{I['iyz']:.9g}", izz=f"{I['izz']:.9g}")

    # 2) 조인트 정의
    # joint origin 은 parent link frame 기준 좌표여야 함.
    # 각 링크 frame origin은 world에서 joint origin으로 정의했으므로:
    # joint origin (parent frame) = (joint_world_origin - parent_link_world_origin)
    for child, j in joint_by_child.items():
        parent = int(j["parent"])
        jtype = _mapping_get(j, "type")
        urdf_type = "revolute" if jtype.lower().startswith("rev") else "prismatic"

        parent_name = "base" if parent == 0 else f"link_{parent}"
        child_name = f"link_{child}"

        # 좌표
        joint_world_o = np.asarray(j["axis"]["origin"], dtype=float)
        parent_world_o = link_frame_origin.get(parent, np.zeros(3))
        joint_in_parent = joint_world_o - parent_world_o

        # 축 (parent frame 기준; 회전 0 가정 → world와 동일)
        axis = _norm(j["axis"]["n"])

        j_el = ET.SubElement(robot, "joint", name=f"joint_{parent}_{child}", type=urdf_type)
        ET.SubElement(j_el, "parent", link=parent_name)
        ET.SubElement(j_el, "child", link=child_name)
        ET.SubElement(j_el, "origin", xyz=_to_str_xyz(joint_in_parent), rpy="0 0 0")
        ET.SubElement(j_el, "axis", xyz=_to_str_xyz(axis))
        ET.SubElement(j_el, "limit", lower="-1.57", upper="1.57", effort="10.0", velocity="1.0")

        # 필요 시 limit 추가 (예: 프리즈마틱/리볼트) → 사용자 파라미터로 확장 가능
        # ET.SubElement(j_el, "limit", lower="-3.14", upper="3.14", effort="10", velocity="1.0")

    # 3) base와 직접 연결되지 않은 루트가 있을 경우, base에 고정 조인트로 연결(옵션)
    # (여기서는 joints가 제공하는 트리만 사용)

    # 저장
    tree = ET.ElementTree(robot)
    urdf_path = os.path.join(out_dir, f"{urdf_name}.urdf")
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path



def stage_articulate(cfgs, ms: pymeshlab.MeshSet):
    # 1) 원본 메쉬
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)


    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    links = states.segment.links
    joints = states.segment.joints

    # 2) 너의 links 구조(현재는 list[list[int]] 형태였죠)
    #    links = [[] for _ in range(categories_num+1)]
    #    for index, cat in enumerate(vert_label): links[cat].append(index)

    # 3) watertight 파트 메쉬 생성
    closed_parts = build_closed_meshes_from_links(
        verts, faces, links,
        pitch=None,  # 자동 해상도 (bbox 기반)
        pitch_rel=0.01,  # 해상도 상대값 (더 작게하면 디테일↑, 폴리곤↑)
        smooth_iters=10,  # 필요시 3~5 정도로 살짝 스무딩
        extrapoints_map=getattr(states.segment, 'extrapoints', None)
    )

    for lid, m in closed_parts.items():
        m.export(f"link_{lid}.ply")

    urdf_path = generate_urdf(
        closed_parts=closed_parts,
        joints=joints,  # parent, child, axis:{origin,n}, type in {"Revolute","Linear"}
        out_dir="out_urdf",
        urdf_name="my_robot",
        density=1000.0,  # 재질에 맞게 조정
        mesh_fmt="stl",
    )
    print("URDF:", urdf_path)

    import vedo
    actors = [vedo.Mesh([m.vertices, m.faces]).c('lightblue').alpha(0.7) for m in closed_parts.values()]
    vedo.show(actors, "Closed parts", axes=1)

    # closed_parts: { link_id(int) : trimesh.Trimesh (watertight) }
    # 저장 예:
    # for lid, m in closed_parts.items():
    #     m.export(f"link_{lid}_closed.ply")