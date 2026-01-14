import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
from functions.lib.geometry import normalize


# ================================
# Small math / format helpers
# ================================
def to_str_xyz(v):
    v = np.asarray(v, dtype=float).ravel()
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"


def mapping_get(mapping, key):
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
# Joint building
# =============================================================================
def build_joints_from_rlps(rlps: List[dict]) -> List[dict]:
    """
    RLP(Radial Line Pair) 정보로부터 조인트 후보군을 추출합니다.
    여기서는 아직 트리 구조를 강제하지 않고, 기하학적으로 발견된 모든 연결을 반환합니다.
    """
    picked = []
    for rlp in rlps:
        p = int(rlp.get("a", {}).get("parent", rlp.get("parent", -1)))
        c = int(rlp.get("a", {}).get("child", rlp.get("child", -1)))

        # 유효성 검사
        if p < 0 or c < 0 or p == c:
            continue

        # 일단 입력된 순서대로 저장 (나중에 트리 구성 단계에서 재정렬됨)
        # 단, 내부적으로는 i < j 순서로 pair를 key로 쓰기 위해 정렬
        i, j = (p, c) if p < c else (c, p)

        for v in (rlp.get("vectors") or []):
            st = v.get("state")
            if st not in ("Revolute", "Prismatic"):
                continue
            o = np.asarray(v.get("center", rlp.get("center", [0, 0, 0])), dtype=float).ravel()
            n = np.asarray(v.get("n") if v.get("n") is not None else v.get("n_poly"), dtype=float).ravel()

            if o.size != 3 or n.size != 3: continue
            n = normalize(n)
            if not np.isfinite(o).all() or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                continue

            picked.append({"pair": (i, j), "origin": o, "n": n, "type": st})

    if not picked:
        return []

    # 클러스터링 로직 (같은 위치의 여러 벡터 병합)
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
        return groups, np.vstack(centers) if centers else np.zeros((0, 3))

    by_pair: Dict[Tuple[int, int], List[int]] = defaultdict(list)
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

            # 타입이 섞이면 에러 처리 (여기서는 첫 번째 타입 채택)
            jtype = next(iter(types))
            dirs = np.vstack([it["n"] for it in items])

            # parent, child는 일단 pair의 순서(작은값, 큰값)대로 저장하지만
            # 나중에 build_valid_tree_joints에서 방향이 결정됨
            joint_def = {
                "parent": pair[0], "child": pair[1],
                "axis": {"origin": origin.astype(float).tolist()},
                "type": jtype
            }

            if jtype == "Revolute" and len(items) >= 2:
                # Spherical 등 복합 조인트 처리
                joint_def["type"] = "Spherical"
                joint_def["axis"]["n"] = dirs[0].astype(float).tolist()  # 대표축
            elif jtype == "Prismatic" and len(items) >= 2:
                # Planar 처리
                _, _, Vt = np.linalg.svd(dirs, full_matrices=False)
                n_plane = normalize(Vt[-1])
                joint_def["type"] = "Planar"
                joint_def["axis"]["n"] = n_plane.astype(float).tolist()
            else:
                # 일반 조인트
                n = dirs[0]
                joint_def["axis"]["n"] = n.astype(float).tolist()

            joints.append(joint_def)

    return joints


# ================================
# Tree Logic (New Algorithm)
# ================================
def build_valid_tree_joints(raw_joints: List[dict], all_link_ids: List[int]) -> Tuple[List[dict], Set[int]]:
    """
    모든 조인트 정보를 그래프로 구성한 뒤, BFS를 통해 순환 없는 트리 구조(Spanning Tree)를 생성합니다.
    1. 작은 ID가 부모가 되도록 우선순위를 둡니다.
    2. 연결이 끊긴 노드를 찾아내거나, 루프를 끊습니다.
    """
    if not all_link_ids:
        return [], set()

    # 1. 인접 리스트 생성 (Undirected Graph)
    # adj[u] = [(v, joint_info), ...]
    adj = defaultdict(list)
    for j in raw_joints:
        p, c = int(j["parent"]), int(j["child"])
        # 원본 데이터와 함께 저장
        adj[p].append((c, j))
        adj[c].append((p, j))

    # 2. BFS 초기화
    # 루트 노드 선정 (가장 작은 ID를 루트로 가정, 보통 1번)
    root_id = min(all_link_ids)
    visited = {root_id}
    queue = deque([root_id])

    valid_joints = []

    # 3. BFS 탐색
    while queue:
        u = queue.popleft()

        # 인접 노드들을 '번호 순'으로 정렬하여 큐에 넣음
        # -> 이렇게 해야 "작은 숫자가 부모가 되는" 경향을 유지할 수 있음
        neighbors = sorted(adj[u], key=lambda x: x[0])

        for v, j_info in neighbors:
            if v in visited:
                continue

            # v를 처음 방문함 -> u가 부모, v가 자식으로 확정
            visited.add(v)
            queue.append(v)

            # 조인트 정보 복사 및 parent/child 재설정
            new_joint = j_info.copy()
            new_joint["parent"] = u
            new_joint["child"] = v

            # 만약 원본 정보와 parent/child가 뒤바꼈다면?
            # (기하학적 축 방향은 유지하되, 물리적 의미는 사용자가 해석해야 함.
            #  보통 회전축 선(Line) 자체는 변하지 않으므로 그대로 둡니다.)

            valid_joints.append(new_joint)

    # 4. 고립된 노드 확인 (Visited에 없는 노드들)
    unvisited = set(all_link_ids) - visited

    return valid_joints, unvisited


# ================================
# URDF Generation
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
    com = np.asarray(mesh.center_mass, dtype=float)
    try:
        vol = float(mesh.volume)
    except Exception:
        vol = 0.0
    mass = float(density) * vol

    # I at COM
    T = np.eye(4);
    T[:3, 3] = -com
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
        I_com = np.asarray(mp["inertia"], dtype=float) if isinstance(mp, dict) and "inertia" in mp else np.zeros((3, 3),
                                                                                                                 float)
    return mass, com, I_com


def urdf_inertia_dict(I):
    return {
        "ixx": float(I[0, 0]), "ixy": float(I[0, 1]), "ixz": float(I[0, 2]),
        "iyy": float(I[1, 1]), "iyz": float(I[1, 2]), "izz": float(I[2, 2]),
    }


def gather_link_frame_origins(joints, base_origin=np.zeros(3), transform_matrix=None):
    """
    트리 구조가 확정된 joints 리스트를 순회하며 각 링크의 World 원점을 계산
    """
    base_origin = np.asarray(base_origin, float)
    if transform_matrix is not None:
        p_h = np.append(base_origin, 1.0)
        p_trans = np.dot(transform_matrix, p_h)
        base_origin = p_trans[:3] / p_trans[3]

    link_o = {0: base_origin}  # 0 reserved for base

    # joints는 이미 BFS로 정렬되어 있으므로(부모가 먼저 방문됨), 순차적으로 계산 가능
    # 하지만 딕셔너리로 안전하게 처리

    # 1. 트리 구성을 위한 룩업
    children_of = defaultdict(list)
    joint_by_child = {}
    for j in joints:
        p, c = int(j["parent"]), int(j["child"])
        children_of[p].append(c)
        joint_by_child[c] = j

    # 2. 루트 찾기 (joints 리스트에 없는 최상위 부모들)
    # 보통 link_1 이지만, valid_tree_joints 로직에 따라 결정됨
    # 이 함수 외부에서 link_1을 base에 붙이므로, 여기선 재귀적으로 계산

    # 계산을 위해 모든 링크 ID 수집
    all_links = set()
    for j in joints:
        all_links.add(int(j["parent"]))
        all_links.add(int(j["child"]))

    # 재귀적으로 위치 계산
    def _calc_pos(lid, parent_pos):
        link_o[lid] = parent_pos  # 링크의 원점은 부모 조인트 위치가 아니라, 일단 부모와 상대적 위치...
        # 수정: URDF에서 Link의 Origin은 보통 Visual mesh의 원점과 일치시킴 (World 좌표)
        # Joint의 Origin이 Link의 pivot point가 됨.

        # 여기서는 "각 링크가 World에서 어디에 있는지" (Mesh 원점 기준)를 저장
        # 부모가 확정되면, 자식 Joint의 Origin을 알 수 있음.
        pass

    # 로직 수정: 단순히 Joint 정의에 있는 'Origin'은 World 좌표계(입력 데이터 기준) 값이므로
    # 부모/자식 관계와 상관없이 Joint Origin 좌표 자체는 불변함.
    # 따라서 그냥 Joint 정보를 순회하며 저장하면 됨.

    for j in joints:
        c = int(j["child"])
        o = np.asarray(j["axis"]["origin"], dtype=float)

        if transform_matrix is not None:
            p_h = np.append(o, 1.0)
            p_trans = np.dot(transform_matrix, p_h)
            o = p_trans[:3] / p_trans[3]

        link_o[c] = o  # 자식 링크의 피벗 포인트(Joint Origin) 저장

        # 부모가 아직 link_o에 없다면? (최상위 루트)
        # 보통 link_1 은 아래 로직에서 base_joint를 통해 0(base)에 연결되므로 초기값 사용
        p = int(j["parent"])
        if p not in link_o:
            link_o[p] = base_origin

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
        transform_matrix: np.ndarray = None,
):
    os.makedirs(out_dir, exist_ok=True)
    mesh_paths = export_meshes(closed_parts, os.path.join(out_dir, "meshes"), fmt=mesh_fmt)

    # ==========================================
    # 1. 트리 구조 재구성 (Tree Logic Applied Here)
    # ==========================================
    link_ids = sorted(closed_parts.keys())

    # BFS를 이용해 순환을 제거하고 트리를 만듦. parent < child 규칙 준수 노력.
    tree_joints, unvisited_links = build_valid_tree_joints(joints, link_ids)

    # ==========================================
    # 2. 링크 원점 계산
    # ==========================================
    link_frame_origin = gather_link_frame_origins(tree_joints, base_origin, transform_matrix)

    # unvisited 링크들(고립된 링크)의 원점은 base_origin으로 가정
    for uid in unvisited_links:
        if uid not in link_frame_origin:
            # transform 적용
            bo = np.asarray(base_origin, float)
            if transform_matrix is not None:
                bo = (np.dot(transform_matrix, np.append(bo, 1.0)))[:3]
            link_frame_origin[uid] = bo

    # ==========================================
    # 3. XML 생성
    # ==========================================
    robot = ET.Element("robot", name=urdf_name)

    # Base Link
    base_link = ET.SubElement(robot, "link", name="base")
    ET.SubElement(base_link, "inertial").extend([
        ET.Element("origin", xyz="0 0 0", rpy="0 0 0"),
        ET.Element("mass", value="0.0"),
        ET.Element("inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0")
    ])

    # Regular Links
    for lid in link_ids:
        link = ET.SubElement(robot, "link", name=f"link_{lid}")
        mesh = closed_parts[lid]

        # Mass Calc (Apply transform temporarily)
        mesh_for_calc = mesh.copy()
        if transform_matrix is not None:
            mesh_for_calc.apply_transform(transform_matrix)
        mass, com_world, I_com = compute_mass_inertia_at_com(mesh_for_calc, density=density)

        link_o_world = link_frame_origin.get(lid, np.zeros(3))

        # Visual/Collision Origin = - (Link Frame Origin)
        # 메쉬는 파일 자체가 World 좌표계(혹은 Local 0,0,0)에 있으므로,
        # URDF상에서 Visual Origin을 역으로 이동시켜야 링크 프레임 원점에 메쉬가 맞음
        rel_xyz = to_str_xyz(np.asarray([0, 0, 0], float) - link_o_world)

        # Visual
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=rel_xyz, rpy="0 0 0")
        ET.SubElement(visual, "geometry").append(
            ET.Element("mesh", filename=os.path.relpath(mesh_paths[lid], out_dir))
        )

        # Collision
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=rel_xyz, rpy="0 0 0")
        ET.SubElement(collision, "geometry").append(
            ET.Element("mesh", filename=os.path.relpath(mesh_paths[lid], out_dir))
        )

        # Inertial (COM relative to Link Frame)
        com_in_link = com_world - link_o_world
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=to_str_xyz(com_in_link), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
        I = urdf_inertia_dict(I_com)
        ET.SubElement(inertial, "inertia", **{k: f"{v:.9g}" for k, v in I.items()})

    # Joints (from Tree)
    for j in tree_joints:
        parent, child = int(j["parent"]), int(j["child"])
        parent_name = f"link_{parent}"  # 트리 내부 연결은 모두 link_X 간 연결
        child_name = f"link_{child}"

        # Origin Calc
        joint_world_o = np.asarray(j["axis"]["origin"], dtype=float)
        if transform_matrix is not None:
            p_h = np.append(joint_world_o, 1.0)
            p_trans = np.dot(transform_matrix, p_h)
            joint_world_o = p_trans[:3] / p_trans[3]

        parent_world_o = link_frame_origin.get(parent, np.zeros(3))
        joint_in_parent = joint_world_o - parent_world_o

        # Type Parsing
        jtype_raw = str(mapping_get(j, "type") or "revolute").strip().lower()
        if jtype_raw.startswith("rev"):
            urdf_type = "revolute"
        elif jtype_raw.startswith("pri"):
            urdf_type = "prismatic"
        elif jtype_raw.startswith("pla"):
            urdf_type = "planar"
        elif jtype_raw.startswith("sph"):
            urdf_type = "floating"
        else:
            urdf_type = "revolute"

        j_el = ET.SubElement(robot, "joint", name=f"joint_{parent}_{child}", type=urdf_type)
        ET.SubElement(j_el, "parent", link=parent_name)
        ET.SubElement(j_el, "child", link=child_name)
        ET.SubElement(j_el, "origin", xyz=to_str_xyz(joint_in_parent), rpy="0 0 0")

        if urdf_type in ("revolute", "prismatic", "planar"):
            n_raw = mapping_get(j, "axis")
            n_vec = mapping_get(n_raw, "n") if isinstance(n_raw, dict) else None
            axis = normalize(n_vec if n_vec is not None else [0, 0, 1])
            if transform_matrix is not None:
                axis = normalize(np.dot(transform_matrix[:3, :3], axis))
            ET.SubElement(j_el, "axis", xyz=to_str_xyz(axis))

        if urdf_type in ("revolute", "prismatic"):
            ET.SubElement(j_el, "limit", lower="-1.57", upper="1.57", effort="10.0", velocity="1.0")

    # ==========================================
    # 4. Base & Disconnected Link Handling
    # ==========================================
    # 4-1. Main Root Connection (Base -> Link 1 or min_id)
    # 트리의 루트(BFS 시작점)를 Base에 고정
    if link_ids:
        root_id = min(link_ids)  # BFS에서 시작점으로 썼던 ID
        base_joint = ET.SubElement(robot, "joint", name=f"base_to_{root_id}", type="fixed")
        ET.SubElement(base_joint, "parent", link="base")
        ET.SubElement(base_joint, "child", link=f"link_{root_id}")

        root_origin = link_frame_origin.get(root_id, np.zeros(3))
        ET.SubElement(base_joint, "origin", xyz=to_str_xyz(root_origin), rpy="0 0 0")

    # 4-2. Floating Nodes Connection (Base -> Orphan)
    # 조인트 정보가 없어서 트리에서 누락된 링크들은 Base에 고정시켜 에러 방지
    for uid in unvisited_links:
        if uid == min(link_ids): continue  # 이미 위에서 연결됨

        fix_joint = ET.SubElement(robot, "joint", name=f"fixed_base_{uid}", type="fixed")
        ET.SubElement(fix_joint, "parent", link="base")
        ET.SubElement(fix_joint, "child", link=f"link_{uid}")

        # 고립된 링크의 원점은 Base 원점과 같다고 가정 (혹은 0,0,0)
        u_origin = link_frame_origin.get(uid, np.zeros(3))
        ET.SubElement(fix_joint, "origin", xyz=to_str_xyz(u_origin), rpy="0 0 0")

    # Save
    tree = ET.ElementTree(robot)
    urdf_path = os.path.join(out_dir, f"{urdf_name}.urdf")
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return urdf_path