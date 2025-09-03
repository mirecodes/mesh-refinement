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
                               smooth_iters: int = 0) -> trimesh.Trimesh:
    """
    open mesh라도 voxelization → marching_cubes로 닫힌 watertight mesh를 생성.
    - pitch가 None이면 bbox 대각선 * pitch_rel 로 자동 설정.
    - smooth_iters > 0 이면 약간의 스무딩 수행(선택).
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
    # marching_cubes on the filled volume → closed surface
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


# ---------- 3) links를 받아 각 링크별 watertight mesh 생성 ----------
def build_closed_meshes_from_links(verts: np.ndarray,
                                   faces: np.ndarray,
                                   links,
                                   *,
                                   pitch: float | None = None,
                                   pitch_rel: float = 0.01,
                                   smooth_iters: int = 0):
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
        closed = remesh_watertight_trimesh(sub_v, sub_f, pitch=pitch, pitch_rel=pitch_rel, smooth_iters=smooth_iters)
        out[int(link_id)] = closed
    return out

def stage_articulate(cfgs, ms: pymeshlab.MeshSet):
    # 1) 원본 메쉬
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)


    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    links = states.segment.links

    # 2) 너의 links 구조(현재는 list[list[int]] 형태였죠)
    #    links = [[] for _ in range(categories_num+1)]
    #    for index, cat in enumerate(vert_label): links[cat].append(index)

    # 3) watertight 파트 메쉬 생성
    closed_parts = build_closed_meshes_from_links(
        verts, faces, links,
        pitch=None,  # 자동 해상도 (bbox 기반)
        pitch_rel=0.01,  # 해상도 상대값 (더 작게하면 디테일↑, 폴리곤↑)
        smooth_iters=0  # 필요시 3~5 정도로 살짝 스무딩
    )

    for lid, m in closed_parts.items():
        m.export(f"link_{lid}.ply")

    import vedo
    actors = [vedo.Mesh([m.vertices, m.faces]).c('lightblue').alpha(0.7) for m in closed_parts.values()]
    vedo.show(actors, "Closed parts", axes=1)

    # closed_parts: { link_id(int) : trimesh.Trimesh (watertight) }
    # 저장 예:
    # for lid, m in closed_parts.items():
    #     m.export(f"link_{lid}_closed.ply")