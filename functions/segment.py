import os
from functools import partial
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pymeshlab
import trimesh
import vedo

from json_handler import JsonHandler
from functions.estimate import learn_separator_main
from functions.graphs import cluster_reciprocal_loop_pairs, classify_vertices


# ================================
# Geometry helpers (planes, projection, polygon tests)
# ================================
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v * 0.0 if n < 1e-12 else v / n


def _orthonormal_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _normalize(n)
    axes = np.eye(3)
    k = int(np.argmin(np.abs(axes @ n)))
    u = _normalize(np.cross(n, axes[k]))
    v = _normalize(np.cross(n, u))
    return u, v, n


def _pca_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    X = pts - c
    C = X.T @ X
    _, V = np.linalg.eigh(C)
    n = _normalize(V[:, 0])
    return c, n


def _points_in_polygon_2d(poly_xy: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    """Ray casting in 2D."""
    x = grid_xy[:, 0]; y = grid_xy[:, 1]
    xp = poly_xy[:, 0]; yp = poly_xy[:, 1]
    n = len(poly_xy)
    inside = np.zeros(len(grid_xy), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = xp[i], yp[i]
        xj, yj = xp[j], yp[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def _nearest_segment_projection(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Project 2D points P onto segment A→B; return (proj, t, d2)."""
    AB = B - A
    denom = (AB @ AB) + 1e-12
    t = ((P - A) @ AB) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = A + t[:, None] * AB
    d2 = np.sum((P - proj) ** 2, axis=1)
    return proj, t, d2


def _boundary_distance_and_height(grid_xy: np.ndarray, poly_uv: np.ndarray, h_loop: np.ndarray):
    """Distance to loop and linearly interpolated boundary height at closest point."""
    if poly_uv.shape[0] >= 2 and not np.allclose(poly_uv[0], poly_uv[-1]):
        poly_uv = np.vstack([poly_uv, poly_uv[0]])
        h_loop = np.concatenate([h_loop, h_loop[0:1]])
    M = grid_xy.shape[0]
    dist2 = np.full(M, np.inf, dtype=float)
    hb = np.zeros(M, dtype=float)
    for i in range(len(poly_uv) - 1):
        A = poly_uv[i]
        B = poly_uv[i + 1]
        _, t, d2 = _nearest_segment_projection(grid_xy, A, B)
        upd = d2 < dist2
        if np.any(upd):
            dist2[upd] = d2[upd]
            hb[upd] = (1.0 - t[upd]) * h_loop[i] + t[upd] * h_loop[i + 1]
    return np.sqrt(dist2), hb


# ================================
# Loop sampling & triangulation
# ================================
def _densify_loop_uv(poly_uv: np.ndarray, step: float) -> np.ndarray:
    """Insert points along polygon edges roughly every `step` (closed polyline returned)."""
    poly_uv = np.asarray(poly_uv, dtype=float)
    if poly_uv.shape[0] >= 2 and not np.allclose(poly_uv[0], poly_uv[-1]):
        poly_uv = np.vstack([poly_uv, poly_uv[0]])
    pts = [poly_uv[0]]
    for i in range(len(poly_uv) - 1):
        a = poly_uv[i]; b = poly_uv[i + 1]
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L <= 1e-12:
            continue
        nadd = max(0, int(np.ceil(L / max(step, 1e-12))) - 1)
        for t in range(1, nadd + 1):
            pts.append(a + (t / (nadd + 1)) * seg)
        pts.append(b)
    return np.asarray(pts, dtype=float)


def _triangulate_points_2d(points_uv: np.ndarray):
    """Delaunay via matplotlib.tri. Returns (triangulation_object, used_flag)."""
    try:
        import matplotlib.tri as mtri
    except Exception:
        return None, False
    pts = np.asarray(points_uv, dtype=float)
    if pts.shape[0] < 3:
        return None, False
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
    if tri.triangles.size == 0:
        return None, False
    return tri, True


def sample_dense_points_on_loop_delaunay(loop_xyz: np.ndarray,
                                         *,
                                         step: float | None = None,
                                         step_rel: float = 0.01,
                                         sigma_scale: float = 2.0,
                                         k_min: int = 4,
                                         max_passes: int = 5) -> np.ndarray:
    """
    (디버그/분석용) 내부 고밀도 점을 만들고 경계에서부터 가우시안 평균 전파로 높이 복원.
    주 메쉬 생성은 triangulate_patch_on_loop 사용.
    """
    loop_xyz = np.asarray(loop_xyz, dtype=float)
    if loop_xyz.shape[0] < 3:
        return np.zeros((0, 3), dtype=float)

    c, n = _pca_plane(loop_xyz)
    u, v, n = _orthonormal_basis_from_normal(n)

    rel = loop_xyz - c
    poly_uv = np.column_stack([rel @ u, rel @ v])

    if not np.allclose(poly_uv[0], poly_uv[-1]):
        poly_closed = np.vstack([poly_uv, poly_uv[0]])
    else:
        poly_closed = poly_uv

    bb_min = poly_uv.min(axis=0); bb_max = poly_uv.max(axis=0)
    diag = float(np.linalg.norm(bb_max - bb_min))
    if diag <= 0:
        return np.zeros((0, 3), dtype=float)
    if step is None:
        step = max(diag * step_rel, 1e-6)

    nx = max(1, int(np.ceil((bb_max[0] - bb_min[0]) / step))) + 1
    ny = max(1, int(np.ceil((bb_max[1] - bb_min[1]) / step))) + 1
    gx = np.linspace(bb_min[0], bb_max[0], nx)
    gy = np.linspace(bb_min[1], bb_max[1], ny)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([GX.ravel(), GY.ravel()])

    inside_flat = _points_in_polygon_2d(poly_closed[:-1] if np.allclose(poly_closed[0], poly_closed[-1]) else poly_closed, grid_xy)
    if not np.any(inside_flat):
        return np.zeros((0, 3), dtype=float)
    inside_idx = np.nonzero(inside_flat)[0]
    pts_inside = grid_xy[inside_idx]

    dense_boundary = _densify_loop_uv(poly_uv, step)
    h_loop = (loop_xyz - c) @ n
    dist_all, hb_all = _boundary_distance_and_height(pts_inside, poly_uv, h_loop)

    pts_uv = np.vstack([pts_inside, dense_boundary])
    M_in = pts_inside.shape[0]
    M_bd = dense_boundary.shape[0]
    dist = np.concatenate([dist_all, np.zeros(M_bd, dtype=float)])
    _, hb_dense = _boundary_distance_and_height(dense_boundary, poly_uv, h_loop)
    hb = np.concatenate([hb_all, hb_dense])

    tri_obj, tri_used = _triangulate_points_2d(pts_uv)
    neighbors = None
    if tri_used:
        # neighbor graph from triangulation
        tris = np.asarray(tri_obj.triangles, dtype=int)
        neighbors = [set() for _ in range(pts_uv.shape[0])]
        for t in tris:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            neighbors[a].update((b, c))
            neighbors[b].update((a, c))
            neighbors[c].update((a, b))
        neighbors = [np.asarray(sorted(list(s)), dtype=int) for s in neighbors]

    H = np.full(pts_uv.shape[0], np.nan, dtype=float)
    known = np.zeros_like(H, dtype=bool)
    bd_adj = (np.arange(pts_uv.shape[0]) >= M_in) | ((dist <= (step * 0.75)) & (np.arange(pts_uv.shape[0]) < M_in))
    H[bd_adj] = hb[bd_adj]; known[bd_adj] = True

    order = np.argsort(dist[:M_in])
    sigma = max(step * sigma_scale, 1e-9)
    radius = 3.0 * sigma

    def _update_from_neighbors(idx):
        if neighbors is not None:
            cand = neighbors[idx]
        else:
            dif = pts_uv - pts_uv[idx]
            d = np.sqrt(np.sum(dif * dif, axis=1))
            cand = np.nonzero((d > 0) & (d <= radius))[0]
        if cand.size == 0:
            return False
        kk = cand[known[cand]]
        if kk.size < max(1, k_min):
            return False
        d2 = np.sum((pts_uv[kk] - pts_uv[idx]) ** 2, axis=1)
        w = np.exp(-d2 / (2.0 * sigma * sigma))
        ws = float(np.sum(w))
        if ws <= 1e-12:
            return False
        H[idx] = float(np.sum(w * H[kk]) / ws)
        known[idx] = True
        return True

    for idx in order:
        if not known[idx]:
            _update_from_neighbors(idx)

    passes = 0
    while (not np.all(known[:M_in])) and passes < max_passes:
        passes += 1
        for idx in np.nonzero(~known[:M_in])[0]:
            _update_from_neighbors(idx)

    pend = np.nonzero(~known[:M_in])[0]
    if pend.size:
        H[pend] = hb[:M_in][pend]; known[pend] = True

    X = pts_uv[:, 0]; Y = pts_uv[:, 1]; Z = H
    P3 = c + np.outer(X, u) + np.outer(Y, v) + np.outer(Z, n)
    return P3


# ================================
# Harmonic height field (smooth patch fill)
# ================================
def _harmonic_heights_on_triangulation(pts_uv: np.ndarray,
                                       tri_obj,
                                       boundary_mask: np.ndarray,
                                       h_boundary: np.ndarray,
                                       *,
                                       prefer_scipy: bool = True,
                                       max_iters: int = 1000,
                                       tol: float = 1e-6,
                                       value_weighted: bool = False,
                                       alpha: float = 0.2,
                                       weight_clip: float = 10.0,
                                       warmup: int = 5) -> np.ndarray:
    """
    Solve Laplacian(H)=0 on interior with Dirichlet boundary.
    - value_weighted=True: Jacobi에서 값(높이)이 큰 이웃일수록 가중치↑
      w_j = 1 + alpha * clip(|H_j|, 0, weight_clip)
      (비선형이므로 SciPy(CG) 경로는 사용하지 않음)
    """
    import numpy as _np
    N = pts_uv.shape[0]
    tris = _np.asarray(tri_obj.triangles, dtype=int)

    # 이웃 리스트 구성 (uniform 그래프)
    neighbors = [set() for _ in range(N)]
    for t in tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    deg = _np.array([len(s) for s in neighbors], dtype=float)

    # 초기값: 경계 고정
    H = _np.zeros(N, dtype=float)
    H[boundary_mask] = h_boundary[boundary_mask]
    interior = ~boundary_mask
    if not interior.any():
        return H

    # 값 가중 켜면 SciPy 경로는 비활성화(비선형)
    if value_weighted:
        prefer_scipy = False

    used_scipy = False
    if prefer_scipy:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            rows, cols, data = [], [], []
            b = _np.zeros(N, dtype=float)
            for i in range(N):
                if boundary_mask[i]:
                    rows.append(i); cols.append(i); data.append(1.0)
                    b[i] = H[i]
                else:
                    rows.append(i); cols.append(i); data.append(1.0)
                    if deg[i] > 0:
                        w = 1.0 / deg[i]
                        for j in neighbors[i]:
                            rows.append(i); cols.append(int(j)); data.append(-w)
            A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
            H = spla.cg(A, b, x0=H, tol=tol, maxiter=max_iters)[0]
            used_scipy = True
        except Exception:
            used_scipy = False

    if not used_scipy:
        # Jacobi 반복 (경계는 고정)
        H_new = H.copy()
        for it in range(max_iters):
            max_delta = 0.0
            for i in _np.nonzero(interior)[0]:
                if deg[i] == 0:
                    continue
                s = 0.0
                wsum = 0.0
                for j in neighbors[i]:
                    Hj = H[int(j)]
                    if value_weighted and it >= warmup:
                        w = 1.0 + alpha * min(abs(Hj), weight_clip)
                    else:
                        w = 1.0
                    s += w * Hj
                    wsum += w
                if wsum <= 1e-12:
                    v = H[i]
                else:
                    v = s / wsum
                d = abs(v - H[i])
                if d > max_delta:
                    max_delta = d
                H_new[i] = v
            H, H_new = H_new, H
            if max_delta < tol:
                break
    return H

def triangulate_patch_on_loop(loop_xyz: np.ndarray,
                              *,
                              step_rel: float = 0.01,
                              sigma_scale: float = 2.0,
                              k_min: int = 6):
    """
    Triangulated patch (faces 포함) 생성:
      1) PCA 평면으로 사영 → 내부 고밀도 샘플 + 경계 densify
      2) Delaunay triangulation (가능 시)
      3) 경계 높이 고정(Dirichlet), 내부는 Harmonic 보간(∇²h=0)
      4) 3D 복원 및 폴리곤 내부 삼각형만 유지
    반환: (P3:(N,3), F:(M,3))
    """
    loop_xyz = np.asarray(loop_xyz, dtype=float)
    if loop_xyz.shape[0] < 3:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    c, n = _pca_plane(loop_xyz)
    u, v, n = _orthonormal_basis_from_normal(n)

    rel = loop_xyz - c
    poly_uv = np.column_stack([rel @ u, rel @ v])

    if not np.allclose(poly_uv[0], poly_uv[-1]):
        poly_closed = np.vstack([poly_uv, poly_uv[0]])
    else:
        poly_closed = poly_uv

    bb_min = poly_uv.min(axis=0); bb_max = poly_uv.max(axis=0)
    diag = float(np.linalg.norm(bb_max - bb_min))
    if diag <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    step = max(diag * step_rel, 1e-6)

    nx = max(1, int(np.ceil((bb_max[0] - bb_min[0]) / step))) + 1
    ny = max(1, int(np.ceil((bb_max[1] - bb_min[1]) / step))) + 1
    gx = np.linspace(bb_min[0], bb_max[0], nx)
    gy = np.linspace(bb_min[1], bb_max[1], ny)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([GX.ravel(), GY.ravel()])
    inside_flat = _points_in_polygon_2d(poly_closed[:-1] if np.allclose(poly_closed[0], poly_closed[-1]) else poly_closed, grid_xy)
    if not np.any(inside_flat):
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    inside_idx = np.nonzero(inside_flat)[0]
    pts_inside = grid_xy[inside_idx]

    dense_boundary = _densify_loop_uv(poly_uv, step)
    h_loop = (loop_xyz - c) @ n
    dist_all, hb_all = _boundary_distance_and_height(pts_inside, poly_uv, h_loop)

    pts_uv = np.vstack([pts_inside, dense_boundary])
    M_in = pts_inside.shape[0]
    M_bd = dense_boundary.shape[0]
    _, hb_dense = _boundary_distance_and_height(dense_boundary, poly_uv, h_loop)

    tri_obj, tri_used = _triangulate_points_2d(pts_uv)
    if not tri_used:
        # triangulation 실패 → faces 없음. (디버그용 포인트만)
        # radius 기반 전파로 높이 대충 채워서 3D 포인트만 반환
        dist = np.concatenate([dist_all, np.zeros(M_bd, dtype=float)])
        hb = np.concatenate([hb_all, hb_dense])
        H = np.full(pts_uv.shape[0], np.nan, dtype=float)
        known = np.zeros_like(H, dtype=bool)
        bd_adj = (np.arange(pts_uv.shape[0]) >= M_in) | (
            (dist <= (step * 0.75)) & (np.arange(pts_uv.shape[0]) < M_in)
        )
        H[bd_adj] = hb[bd_adj]; known[bd_adj] = True
        sigma = max(step * sigma_scale, 1e-9)
        radius = 3.0 * sigma
        order = np.argsort(dist[:M_in])

        def _upd(idx):
            d = np.sqrt(np.sum((pts_uv - pts_uv[idx])**2, axis=1))
            cand = np.nonzero((d > 0) & (d <= radius))[0]
            kk = cand[known[cand]]
            if kk.size < k_min:
                return False
            w = np.exp(-(d[kk]**2) / (2.0 * sigma * sigma))
            ws = float(np.sum(w))
            if ws <= 1e-12: return False
            H[idx] = float(np.sum(w * H[kk]) / ws); return True

        for idx in order:
            if not known[idx]:
                known[idx] = _upd(idx)
        pend = np.nonzero(~known[:M_in])[0]
        if pend.size:
            H[pend] = hb[:M_in][pend]
        X = pts_uv[:, 0]; Y = pts_uv[:, 1]; Z = H
        P3 = c + np.outer(X, u) + np.outer(Y, v) + np.outer(Z, n)
        return P3, np.zeros((0, 3), dtype=int)

    # Harmonic solve on triangulation (경계: 마지막 M_bd 포인트)
    N = pts_uv.shape[0]
    boundary_mask = np.zeros(N, dtype=bool)
    boundary_mask[M_in:M_in + M_bd] = True
    hB = np.zeros(N, dtype=float)
    hB[M_in:M_in + M_bd] = hb_dense
    H = _harmonic_heights_on_triangulation(
        pts_uv, tri_obj, boundary_mask, hB,
        prefer_scipy=True,  # value_weighted=True이면 함수 내부에서 False로 전환
        max_iters=1000, tol=1e-6,
        value_weighted=True,  # ★ 값 기반 가중 Jacobi 활성화
        alpha=0.5,  # ★ 가중 강도 (0.1~0.3 권장)
        weight_clip=10.0,  # ★ 안정화용 상한
        warmup=5  # ★ 초기엔 uniform 평균으로 예열
    )

    # 3D embedding
    X = pts_uv[:, 0]; Y = pts_uv[:, 1]; Z = H
    P3 = c + np.outer(X, u) + np.outer(Y, v) + np.outer(Z, n)

    # keep only triangles whose centroid lies inside polygon
    tris = np.asarray(tri_obj.triangles, dtype=int)
    cent = (pts_uv[tris][:, 0, :] + pts_uv[tris][:, 1, :] + pts_uv[tris][:, 2, :]) / 3.0
    inside_tri = _points_in_polygon_2d(poly_closed[:-1] if np.allclose(poly_closed[0], poly_closed[-1]) else poly_closed, cent)
    tris = tris[inside_tri]
    return P3, tris.astype(int)


# ================================
# Mesh extraction / I/O helpers
# ================================
def _extract_submesh(verts_np: np.ndarray, faces_np: np.ndarray, labels_np: np.ndarray, k: int):
    """Faces whose 3 vertices all have label k → submesh (verts reindexed)."""
    faces_k_mask = np.all(labels_np[faces_np] == k, axis=1)
    faces_k = faces_np[faces_k_mask]
    if faces_k.size == 0:
        return None, None
    vidx = np.unique(faces_k.ravel())
    vmap = -np.ones(labels_np.shape[0], dtype=int)
    vmap[vidx] = np.arange(vidx.size, dtype=int)
    sub_faces = vmap[faces_k]
    sub_verts = verts_np[vidx]
    return sub_verts, sub_faces


def save_link_meshes_from_labels(verts: np.ndarray,
                                 faces: np.ndarray,
                                 labels: np.ndarray,
                                 categories_num: int,
                                 out_dir: str) -> Dict[str, str]:
    """라벨별 서브메쉬 저장. 반환: {str(k): path}."""
    os.makedirs(out_dir, exist_ok=True)
    mesh_paths: Dict[str, str] = {}
    for k in range(1, categories_num + 1):
        sv, sf = _extract_submesh(verts, faces, labels, k)
        if sv is None or sf is None:
            continue
        tri_k = trimesh.Trimesh(vertices=sv, faces=sf, process=False)
        path = os.path.join(out_dir, f"link_{k}.ply")
        try:
            tri_k.export(path)
        except Exception:
            path = os.path.join(out_dir, f"link_{k}.obj")
            tri_k.export(path)
        mesh_paths[str(k)] = path
    return mesh_paths


def append_patches_to_mesh_files(mesh_paths: Dict[str, str],
                                 patch_vertices: Dict[int, List[np.ndarray]],
                                 patch_faces: Dict[int, List[np.ndarray]]) -> Dict[str, str]:
    """각 링크 메시에 패치 (P,F) append 후 덮어쓰기."""
    for k_str, path in list(mesh_paths.items()):
        try:
            k = int(k_str)
        except Exception:
            continue
        if (k not in patch_vertices) or (len(patch_vertices[k]) == 0):
            continue
        tri_k = trimesh.load(path, process=False)
        V = tri_k.vertices
        F = tri_k.faces
        for P3, Ft in zip(patch_vertices[k], patch_faces[k]):
            if P3.size == 0 or Ft.size == 0:
                continue
            offset = V.shape[0]
            V = np.vstack([V, P3])
            F = np.vstack([F, Ft + offset])
        tri_out = trimesh.Trimesh(vertices=V, faces=F, process=False)
        try:
            tri_out.remove_duplicate_vertices()
            tri_out.remove_degenerate_faces()
            tri_out.remove_unreferenced_vertices()
        except Exception:
            pass
        try:
            tri_out.export(path)
        except Exception:
            alt = os.path.splitext(path)[0] + ".obj"
            tri_out.export(alt)
            mesh_paths[k_str] = alt
    return mesh_paths


# ================================
# Joints
# ================================
def build_joints_from_rlps(rlps: List[dict]) -> List[dict]:
    """
    rlp['vectors'] 중 사용자가 선택(Linear/Revolute)한 축들을 joint로 변환.
    여러 축이 선택되면 모두 기록.
    """
    joints = []
    for rlp in rlps:
        p = int(rlp["a"]["parent"]); c = int(rlp["a"]["child"])
        vecs = rlp.get("vectors", []) or []
        chosen = [v for v in vecs if isinstance(v, dict) and v.get("state") in ("Linear", "Revolute")]

        def _ord_key(v):
            o = v.get("order", None)
            return (1, 1e18) if o is None else (0, int(o))
        chosen.sort(key=_ord_key)

        for v in chosen:
            ori = np.asarray(v.get("center", rlp.get("center", [0, 0, 0])), dtype=float).ravel()
            nvec = v.get("n") if v.get("n") is not None else v.get("n_poly")
            nvec = _normalize(np.asarray(nvec, dtype=float).ravel())
            if not np.isfinite(nvec).all() or np.linalg.norm(nvec) < 1e-12:
                continue
            joints.append({
                "parent": p, "child": c,
                "axis": {"origin": ori.astype(float).tolist(), "n": nvec.astype(float).tolist()},
                "type": v.get("state")
            })
    return joints


# ================================
# Pipeline helpers (keeps stage_segment minimal)
# ================================
def compute_separation_results(ms: pymeshlab.MeshSet,
                               categories: List[List[trimesh.Trimesh]],
                               categories_num: int,
                               method: str = "all") -> List[dict]:
    """Run learn_separator_main per category and flatten results."""
    results: Dict[int, dict | List[dict]] = {}
    for idx_part in range(1, categories_num + 1):
        out = learn_separator_main(
            ms, categories, idx_part,
            max_hops=10, method=method,
            C=1.0, gamma='scale',
            use_signed_dist=True, sdf_thresh=0.0,
            balance='None'
        )
        results[idx_part] = out
    results_list = [item for v in results.values() for item in (v if isinstance(v, list) else [v])]
    return results_list


def prepare_vectors_for_rlps(rlps: List[dict]) -> None:
    """평면/법선 정보를 바탕으로 rlp에 벡터 후보들을 채워 넣기."""
    for rlp in rlps:
        center = rlp['center']
        A = rlp['a']['linear']; B = rlp['b']['linear']
        if A.get('plane') and len(A['plane']) > 0 and B.get('plane') and len(B['plane']) > 0:
            nA = np.asarray(A['plane'][0][0], dtype=float)
            nB = np.asarray(B['plane'][0][0], dtype=float)
            dA = float(A['plane'][0][1]); dB = float(B['plane'][0][1])
        else:
            nA = np.asarray(A.get('pca_normal', [0, 0, 1]), dtype=float)
            nB = np.asarray(B.get('pca_normal', [0, 0, 1]), dtype=float)
            dA = dB = 0.0
        if float(np.dot(nA, nB)) < 0.0:
            nB = -nB
        n = _normalize(np.mean(np.vstack([nA, nB]), axis=0))
        d = 0.5 * (dA + dB)
        rlp['vectors'] = [{'center': center, 'n': n, 'd': d}]

        A = rlp['a']['polyhedral']
        if A.get('plane') and len(A['plane']) > 0:
            planes = A['plane']; m = len(planes)
            for u in range(m):
                for v in range(u + 1, m):
                    n = np.cross(planes[u][0], planes[v][0])
                    rlp['vectors'].append({'center': center, 'n': n, 'd': np.zeros(3)})


def build_extrapoints_and_patches(results_list: List[dict],
                                  verts: np.ndarray):
    """
    각 loop에 대해:
      - (디버그) 고밀도 pts3d
      - 실제 메움용 triangulated patch (P3, F)
    """
    extrapoints: Dict[int, List[List[float]]] = defaultdict(list)
    patch_vertices: Dict[int, List[np.ndarray]] = defaultdict(list)
    patch_faces: Dict[int, List[np.ndarray]] = defaultdict(list)
    for res in results_list:
        pid = int(res.get('parent', 0))
        loop_idx = res.get('loop', None)
        if pid <= 0 or loop_idx is None or len(loop_idx) < 3:
            continue
        loop_xyz = verts[np.asarray(loop_idx, dtype=int)]
        pts3d = sample_dense_points_on_loop_delaunay(loop_xyz, step_rel=0.03, sigma_scale=2.0, k_min=6)
        if pts3d.size:
            extrapoints[pid].extend(pts3d.astype(float).tolist())
        P3, F = triangulate_patch_on_loop(loop_xyz, step_rel=0.03, sigma_scale=2.0, k_min=6)
        if P3.size and F.size:
            patch_vertices[pid].append(P3.astype(float))
            patch_faces[pid].append(F.astype(np.int64))
    return extrapoints, patch_vertices, patch_faces


# ================================
# Visualization callbacks (unchanged API)
# ================================
def recolor_parts(vmeshes, belongings, mode, rand_colors):
    if mode == 0:
        for i, vmesh in enumerate(vmeshes):
            vmesh.c(rand_colors[i]).alpha(0.5)
    else:
        for i, vmesh in enumerate(vmeshes):
            if belongings[i] == mode:
                vmesh.c("red").alpha(0.8)
            else:
                vmesh.c("lightblue").alpha(0.4)


def on_left_click(event, vmeshes, belongings, prevs, shared, rand_colors, plt):
    vmesh = event.actor
    mode = shared['mode']
    if vmesh is None or mode == 0:
        return
    if vmesh in vmeshes[1:]:
        idx = vmesh.idx_part
        prevs[idx] = belongings[idx]
        belongings[idx] = mode
        recolor_parts(vmeshes, belongings, mode, rand_colors)
        plt.render()


def on_right_click(event, vmeshes, belongings, prevs, shared, rand_colors, plt):
    vmesh = event.actor
    mode = shared['mode']
    if vmesh is None or mode == 0:
        return
    if vmesh in vmeshes[1:]:
        idx = vmesh.idx_part
        belongings[idx] = prevs[idx]
        recolor_parts(vmeshes, belongings, mode, rand_colors)
        plt.render()


def on_tab(event, vmeshes, belongings, prevs, shared, categories_num, rand_colors, plt):
    if event.keypress in ('\t', 'Tab'):
        shared['mode'] = (shared['mode'] + 1) % (categories_num + 1)
        mode = shared['mode']
        recolor_parts(vmeshes, belongings, mode, rand_colors)
        plt.render()
        if mode == 0:
            print(f"[info]: defualt mode")
        else:
            print(f"[info]: selection mode #{mode}")


def on_key(event, vmeshes, belongings, prevs, shared, rand_colors, categories_num, plt):
    """ESC/q 종료, Tab 선택 모드 순환."""
    k = getattr(event, "keypress", None)
    if k in ("q", "Q", "Esc", "Escape", "\x1b"):
        try: plt.close()
        except Exception: pass
        try: vedo.close()
        except Exception: pass
        return
    if k in ('\t', 'Tab'):
        on_tab(event, vmeshes, belongings, prevs, shared, categories_num, rand_colors, plt)


def on_vector_click(event, *, vec_actors, rlp, click_counter):
    """'None'→'Linear'→'Revolute'→'None' 순환. 최초 선택 order 기록."""
    act = getattr(event, "actor", None)
    if act is None or act not in vec_actors:
        return
    st = getattr(act, "vec_state", "None")
    idx = getattr(act, "vec_idx", None)
    if idx is None:
        return

    if st == "None":
        try: act.c("yellow").alpha(1.0)
        except Exception: pass
        setattr(act, "vec_state", "Linear")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            click_counter['count'] = int(click_counter.get('count', 0)) + 1
            rlp['vectors'][idx]['state'] = "Linear"
            rlp['vectors'][idx]['order'] = int(click_counter['count'])
    elif st == "Linear":
        try: act.c("blue").alpha(1.0)
        except Exception: pass
        setattr(act, "vec_state", "Revolute")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "Revolute"
    else:
        try: act.c("gray").alpha(0.5)
        except Exception: pass
        setattr(act, "vec_state", "None")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "None"
            rlp['vectors'][idx]['order'] = None
    try: event.plotter.render()
    except Exception: pass

def visualize_and_select_vectors_for_rlps(rlps: List[dict],
                                          parts: List[trimesh.Trimesh],
                                          categories: List[List[trimesh.Trimesh]],
                                          verts: np.ndarray,
                                          faces: np.ndarray) -> None:
    """
    각 RLP에 대해:
      - 베이스 메쉬(흰색) + 두 파트(색상) 렌더
      - rlp['vectors'] 후보 축을 화살표로 표시
      - 클릭으로 'None'→'Linear'→'Revolute' 순환 (on_vector_click 콜백 재사용)
    rlp['vectors'][k]['state'], 'order'가 사용자 선택에 따라 갱신됨.
    """
    for i, rlp in enumerate(rlps):
        # base mesh
        mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.35)
        try:
            mesh_actor.pickable(False)
        except Exception:
            pass

        # two parts (parent/child) in distinct colors
        p_idx = int(rlp["a"]["parent"])
        q_idx = int(rlp["a"]["child"])
        part_ids = [pid for pid in (p_idx, q_idx) if pid != 0 and 1 <= pid <= len(parts)]
        part_ids = list(dict.fromkeys(part_ids))

        part_colors = [(0.85, 0.2, 0.2), (0.2, 0.4, 0.85)]
        part_actors = []
        for j, pid in enumerate(part_ids):
            for tri in categories[pid]:
                actor = vedo.Mesh([tri.vertices, tri.faces]).c(part_colors[min(j, 1)]).alpha(0.25)
                try:
                    actor.pickable(False)
                except Exception:
                    pass
                part_actors.append(actor)

        # vector arrows
        seg_len = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) * 0.2
        vec_actors = []
        for k, v in enumerate(rlp.get('vectors', []) or []):
            c = np.asarray(v.get("center", rlp.get("center", [0, 0, 0])), dtype=float)
            n = _normalize(np.asarray(v.get("n"), dtype=float))
            if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                continue
            p0 = c
            p1 = c + seg_len * n
            try:
                a = vedo.Arrow(p0, p1, shaftRadius=seg_len * 0.02).c("gray").alpha(1.0)
            except TypeError:
                try:
                    a = vedo.Tube([p0, p1], r=seg_len * 0.02, cap=True).c("gray").alpha(1.0)
                except Exception:
                    a = vedo.Line(p0, p1).c("gray").alpha(1.0).lw(8)
            setattr(a, "vec_state", "None")
            setattr(a, "vec_idx", k)
            vec_actors.append(a)

        # click order counter per RLP
        click_counter = {'count': 0}

        plt_rlp = vedo.Plotter(title=f"RLP #{i}: parts {p_idx} ↔ {q_idx}", axes=1)
        plt_rlp.add_callback(
            "LeftButtonPress",
            partial(on_vector_click, vec_actors=vec_actors, rlp=rlp, click_counter=click_counter)
        )
        plt_rlp.show([mesh_actor, *part_actors, *vec_actors], interactive=True).close()



# ================================
# Stage function (kept minimal)
# ================================
def stage_segment(cfgs, ms: pymeshlab.MeshSet):
    """
    1) 파트 선택(UI) → 카테고리 구성
    2) 분리 평면 추정 → RLP 클러스터링 → 축 벡터 후보 생성(+선택 UI)
    3) 라벨 기반 링크별 서브메쉬 저장
    4) 루프 패치(조화 보간) 메움 + 엑스트라 포인트 저장
    5) 상태 저장(states.segment.*)
    """
    categories_num = 3
    categories = [[] for _ in range(categories_num + 1)]

    # Load states & base mesh
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    parts = [trimesh.load(states.decompose.dirs.mesh[i]) for i in range(states.decompose.length)]

    ms.set_current_mesh(0)
    current_mesh = ms.current_mesh()
    verts = current_mesh.vertex_matrix()
    faces = current_mesh.face_matrix().astype(np.int64)

    # ---- Selection UI ----
    vmeshes, rand_colors = [], []
    base_actor = vedo.Mesh([verts, faces])
    vmeshes.append(base_actor); rand_colors.append((255, 255, 255))
    for i, part in enumerate(parts, start=1):
        actor = vedo.Mesh([part.vertices, part.faces])
        color = np.random.rand(3); actor.c(color).alpha(0.5)
        actor.idx_part = i
        vmeshes.append(actor); rand_colors.append(color)

    belongings = [1 for _ in range(len(vmeshes))]; belongings[0] = 0
    prevs = [1 for _ in range(len(vmeshes))]; prevs[0] = 0
    shared = {"mode": 0}

    plt = vedo.Plotter(title="Part selection")
    # Disable default vtk keybindings
    try:
        iren = plt.interactor
        for _ev in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
            iren.RemoveObservers(_ev)
    except Exception:
        pass

    plt.add_callback("LeftButtonPress", partial(on_left_click,  vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, plt=plt))
    plt.add_callback("RightButtonPress", partial(on_right_click, vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, plt=plt))
    plt.add_callback("KeyPress",        partial(on_key,        vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, categories_num=categories_num, plt=plt))
    recolor_parts(vmeshes, belongings, 0, rand_colors)
    plt.show(vmeshes, interactive=True)

    # Build categories from selection
    for i in range(1, len(belongings)):
        categories[belongings[i]].append(parts[i - 1])

    # ---- Separation & RLPs ----
    results_list = compute_separation_results(ms, categories, categories_num, method="all")
    rlps = cluster_reciprocal_loop_pairs(results_list, w_pos=1.0, w_ang=0.5, cost_max=None)
    prepare_vectors_for_rlps(rlps)

    # ---- RLP별 벡터 시각화/선택 ----
    visualize_and_select_vectors_for_rlps(rlps, parts, categories, verts, faces)

    # ---- Save state & per-link meshes ----
    vert_label = classify_vertices(verts, categories, use_signed_dist=True).astype(int)
    seg_dir = os.path.join(os.path.dirname(cfgs.json_states_dir), "segment_mesh")
    segment_mesh_paths = save_link_meshes_from_labels(verts, faces, vert_label, categories_num, seg_dir)
    joints = build_joints_from_rlps(rlps)

    states.segment = {}
    states.segment.dirs = {}
    states.segment.dirs.mesh = segment_mesh_paths
    states.segment.joints = joints