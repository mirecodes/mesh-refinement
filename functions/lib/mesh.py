import os
import numpy as np
import trimesh
from typing import Dict, List, Tuple

from functions.lib.geometry import (
    pca_plane, orthonormal_basis_from_normal, densify_loop_uv, 
    boundary_distance_and_height, triangulate_points_2d, points_in_polygon_2d,
    upd_radius
)

# =============================================================================
# Dense sampling (debug) & harmonic-field patching
# =============================================================================
def harmonic_heights_on_triangulation(pts_uv: np.ndarray,
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
    """Solve Laplacian(H)=0 with Dirichlet boundary on given triangulation."""
    import numpy as _np
    N = pts_uv.shape[0]
    tris = _np.asarray(tri_obj.triangles, dtype=int)

    # neighbor graph
    neighbors = [set() for _ in range(N)]
    for t in tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    deg = _np.array([len(s) for s in neighbors], dtype=float)

    # init
    H = _np.zeros(N, dtype=float)
    H[boundary_mask] = h_boundary[boundary_mask]
    interior = ~boundary_mask
    if not interior.any():
        return H

    if value_weighted:
        prefer_scipy = False

    used_scipy = False
    if prefer_scipy:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
        except Exception as e:
            raise ImportError("scipy is required for sparse solve") from e
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

    if not used_scipy:
        # Jacobi (fixed boundary)
        H_new = H.copy()
        for _ in range(max_iters):
            max_delta = 0.0
            for i in _np.nonzero(interior)[0]:
                if deg[i] == 0:
                    continue
                s = 0.0
                wsum = 0.0
                for j in neighbors[i]:
                    Hj = H[int(j)]
                    w = 1.0 + alpha * min(abs(Hj), weight_clip) if value_weighted else 1.0
                    s += w * Hj
                    wsum += w
                v = H[i] if wsum <= 1e-12 else (s / wsum)
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
    """Delaunay + harmonic height interpolation patch."""
    loop_xyz = np.asarray(loop_xyz, dtype=float)
    if loop_xyz.shape[0] < 3:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)

    c, n = pca_plane(loop_xyz)
    u, v, n = orthonormal_basis_from_normal(n)

    rel = loop_xyz - c
    poly_uv = np.column_stack([rel @ u, rel @ v])

    poly_closed = np.vstack([poly_uv, poly_uv[0]]) if not np.allclose(poly_uv[0], poly_uv[-1]) else poly_uv

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

    poly_for_test = poly_closed[:-1] if np.allclose(poly_closed[0], poly_closed[-1]) else poly_closed
    inside_flat = points_in_polygon_2d(poly_for_test, grid_xy)
    if not np.any(inside_flat):
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    inside_idx = np.nonzero(inside_flat)[0]
    pts_inside = grid_xy[inside_idx]

    dense_boundary = densify_loop_uv(poly_uv, step)
    h_loop = (loop_xyz - c) @ n
    _, hb_dense = boundary_distance_and_height(dense_boundary, poly_uv, h_loop)

    pts_uv = np.vstack([pts_inside, dense_boundary])
    M_in = pts_inside.shape[0]
    M_bd = dense_boundary.shape[0]

    tri_obj, tri_used = triangulate_points_2d(pts_uv)
    if not tri_used:
        # No triangulation: radial propagation fallback
        dist_all, hb_all = boundary_distance_and_height(pts_inside, poly_uv, h_loop)
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
        for idx in order:
            if not known[idx]:
                known[idx] = upd_radius(idx, pts_uv, H, known, sigma, k_min, radius)
        pend = np.nonzero(~known[:M_in])[0]
        if pend.size:
            H[pend] = hb[:M_in][pend]
        X = pts_uv[:, 0]; Y = pts_uv[:, 1]; Z = H
        P3 = c + np.outer(X, u) + np.outer(Y, v) + np.outer(Z, n)
        return P3, np.zeros((0, 3), dtype=int)

    # Harmonic interpolation
    N = pts_uv.shape[0]
    boundary_mask = np.zeros(N, dtype=bool)
    boundary_mask[M_in:M_in + M_bd] = True
    hB = np.zeros(N, dtype=float)
    hB[M_in:M_in + M_bd] = hb_dense
    H = harmonic_heights_on_triangulation(
        pts_uv, tri_obj, boundary_mask, hB,
        prefer_scipy=True, max_iters=1000, tol=1e-6,
        value_weighted=True, alpha=0.5, weight_clip=10.0, warmup=5
    )

    # Embed to 3D and keep only interior triangles
    X = pts_uv[:, 0]; Y = pts_uv[:, 1]; Z = H
    P3 = c + np.outer(X, u) + np.outer(Y, v) + np.outer(Z, n)
    tris = np.asarray(tri_obj.triangles, dtype=int)
    cent = (pts_uv[tris][:, 0, :] + pts_uv[tris][:, 1, :] + pts_uv[tris][:, 2, :]) / 3.0
    inside_tri = points_in_polygon_2d(poly_for_test, cent)
    tris = tris[inside_tri]
    return P3, tris.astype(int)


# =============================================================================
# Mesh extraction & I/O
# =============================================================================
def extract_submesh(verts_np: np.ndarray, faces_np: np.ndarray, labels_np: np.ndarray, k: int):
    """Extract submesh whose all vertices share label k."""
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
    """Save per-label submeshes. Returns {label_str: path}."""
    os.makedirs(out_dir, exist_ok=True)
    mesh_paths: Dict[str, str] = {}
    for k in range(1, categories_num + 1):
        sv, sf = extract_submesh(verts, faces, labels, k)
        if sv is None or sf is None:
            continue
        tri_k = trimesh.Trimesh(vertices=sv, faces=sf, process=False)
        ply_path = os.path.join(out_dir, f"link_{k}.ply")
        try:
            tri_k.export(ply_path)
            path = ply_path
        except Exception:
            obj_path = os.path.join(out_dir, f"link_{k}.obj")
            try:
                tri_k.export(obj_path)
                path = obj_path
            except Exception as e:
                raise IOError(f"Failed to export mesh for label {k} to PLY and OBJ") from e
        mesh_paths[str(k)] = path
    return mesh_paths


def append_patches_to_mesh_files(mesh_paths: Dict[str, str],
                                 patch_vertices: Dict[int, List[np.ndarray]],
                                 patch_faces: Dict[int, List[np.ndarray]]) -> Dict[str, str]:
    """Append patches (P,F) to existing link meshes and overwrite on disk."""
    for k_str, path in list(mesh_paths.items()):
        k = int(k_str)
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
        # cleanup (these methods exist in trimesh)
        tri_out.remove_duplicate_faces()
        tri_out.remove_degenerate_faces()
        tri_out.remove_unreferenced_vertices()
        # export back to original path (fallback to OBJ if fails)
        try:
            tri_out.export(path)
        except Exception:
            alt = os.path.splitext(path)[0] + ".obj"
            try:
                tri_out.export(alt)
            except Exception as e:
                raise IOError(f"Failed to export patched mesh to {path} and {alt}") from e
            mesh_paths[k_str] = alt
    return mesh_paths
