import os
from dataclasses import dataclass

import numpy as np
import pymeshlab
import trimesh
from json_handler import JsonHandler
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

# --- minimal helpers (kept) ---------------------------------------------------

from scipy.spatial import cKDTree

def _assign_categories_proximity(mesh, vert_labels, gauss_xyz):
    """
    노말/가우시안 스코어링 없이, 가장 가까운 mesh vertex의 label로 소속을 결정.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    tree = cKDTree(V)
    # 각 가우시안에 대해 최근접 정점 하나(k=1)
    dists, idxs = tree.query(gauss_xyz, k=1)
    # vert_labels[idxs] 그대로 반환
    return np.asarray(vert_labels, dtype=np.int32)[idxs]


def _ray_filter(mesh, points, normals, max_len, eps):
    if normals is None:
        return np.ones(len(points), dtype=bool)
    rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    keep = np.zeros(len(points), dtype=bool)
    batch = 4096
    for s in range(0, len(points), batch):
        e = min(len(points), s + batch)
        o = points[s:e]
        n = normals[s:e]
        for sign in (1.0, -1.0):
            dirs = n * sign
            locs, idx_ray, _ = rmi.intersects_location(o, dirs, multiple_hits=False)
            d = np.full(e - s, np.inf)
            if len(idx_ray):
                d[idx_ray] = np.linalg.norm(locs - o[idx_ray], axis=1)
            keep[s:e] |= (d > eps) & (d < max_len)
    return keep

def _assign_categories(mesh, vert_labels, gauss_xyz, gauss_n, cfgs):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    VN = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tree = cKDTree(V)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    sigma = max(1e-9, cfgs.sigma_rel * diag)
    k = max(1, cfgs.k_neighbors)
    out = np.zeros(len(gauss_xyz), dtype=np.int32)

    for i, p in enumerate(gauss_xyz):
        dists, idxs = tree.query(p, k=min(k, len(V)))
        if np.isscalar(dists):
            dists = np.array([dists]); idxs = np.array([idxs], dtype=int)
        w_dist = np.exp(-(dists**2) / (2.0 * sigma * sigma)) * cfgs.w_dist
        if gauss_n is not None:
            aligns = np.maximum(0.0, VN[idxs] @ gauss_n[i])
            w_align = (aligns ** cfgs.align_power) * cfgs.w_align
        else:
            w_align = np.ones_like(w_dist)
        score = w_dist * w_align
        j = int(idxs[np.argmax(score)]) if np.any(score > 0) else int(idxs[0])
        out[i] = int(vert_labels[j])
    return out

# --- robust field extraction / preservation -----------------------------------

def _get_positions(rec):
    names = rec.dtype.names or ()
    if all(k in names for k in ('x','y','z')):
        return np.stack([rec['x'], rec['y'], rec['z']], axis=1).astype(np.float64)
    cand_triplets = [
        ('pos_x','pos_y','pos_z'),
        ('px','py','pz'),
        ('position_0','position_1','position_2'),
        ('position0','position1','position2'),
        ('mean_0','mean_1','mean_2'),
        ('mu_0','mu_1','mu_2'),
    ]
    for a,b,c in cand_triplets:
        if a in names and b in names and c in names:
            return np.stack([rec[a], rec[b], rec[c]], axis=1).astype(np.float64)
    for base in ('position','pos','mean','mu'):
        if f'{base}[0]' in names and f'{base}[1]' in names and f'{base}[2]' in names:
            return np.stack([rec[f'{base}[0]'], rec[f'{base}[1]'], rec[f'{base}[2]']], axis=1).astype(np.float64)
    raise KeyError("Cannot find position fields (x,y,z or known aliases).")

def _ensure_normals_any(rec):
    names = rec.dtype.names or ()
    if all(k in names for k in ('nx','ny','nz')):
        n = np.stack([rec['nx'], rec['ny'], rec['nz']], axis=1).astype(np.float64)
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        return n
    def _quat_to_R(q):
        w,x,y,z = q[...,0], q[...,1], q[...,2], q[...,3]
        s = np.sqrt(w*w+x*x+y*y+z*z) + 1e-12
        w,x,y,z = w/s, x/s, y/s, z/s
        R = np.empty(q.shape[:-1] + (3,3), dtype=np.float64)
        R[...,0,0] = 1 - 2*(y*y + z*z)
        R[...,0,1] = 2*(x*y - z*w)
        R[...,0,2] = 2*(x*z + y*w)
        R[...,1,0] = 2*(x*y + z*w)
        R[...,1,1] = 1 - 2*(x*x + z*z)
        R[...,1,2] = 2*(y*z - x*w)
        R[...,2,0] = 2*(x*z - y*w)
        R[...,2,1] = 2*(y*z + x*w)
        R[...,2,2] = 1 - 2*(x*x + y*y)
        return R
    if all(f'rot_{i}' in names for i in range(4)):
        q = np.stack([rec['rot_0'], rec['rot_1'], rec['rot_2'], rec['rot_3']], axis=1).astype(np.float64)
        Rm = _quat_to_R(q); z = np.array([0.0,0.0,1.0])
        n = (Rm @ z).astype(np.float64); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        return n
    if all(k in names for k in ('qw','qx','qy','qz')):
        q = np.stack([rec['qw'], rec['qx'], rec['qy'], rec['qz']], axis=1).astype(np.float64)
        Rm = _quat_to_R(q); z = np.array([0.0,0.0,1.0])
        n = (Rm @ z).astype(np.float64); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        return n
    return None

def _slice_vertex_in_ply(ply: PlyData, idxs):
    new_elements = []
    for el in ply.elements:
        if el.name == 'vertex':
            names = el.data.dtype.names
            sliced = np.empty(len(idxs), dtype=el.data.dtype)  # dtype 그대로 유지 → 모든 필드/타입 보존
            for n in names:
                sliced[n] = el.data[n][idxs]
            new_el = PlyElement.describe(sliced, 'vertex')
            new_elements.append(new_el)
        else:
            # vertex 외 element(예: face, custom element)는 그대로 유지
            new_elements.append(el)

    # 🔧 변경점: 원본의 text/binary 모드와 byte_order를 그대로 사용
    new_ply = PlyData(new_elements, text=ply.text)
    new_ply.byte_order = ply.byte_order
    new_ply.comments = list(ply.comments)
    new_ply.obj_info = list(ply.obj_info)
    return new_ply

# --- config for scoring/filtering ---------------------------------------------

@dataclass
class BindCfgs:
    label_mode: str = "auto"
    k_neighbors: int = 8
    sigma_rel: float = 0.02
    align_power: float = 1.0
    w_dist: float = 1.0
    w_align: float = 1.0
    ray_max_rel: float = 2.0
    ray_eps: float = 1e-6
    drop_if_ray_miss: bool = True
    verbose: bool = True

# --- main ---------------------------------------------------------------------

def stage_bind(cfgs, ms: pymeshlab.MeshSet):
    os.makedirs(cfgs.gaussian_out_dir, exist_ok=True)

    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    mesh = trimesh.load(states.refine.dirs.mesh)
    vert_labels = states.segment.vert_label

    # Read original Gaussian PLY (preserve all fields/elements)
    g_ply = PlyData.read(states.transform.dirs.gaussian)
    g_rec = g_ply['vertex'].data

    # Robust position/normal extraction (any field names / quaternion)
    gauss_xyz = _get_positions(g_rec)
    gauss_n   = _ensure_normals_any(g_rec)

    bind_cfgs = BindCfgs()

    # Filter by normal-ray exit (if normals available)
    # diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    # ray_max = bind_cfgs.ray_max_rel * diag
    # keep = _ray_filter(mesh, gauss_xyz, gauss_n, ray_max, bind_cfgs.ray_eps) if bind_cfgs.drop_if_ray_miss else np.ones(len(gauss_xyz), bool)
    #
    # # Assign categories (distance + normal alignment)
    # assigned = _assign_categories(mesh, vert_labels, gauss_xyz, gauss_n, bind_cfgs)

    # ==== (변경) ray 필터 완전 비활성화 ====
    keep = np.ones(len(gauss_xyz), dtype=bool)
    if bind_cfgs.verbose:
        print(f"[ray] disabled — keeping all {len(gauss_xyz)} points")

    # ==== (변경) proximity 기반 카테고리 배정 ====
    assigned = _assign_categories_proximity(mesh, vert_labels, gauss_xyz)
    if bind_cfgs.verbose:
        uniq, cnts = np.unique(assigned, return_counts=True)
        print("[assign-proximity] category counts:", dict(zip(uniq.tolist(), cnts.tolist())))

    # Save per-category with ALL original fields/elements preserved
    cats = np.unique(assigned[keep])
    print(cats)
    for c in cats:
        idxs = np.nonzero(keep & (assigned == c))[0]
        if idxs.size == 0:
            continue
        out_path = os.path.join(cfgs.gaussian_out_dir, f"gaussian_{int(c)}.ply")
        print(f"Writing {out_path}")
        _slice_vertex_in_ply(g_ply, idxs).write(out_path)