import os
import numpy as np
import pymeshlab
import trimesh
from dataclasses import dataclass
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
from json_handler import JsonHandler

# -------- helpers: 좌표/노멀 추출, 레이 필터, 카테고리 할당 --------

def _get_positions(rec):
    names = rec.dtype.names or ()
    if all(k in names for k in ('x','y','z')):
        return np.stack([rec['x'], rec['y'], rec['z']], axis=1).astype(np.float64)
    cand = [
        ('pos_x','pos_y','pos_z'),
        ('px','py','pz'),
        ('position_0','position_1','position_2'),
        ('position0','position1','position2'),
        ('mean_0','mean_1','mean_2'),
        ('mu_0','mu_1','mu_2'),
    ]
    for a,b,c in cand:
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
        R[...,0,0] = 1 - 2*(y*y + z*z); R[...,0,1] = 2*(x*y - z*w); R[...,0,2] = 2*(x*z + y*w)
        R[...,1,0] = 2*(x*y + z*w);     R[...,1,1] = 1 - 2*(x*x + z*z); R[...,1,2] = 2*(y*z - x*w)
        R[...,2,0] = 2*(x*z - y*w);     R[...,2,1] = 2*(y*z + x*w);     R[...,2,2] = 1 - 2*(x*x + y*y)
        return R
    if all(f'rot_{i}' in names for i in range(4)):
        q = np.stack([rec['rot_0'], rec['rot_1'], rec['rot_2'], rec['rot_3']], axis=1).astype(np.float64)
        z = np.array([0.0,0.0,1.0]); R = _quat_to_R(q)
        n = (R @ z).astype(np.float64); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        return n
    if all(k in names for k in ('qw','qx','qy','qz')):
        q = np.stack([rec['qw'], rec['qx'], rec['qy'], rec['qz']], axis=1).astype(np.float64)
        z = np.array([0.0,0.0,1.0]); R = _quat_to_R(q)
        n = (R @ z).astype(np.float64); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        return n
    return None  # 노멀 불가 시 정렬/레이 필터 비활성

def _ray_filter(mesh, points, normals, max_len, eps):
    if normals is None:
        return np.ones(len(points), dtype=bool)
    rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    keep = np.zeros(len(points), dtype=bool)
    batch = 4096
    for s in range(0, len(points), batch):
        e = min(len(points), s + batch)
        o = points[s:e]; n = normals[s:e]
        for sign in (1.0, -1.0):
            dirs = n * sign
            locs, idx_ray, _ = rmi.intersects_location(o, dirs, multiple_hits=False)
            d = np.full(e - s, np.inf)
            if len(idx_ray):
                d[idx_ray] = np.linalg.norm(locs - o[idx_ray], axis=1)
            keep[s:e] |= (d > eps) & (d < max_len)
    return keep

def _assign_categories(mesh, vert_labels, gauss_xyz, gauss_n, cfgs):
    V  = np.asarray(mesh.vertices, dtype=np.float64)
    VN = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tree = cKDTree(V)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    sigma = max(1e-9, cfgs.sigma_rel * diag)
    k = max(1, cfgs.k_neighbors)
    out = np.zeros(len(gauss_xyz), dtype=np.int32)
    for i, p in enumerate(gauss_xyz):
        dists, idxs = tree.query(p, k=min(k, len(V)))
        if np.isscalar(dists): dists = np.array([dists]); idxs = np.array([idxs], dtype=int)
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

# -------- scoring/filter cfg --------

@dataclass
class BindCfgs:
    k_neighbors: int = 8
    sigma_rel: float = 0.02
    align_power: float = 1.0
    w_dist: float = 1.0
    w_align: float = 1.0
    ray_max_rel: float = 2.0
    ray_eps: float = 1e-6
    drop_if_ray_miss: bool = True
    verbose: bool = True

# -------- main: PlyData 방식 적용(모든 속성 보존 저장) --------

def stage_bind(cfgs, ms: pymeshlab.MeshSet):
    os.makedirs(cfgs.gaussian_out_dir, exist_ok=True)

    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    mesh = trimesh.load(states.refine.dirs.mesh)
    vert_labels = states.segment.vert_label

    # (핵심) Gaussian은 PlyData로 읽고 vertex 구조체를 그대로 유지
    g_ply = PlyData.read(states.transform.dirs.gaussian)
    g_rec = g_ply['vertex'].data

    # 좌표/노멀 안전 추출
    gauss_xyz = _get_positions(g_rec)
    gauss_n   = _ensure_normals_any(g_rec)

    cfg = BindCfgs()

    # 레이 기반 필터(노멀이 있을 경우에만)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    ray_max = cfg.ray_max_rel * diag
    keep = _ray_filter(mesh, gauss_xyz, gauss_n, ray_max, cfg.ray_eps) if cfg.drop_if_ray_miss else np.ones(len(gauss_xyz), bool)

    # 카테고리 할당
    assigned = _assign_categories(mesh, vert_labels, gauss_xyz, gauss_n, cfg)

    # (핵심) 저장: 원본 vertex의 모든 필드/순서/dtype 그대로, 마스크만 적용하여 깔끔 저장
    cats = np.unique(assigned[keep])
    base_name = os.path.splitext(os.path.basename(states.transform.dirs.gaussian))[0]
    for c in cats:
        idxs = np.nonzero(keep & (assigned == c))[0]
        if idxs.size == 0:
            continue
        sub = g_rec[idxs]  # 원본 구조체 배열 슬라이스(모든 property 그대로)
        new_vertex = PlyElement.describe(sub, 'vertex')
        new_ply = PlyData([new_vertex], text=False)  # 예제와 동일한 “깔끔한” 저장 스타일
        out_path = os.path.join(cfgs.gaussian_out_dir, f"{base_name}_{int(c)}.ply")
        new_ply.write(out_path)
        if cfg.verbose:
            print(f"Saved {len(sub)} points -> {out_path}")