
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


@dataclass
class SystemConfig:
    # Paths
    root_dir: str = "../../../../Downloads"
    mesh_path: str = "meshes/full_mesh.ply"           # Triangular mesh with vertices & (ideally) vertex normals
    gaussian_in_path: str = "gaussians/input.ply"     # PLY with x,y,z and ideally nx,ny,nz for each Gaussian
    gaussian_out_dir: str = "gaussians/out"           # Output directory for gaussian_{category}.ply
    decomposed_mesh_dir: Optional[str] = None         # Optional dir of per-part meshes to infer labels if needed

    # Label loading preference:
    #  - "auto": try mesh-embedded labels/property, fallback to .labels.npy, fallback to nearest-part labeling
    #  - "mesh_prop": require per-vertex int property 'label' or 'vert_label' in the mesh PLY
    #  - "npy": require <mesh_path>.labels.npy file
    #  - "nearest_parts": compute per-vertex labels by nearest decomposed part meshes in decomposed_mesh_dir
    label_mode: str = "auto"

    # Scoring / filtering
    k_neighbors: int = 8              # neighbor vertices for likelihood
    sigma_rel: float = 0.02           # sigma as fraction of mesh bbox diagonal for distance term
    align_power: float = 1.0          # exponent on normal alignment term (>=1)
    w_dist: float = 1.0               # weight for distance (Gaussian kernel)
    w_align: float = 1.0              # weight for alignment (cosine)
    ray_max_rel: float = 2.0          # ray length as multiple of bbox diagonal
    ray_eps: float = 1e-6             # small epsilon to ignore self-intersection at origin
    drop_if_ray_miss: bool = True     # drop Gaussians whose normal-ray exits the mesh

    # Misc
    verbose: bool = True


# ----------------------------- Utilities -----------------------------

def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangular mesh at {mesh_path}, got {type(mesh)}")
    if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.vertex_normals = mesh.vertex_normals  # triggers computation
    return mesh

def _try_load_vertex_labels_from_mesh_ply(mesh_path: str) -> Optional[np.ndarray]:
    try:
        ply = PlyData.read(mesh_path)
    except Exception:
        return None
    if 'vertex' not in ply:
        return None
    v = ply['vertex']
    for key in ('label', 'vert_label', 'category', 'part_id'):
        if key in v.data.dtype.names:
            arr = np.asarray(v.data[key], dtype=np.int32)
            return arr
    return None

def _try_load_vertex_labels_npy(mesh_path: str) -> Optional[np.ndarray]:
    base = os.path.splitext(mesh_path)[0]
    npy_path = base + ".labels.npy"
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        return arr.astype(np.int32)
    return None

def _compute_vertex_labels_by_nearest_parts(mesh: trimesh.Trimesh, parts_dir: str) -> np.ndarray:
    # Load each part mesh; assign each vertex to nearest part by distance
    part_files = []
    for fn in os.listdir(parts_dir):
        if fn.lower().endswith(('.ply', '.obj', '.stl', '.glb', '.off')):
            part_files.append(os.path.join(parts_dir, fn))
    if not part_files:
        raise FileNotFoundError(f"No part meshes found in {parts_dir}")

    part_meshes = [trimesh.load(pf, force='mesh') for pf in sorted(part_files)]
    part_kdtrees = []
    for pm in part_meshes:
        if isinstance(pm, trimesh.Scene):
            pm = pm.dump().sum()
        part_kdtrees.append(cKDTree(np.asarray(pm.vertices)))

    V = np.asarray(mesh.vertices)
    labels = np.zeros(len(V), dtype=np.int32)
    for i, v in enumerate(V):
        dmins = [kdtree.query(v, k=1)[0] for kdtree in part_kdtrees]
        labels[i] = int(np.argmin(dmins)) + 1  # categories start at 1
    return labels

def _load_vertex_labels(cfg: SystemConfig, mesh: trimesh.Trimesh) -> np.ndarray:
    mode = cfg.label_mode
    tried: List[str] = []
    if mode in ("auto", "mesh_prop"):
        arr = _try_load_vertex_labels_from_mesh_ply(cfg.mesh_path)
        if arr is not None and len(arr) == len(mesh.vertices):
            if cfg.verbose: print("Loaded vertex labels from mesh PLY properties.")
            return arr.astype(np.int32)
        tried.append("mesh_prop")
        if mode == "mesh_prop":
            raise ValueError("Vertex labels not found in mesh PLY ('label'/'vert_label' missing).")
    if mode in ("auto", "npy"):
        arr = _try_load_vertex_labels_npy(cfg.mesh_path)
        if arr is not None and len(arr) == len(mesh.vertices):
            if cfg.verbose: print("Loaded vertex labels from .labels.npy")
            return arr.astype(np.int32)
        tried.append(".labels.npy")
        if mode == "npy":
            raise ValueError(f"Expected labels npy next to mesh: {os.path.splitext(cfg.mesh_path)[0]}.labels.npy")
    if mode in ("auto", "nearest_parts"):
        if cfg.decomposed_mesh_dir is None:
            tried.append("nearest_parts (missing decomposed_mesh_dir)")
        else:
            if cfg.verbose: print("Computing vertex labels by nearest decomposed part meshes...")
            arr = _compute_vertex_labels_by_nearest_parts(mesh, cfg.decomposed_mesh_dir)
            return arr
    raise ValueError("Could not obtain vertex labels. Tried: " + ", ".join(tried))

def _ensure_gaussian_normals(gauss_rec: np.ndarray) -> Optional[np.ndarray]:
    names = gauss_rec.dtype.names
    if names is None:
        return None
    has_n = all(k in names for k in ('nx','ny','nz'))
    if not has_n:
        return None
    n = np.stack([gauss_rec['nx'], gauss_rec['ny'], gauss_rec['nz']], axis=1).astype(np.float64)
    # normalize
    lens = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n = n / lens
    return n

def _load_gaussians(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return (structured vertex array, dict of all per-vertex properties)."""
    ply = PlyData.read(path)
    if 'vertex' not in ply:
        raise ValueError("Input PLY lacks 'vertex' element")
    v = ply['vertex']
    props = {name: np.asarray(v.data[name]) for name in v.data.dtype.names}
    return v.data, props

def _save_points_ply(path: str, props: Dict[str, np.ndarray], idxs: np.ndarray):
    # Slice all properties consistently by idxs and write
    sliced = {}
    for k, arr in props.items():
        sliced[k] = np.asarray(arr)[idxs]
    # Build dtype from original
    names = list(props.keys())
    dtype = [(name, np.asarray(props[name]).dtype) for name in names]
    data = np.empty(len(idxs), dtype=dtype)
    for name in names:
        data[name] = sliced[name]
    elem = PlyElement.describe(data, 'vertex')
    PlyData([elem], text=False).write(path)

def _ray_filter(mesh: trimesh.Trimesh, points: np.ndarray, normals: Optional[np.ndarray],
                max_len: float, eps: float) -> np.ndarray:
    """Keep points whose normal ray intersects the mesh within max_len (both +/- directions)."""
    if normals is None:
        # If normals are absent, keep all (cannot perform this filter)
        return np.ones(len(points), dtype=bool)

    # Use robust ray-mesh intersector
    rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    # Cast both +n and -n
    keep = np.zeros(len(points), dtype=bool)
    origins = np.asarray(points, dtype=np.float64)

    # Split to avoid huge memory
    batch = 4096
    for s in range(0, len(points), batch):
        e = min(len(points), s + batch)
        o = origins[s:e]
        n = normals[s:e]

        for sign in (1.0, -1.0):
            dirs = n * sign
            locs, index_ray, _ = rmi.intersects_location(o, dirs, multiple_hits=False)
            # Map rays to hit distance
            hit_d = np.full(e - s, np.inf)
            if len(index_ray):
                # distance along ray
                d = np.linalg.norm(locs - o[index_ray], axis=1)
                hit_d[index_ray] = d
            keep[s:e] |= (hit_d > eps) & (hit_d < max_len)
    return keep

def _assign_categories(mesh: trimesh.Trimesh, vert_labels: np.ndarray,
                       gauss_xyz: np.ndarray, gauss_n: Optional[np.ndarray],
                       cfg: SystemConfig) -> np.ndarray:
    V = np.asarray(mesh.vertices, dtype=np.float64)
    VN = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tree = cKDTree(V)

    # bandwidth from bbox
    bb = mesh.bounds
    diag = float(np.linalg.norm(bb[1] - bb[0]))
    sigma = max(1e-9, cfg.sigma_rel * diag)

    # for each gaussian, consider k nearest vertices and score
    k = max(1, cfg.k_neighbors)
    assigned = np.zeros(len(gauss_xyz), dtype=np.int32)

    for i, p in enumerate(gauss_xyz):
        dists, idxs = tree.query(p, k=min(k, len(V)))
        if np.isscalar(dists):
            dists = np.array([dists])
            idxs = np.array([idxs], dtype=int)
        # distance kernel
        w_dist = np.exp(-(dists ** 2) / (2.0 * sigma * sigma)) * cfg.w_dist

        # alignment with vertex normals
        if gauss_n is not None:
            n = gauss_n[i]
            aligns = np.maximum(0.0, np.einsum('ij,j->i', VN[idxs], n))  # cos >= 0
            w_align = (aligns ** cfg.align_power) * cfg.w_align
        else:
            w_align = np.ones_like(w_dist)

        score = w_dist * w_align
        # If all zero (rare), fallback to nearest
        j = int(idxs[np.argmax(score)]) if np.any(score > 0) else int(idxs[0])
        assigned[i] = int(vert_labels[j])
    return assigned


# ----------------------------- Main entry -----------------------------

def assign_gaussians_to_categories(cfg: SystemConfig):
    os.makedirs(cfg.gaussian_out_dir, exist_ok=True)

    # Load mesh
    mesh_path = os.path.join(cfg.root_dir, cfg.mesh_path) if not os.path.isabs(cfg.mesh_path) else cfg.mesh_path
    mesh = _load_mesh(mesh_path)

    # Vertex labels
    vert_labels = _load_vertex_labels(cfg, mesh)
    if len(vert_labels) != len(mesh.vertices):
        raise ValueError("Label length does not match number of mesh vertices.")

    # Load Gaussians
    g_path = os.path.join(cfg.root_dir, cfg.gaussian_in_path) if not os.path.isabs(cfg.gaussian_in_path) else cfg.gaussian_in_path
    g_rec, g_props = _load_gaussians(g_path)
    gauss_xyz = np.stack([g_rec['x'], g_rec['y'], g_rec['z']], axis=1).astype(np.float64)
    gauss_n = _ensure_gaussian_normals(g_rec)

    # Filter by ray-mesh exit condition
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    ray_max = cfg.ray_max_rel * diag
    keep_mask = _ray_filter(mesh, gauss_xyz, gauss_n, ray_max, cfg.ray_eps) if cfg.drop_if_ray_miss else np.ones(len(gauss_xyz), dtype=bool)
    if cfg.verbose:
        kept = int(np.count_nonzero(keep_mask))
        print(f"Ray filter: keeping {kept}/{len(gauss_xyz)} Gaussians ({100.0*kept/len(gauss_xyz):.1f}%).")

    # Assign categories using likelihood (distance + alignment)
    assigned = _assign_categories(mesh, vert_labels, gauss_xyz, gauss_n, cfg)

    # Group and save
    unique_cats = np.unique(assigned[keep_mask])
    if cfg.verbose:
        print(f"Saving categories: {unique_cats.tolist()} to {cfg.gaussian_out_dir}")

    # Precompute indices per category
    for cat in unique_cats:
        idxs = np.nonzero(keep_mask & (assigned == cat))[0]
        if len(idxs) == 0:
            continue
        out_name = f"gaussian_{int(cat)}.ply"
        out_path = os.path.join(cfg.gaussian_out_dir, out_name)
        _save_points_ply(out_path, g_props, idxs)
        if cfg.verbose:
            print(f"Saved {len(idxs)} points -> {out_path}")

    if cfg.verbose:
        print("Done.")


if __name__ == "__main__":
    cfg = SystemConfig()
    assign_gaussians_to_categories(cfg)
