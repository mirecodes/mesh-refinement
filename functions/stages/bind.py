import os
from dataclasses import dataclass

import numpy as np
import pymeshlab
import trimesh
from json_handler import JsonHandler
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


# ------------------------------ config ----------------------------------------

@dataclass
class BindCfgs:
    k_neighbors: int = 3            # fixed: use k=3
    max_dist_rel: float = 0.20      # exclude if nearest vertex farther than this * mesh diagonal
    z_cut_rel: float = -1000.0        # exclude if Gaussian z < (z_cut_rel * mesh diagonal)
    verbose: bool = True


# ------------------------------ helpers ---------------------------------------

def _get_positions(rec):
    """Robustly extract (N,3) positions from arbitrary PLY vertex fields."""
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


def _slice_vertex_in_ply(ply: PlyData, idxs):
    """Return a new PlyData that keeps only selected vertex rows; preserve all metadata/elements."""
    new_elements = []
    for el in ply.elements:
        if el.name == 'vertex':
            # Use numpy structured array slicing directly
            sliced_data = el.data[idxs]
            new_el = PlyElement.describe(sliced_data, 'vertex')
            new_elements.append(new_el)
        else:
            new_elements.append(el)
            
    new_ply = PlyData(new_elements, text=ply.text, byte_order=ply.byte_order)
    new_ply.comments = list(ply.comments)
    new_ply.obj_info = list(ply.obj_info)
    return new_ply


# ------------------------------ core mapping ----------------------------------

def _assign_categories_knn3(mesh, vert_labels, gauss_xyz, k=3, max_dist_rel=0.02):
    """
    k=3 K-NN majority vote on vertex labels.
    - If nearest distance > max_dist_rel * diag: assign -1 (exclude).
    - Tie-breaker: pick the label of the closest among tied labels.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    labels = np.asarray(vert_labels, dtype=np.int32)
    if len(V) == 0:
        raise ValueError("Mesh has no vertices.")

    tree = cKDTree(V)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_abs = float(max_dist_rel) * diag

    # Query all at once for speed; handle cases where len(V) < k
    k_eff = min(int(k), len(V))
    dists, idxs = tree.query(gauss_xyz, k=k_eff)

    # Ensure shapes (N,) -> (N,1)
    if k_eff == 1:
        dists = dists[:, None]
        idxs  = idxs[:, None]

    N = len(gauss_xyz)
    out = np.full(N, -1, dtype=np.int32)  # -1 means "excluded"

    for i in range(N):
        # distance threshold: exclude far points
        nearest = dists[i, 0]
        if not np.isfinite(nearest) or nearest > max_abs:
            continue

        neigh_idx = idxs[i]
        neigh_labels = labels[neigh_idx]

        # majority vote
        uniq, counts = np.unique(neigh_labels, return_counts=True)
        winner_count = counts.max()
        winners = uniq[counts == winner_count]

        if len(winners) == 1:
            out[i] = int(winners[0])
        else:
            # tie-breaker: choose label of the closest neighbor among tied labels
            order = np.argsort(dists[i])  # ascending distance
            for j in order:
                cand = int(neigh_labels[j])
                if cand in winners:
                    out[i] = cand
                    break
    return out


# ------------------------------ main ------------------------------------------

def stage_bind(cfgs):
    """
    Minimal bind:
      - Read mesh + vertex labels
      - Read Gaussian PLY
      - Z-cut: drop Gaussians with z < z_cut_rel * diag
      - Map remaining Gaussians via k=3 KNN (majority)
      - Drop Gaussians whose nearest vertex is too far
      - Save per-part PLYs, preserving all original fields/elements
    """
    print(f"[Info] Creating output directory: {cfgs.gaussian_out_dir}")
    os.makedirs(cfgs.gaussian_out_dir, exist_ok=True)

    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    
    mesh_path = states.refine.dirs.mesh
    print(f"[Info] Loading mesh from: {mesh_path}")
    
    # [Fix] process=False is crucial!
    # Trimesh defaults to process=True, which merges vertices and changes indices.
    # This causes mismatch with vert_labels which are based on the original file indices.
    mesh = trimesh.load(mesh_path, process=False)
    
    vert_labels = states.segment.vert_label
    print(f"[Info] Loaded vertex labels: {len(vert_labels)} labels")

    # [Check] Validate counts
    if len(mesh.vertices) != len(vert_labels):
        print(f"[Error] Vertex count mismatch! Mesh: {len(mesh.vertices)}, Labels: {len(vert_labels)}")
        print("[Error] This will cause incorrect binding. Please check if the mesh file matches the labels.")
        # We proceed, but the result will likely be wrong.

    gaussian_path = states.transform.dirs.gaussian
    print(f"[Info] Reading Gaussian from: {gaussian_path}")
    if not os.path.exists(gaussian_path):
        print(f"[Error] Gaussian file not found at: {gaussian_path}")
        return

    g_ply = PlyData.read(gaussian_path)
    g_rec = g_ply['vertex'].data
    gauss_xyz = _get_positions(g_rec)
    print(f"[Info] Loaded {len(gauss_xyz)} Gaussian points")

    bind_cfgs = BindCfgs()  # k=3, thresholds from config

    # --- Z cut by relative threshold to mesh diagonal ---
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    z_cut = bind_cfgs.z_cut_rel * diag
    keep_z = gauss_xyz[:, 2] >= z_cut
    if bind_cfgs.verbose:
        print(f"[z-cut] diag={diag:.6g}, z_cut={z_cut:.6g}, kept {int(keep_z.sum())}/{len(gauss_xyz)}")

    # Work only on survivors of z-cut
    xyz_kept = gauss_xyz[keep_z]

    if len(xyz_kept) == 0:
        print("[Warning] No Gaussian points kept after Z-cut.")
        return

    # --- KNN assignment on z-kept points ---
    print(f"[Info] Running KNN assignment for {len(xyz_kept)} points...")
    assigned_kept = _assign_categories_knn3(
        mesh, vert_labels, xyz_kept,
        k=bind_cfgs.k_neighbors,
        max_dist_rel=bind_cfgs.max_dist_rel
    )

    # Rebuild full assignment array with -1 default
    assigned = np.full(len(gauss_xyz), -1, dtype=np.int32)
    assigned[keep_z] = assigned_kept

    if bind_cfgs.verbose:
        uniq, cnts = np.unique(assigned, return_counts=True)
        print("[assign-knn3] label counts (including -1 excluded):", dict(zip(uniq.tolist(), cnts.tolist())))

    # Save per-category except -1
    keep = (assigned != -1)
    cats = np.unique(assigned[keep])
    
    print(f"[Info] Found {len(cats)} categories to save: {cats}")

    if len(cats) == 0:
        print("[Warning] No valid categories found to save (all points excluded or no labels assigned).")

    for c in cats:
        idxs = np.nonzero(keep & (assigned == c))[0]
        if idxs.size == 0:
            continue
        out_path = os.path.join(cfgs.gaussian_out_dir, f"gaussian_{int(c)}.ply")
        print(f"[Info] Saving category {c} to {out_path} with {idxs.size} points")
        try:
            _slice_vertex_in_ply(g_ply, idxs).write(out_path)
            print(f"[Success] Saved {out_path}")
        except Exception as e:
            print(f"[Error] Failed to write {out_path}: {e}")
