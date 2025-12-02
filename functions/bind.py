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
    max_dist_rel: float = 0.10      # exclude if nearest vertex farther than this * mesh diagonal
    z_cut_rel: float = -0.01        # exclude if Gaussian z < (z_cut_rel * mesh diagonal)
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
            names = el.data.dtype.names
            sliced = np.empty(len(idxs), dtype=el.data.dtype)
            for n in names:
                sliced[n] = el.data[n][idxs]
            new_el = PlyElement.describe(sliced, 'vertex')
            new_elements.append(new_el)
        else:
            new_elements.append(el)
    new_ply = PlyData(new_elements, text=ply.text)
    new_ply.byte_order = ply.byte_order
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
    os.makedirs(cfgs.gaussian_out_dir, exist_ok=True)

    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    mesh = trimesh.load(states.refine.dirs.mesh)
    vert_labels = states.segment.vert_label

    g_ply = PlyData.read(states.transform.dirs.gaussian)
    g_rec = g_ply['vertex'].data
    gauss_xyz = _get_positions(g_rec)

    bind_cfgs = BindCfgs()  # k=3, thresholds from config

    # --- Z cut by relative threshold to mesh diagonal ---
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    z_cut = bind_cfgs.z_cut_rel * diag
    keep_z = gauss_xyz[:, 2] >= z_cut
    if bind_cfgs.verbose:
        print(f"[z-cut] diag={diag:.6g}, z_cut={z_cut:.6g}, kept {int(keep_z.sum())}/{len(gauss_xyz)}")

    # Work only on survivors of z-cut
    xyz_kept = gauss_xyz[keep_z]

    # --- KNN assignment on z-kept points ---
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

    for c in cats:
        idxs = np.nonzero(keep & (assigned == c))[0]
        if idxs.size == 0:
            continue
        out_path = os.path.join(cfgs.gaussian_out_dir, f"gaussian_{int(c)}.ply")
        print(f"Writing {out_path} ({idxs.size} points)")
        _slice_vertex_in_ply(g_ply, idxs).write(out_path)