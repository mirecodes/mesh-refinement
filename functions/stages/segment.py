import os
import numpy as np
import pymeshlab
import trimesh
import vedo
from typing import Dict, List, Tuple
from collections import defaultdict
from functools import partial

from json_handler import JsonHandler
from functions.lib.geometry import normalize, closest_points_between_lines
from functions.lib.graph import build_adjacency_graph, classify_vertices, cluster_reciprocal_loop_pairs
from functions.lib.estimation import learn_separator_main
from functions.lib.mesh import append_patches_to_mesh_files, triangulate_patch_on_loop, save_link_meshes_from_labels
from functions.lib.urdf import build_joints_from_rlps
from functions.lib.visualization import recolor_parts, select_parts_interactive, visualize_and_select_vectors_for_rlps


# =============================================================================
# Pipeline helpers (separation, vectors, patches)
# =============================================================================
def compute_separation_results(ms: pymeshlab.MeshSet,
                               categories: List[List[trimesh.Trimesh]],
                               categories_num: int,
                               adj,
                               vert_label,
                               method: str = "all",
                               visualize_results: bool = False,
                               neighbor_threshold: float = 0.25) -> List[dict]:
    """Run learn_separator_main per category and flatten outputs."""
    results: Dict[int, dict | List[dict]] = {}
    print(f"[info] Starting separation computation for {categories_num} categories...")
    for idx_part in range(1, categories_num + 1):
        print(f"[info] Processing category {idx_part}/{categories_num}...")
        out = learn_separator_main(
            ms, categories, idx_part, adj, vert_label,
            max_hops=10, method=method,
            C=1.0, gamma='scale',
            use_signed_dist=True, sdf_thresh=0.0,
            balance='None',
            visualize_results=visualize_results,
            neighbor_threshold=neighbor_threshold
        )
        results[idx_part] = out
    results_list = [item for v in results.values() for item in (v if isinstance(v, list) else [v])]
    print(f"[info] Separation computation finished. Total results: {len(results_list)}")
    return results_list


def prepare_vectors_for_rlps(rlps: List[dict]) -> None:
    """Create vector candidates on each RLP from plane/normal info."""
    print(f"[info] Preparing vectors for {len(rlps)} RLPs...")
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
        n = normalize(np.mean(np.vstack([nA, nB]), axis=0))
        d = 0.5 * (dA + dB)
        rlp['vectors'] = [{'center': center, 'n': n, 'd': d}]

        A = rlp['a']['polyhedral']
        if A.get('plane') and len(A['plane']) > 0:
            planes = A['plane']; m = len(planes)
            for u in range(m):
                for v in range(u + 1, m):
                    n = np.cross(planes[u][0], planes[v][0])
                    rlp['vectors'].append({'center': center, 'n': n, 'd': np.zeros(3)})
    print("[info] Vector preparation finished.")


def build_extrapoints_and_patches(results_list: List[dict],
                                  verts: np.ndarray):
    """For each loop: dense sampling + triangulated patch."""
    print("[info] Building extra points and patches...")
    patch_vertices: Dict[int, List[np.ndarray]] = defaultdict(list)
    patch_faces: Dict[int, List[np.ndarray]] = defaultdict(list)
    for res in results_list:
        pid = int(res.get('parent', 0))
        loop_idx = res.get('loop', None)
        if pid <= 0 or loop_idx is None or len(loop_idx) < 3:
            continue
        loop_xyz = verts[np.asarray(loop_idx, dtype=int)]
        P3, F = triangulate_patch_on_loop(loop_xyz, step_rel=0.03, sigma_scale=2.0, k_min=6)
        if P3.size and F.size:
            patch_vertices[pid].append(P3.astype(float))
            patch_faces[pid].append(F.astype(np.int64))
    print("[info] Patch building finished.")
    return patch_vertices, patch_faces


# =============================================================================
# Link-pair utilities (used for augmentation only)
# =============================================================================
def build_link_map(rlps: List[dict]) -> Dict[Tuple[int, int], List[int]]:
    """(i,j)(i<j) -> indices of RLPs connecting the two parts."""
    link_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, r in enumerate(rlps):
        p = int(r.get("a", {}).get("parent", r.get("parent", -1)))
        c = int(r.get("a", {}).get("child",  r.get("child",  -1)))
        if p < 0 or c < 0 or p == c:
            continue
        key = (p, c) if p < c else (c, p)
        link_map[key].append(idx)
    return link_map


def pick_normal_from_rlp(rlp: dict) -> np.ndarray | None:
    """Try linear plane normal, then PCA normal."""
    plane = rlp.get("linear", {}).get("plane", None)
    if plane and len(plane) >= 1 and len(plane[0]) >= 1:
        n = np.asarray(plane[0][0], dtype=float)
        if np.isfinite(n).all() and np.linalg.norm(n) > 1e-12:
            return normalize(n)
    n = np.asarray(rlp.get("pca_normal", None), dtype=float) if rlp.get("pca_normal", None) is not None else None
    if n is None or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
        return None
    return normalize(n)


def augment_for_group(rlps: List[dict], group_indices: List[int]) -> None:
    """Add a pair-level vector suggestion into each RLP in the group."""
    if len(group_indices) < 2:
        return
    centers, normals, weights = [], [], []
    for i in group_indices:
        r = rlps[i]
        ctr = np.asarray(r.get("center", None), dtype=float)
        nrm = pick_normal_from_rlp(r)
        if ctr is None or ctr.size != 3 or not np.isfinite(ctr).all():
            continue
        centers.append(ctr); normals.append(nrm)
        loop_idx = r.get("loop", [])
        w = int(len(loop_idx)) if loop_idx is not None else 1
        weights.append(max(w, 1))

    if len(centers) < 2:
        return

    if len(group_indices) == 2:
        c1, c2 = centers[0], centers[1]
        n1 = normals[0] if normals[0] is not None else normalize(c2 - c1)
        n2 = normals[1] if normals[1] is not None else normalize(c1 - c2)
        q1, q2, dist = closest_points_between_lines(c1, n1, c2, n2)
        if dist < 1e-6:
            start = q1
            dirv = normalize(c2 - c1) if np.linalg.norm(c2 - c1) > 1e-12 else normalize((n1 or 0)+(n2 or 0))
        else:
            start = 0.5 * (q1 + q2)
            seg   = q2 - q1
            dirv  = normalize(seg) if np.linalg.norm(seg) > 1e-12 else normalize(c2 - c1)
        vec = {"center": start.astype(float).tolist(), "n": dirv.astype(float).tolist(),
               "state": "None", "source": "pair2"}
        for i in group_indices:
            rlps[i].setdefault("vectors", [])
            rlps[i]["vectors"].append(dict(vec))
        return

    w = np.asarray(weights, dtype=float); W = float(w.sum())
    C = np.vstack(centers)
    start = (w[:, None] * C).sum(axis=0) / max(W, 1e-12)
    Ns = [n for n in normals if n is not None]
    if Ns:
        N = np.vstack(Ns)
        ref = N[0]
        for k in range(len(N)):
            if np.dot(N[k], ref) < 0: N[k] = -N[k]
        navg = normalize(N.sum(axis=0))
    else:
        X = C - C.mean(axis=0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        navg = normalize(Vt[0])

    vec = {"center": start.astype(float).tolist(), "n": navg.astype(float).tolist(),
           "state": "None", "source": "pairN"}
    for i in group_indices:
        rlps[i].setdefault("vectors", [])
        rlps[i]["vectors"].append(dict(vec))


def augment_vectors_for_pairs_with_map(rlps: List[dict], min_count: int = 2) -> None:
    """Add pair-level suggestions only when an (i,j) pair has ≥ min_count RLPs."""
    print("[info] Augmenting vectors for pairs...")
    link_map = build_link_map(rlps)
    for _, idxs in link_map.items():
        if len(idxs) >= min_count:
            augment_for_group(rlps, idxs)
    print("[info] Vector augmentation finished.")


# =============================================================================
# Stage (entry point)
# =============================================================================
def stage_segment(cfgs):
    """
    1) Part selection (UI) → categories
    2) Separation → RLP clustering → vector candidates (+UI selection)
    3) Save per-link meshes from labels
    4) Patch fill (harmonic) and append to meshes
    5) Persist states (states.segment.*)
    """
    print("=" * 60)
    print("[info] Starting stage_segment...")
    
    # load states & base mesh
    print("[info] Loading states and base mesh...")
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    parts = [trimesh.load(states.decompose.dirs.mesh[i]) for i in range(states.decompose.length)]

    ms = pymeshlab.MeshSet()
    mesh_in_dir = states.refine.dirs.mesh
    ms.load_new_mesh(mesh_in_dir)
    ms.set_current_mesh(0)
    current_mesh = ms.current_mesh()
    verts = current_mesh.vertex_matrix()
    faces = current_mesh.face_matrix().astype(np.int64)

    categories_num = cfgs.categories_num
    
    # selection UI
    if cfgs.debug_mode:
        print("=" * 60)
        print("PART SELECTION STAGE CONTROLS:")
        print("  Mouse:")
        print("    Left Click  : Assign part to current selection mode")
        print("    Right Click : Revert assignment")
        print("  Keyboard:")
        print("    Tab         : Cycle selection mode (Category 1 -> 2 -> ... -> 0)")
        print("    Q / Esc     : Finish and Proceed")
        print("=" * 60)

    print("[info] Starting interactive part selection...")
    categories, belongings, rand_colors = select_parts_interactive(parts, verts, faces, categories_num, display=cfgs.debug_mode)
    print("[info] Part selection finished.")
    
    # --- Compact categories by removing empty ones (keeping index 0) ---
    if len(categories) > 1:
        new_categories = [categories[0]]
        for c in categories[1:]:
            if c:
                new_categories.append(c)
        
        if len(new_categories) != len(categories):
            print(f"[info] Removed {len(categories) - len(new_categories)} empty categories.")
            categories = new_categories
            categories_num = len(categories) - 1
            print(f"[info] Updated categories_num: {categories_num}")
    # -------------------------------------------------------------------

    # Manually fill categories if UI skipped (not handled here fully, assumed user wants UI if debug_mode is True)
    # If debug_mode is false, select_parts_interactive returns empty categories. 
    # This might break things if not handled. 
    # Assuming standard flow requires selection. 
    if not any(categories) and categories_num > 0:
         print("[warn] No parts selected or UI skipped. Defaulting to all parts unassigned or similar?")
         # Fallback logic if needed, or just proceed (likely failing later).

    # adjacency, vertex labels
    print("[info] Building adjacency graph and classifying vertices...")
    adj = build_adjacency_graph(faces, len(verts))
    vert_label = classify_vertices(verts, categories, use_signed_dist=True).astype(int)
    print("[info] Graph build and classification finished.")

    # separation + clustering + vectors
    # Pass neighbor_threshold from cfgs if available, else default to 0.25
    neighbor_threshold = getattr(cfgs, 'neighbor_threshold', 0.25)
    
    print("[info] Computing separation results...")
    results_list = compute_separation_results(
        ms, categories, categories_num, adj, vert_label, 
        method="all", visualize_results=cfgs.debug_mode,
        neighbor_threshold=neighbor_threshold
    )
    
    print("[info] Clustering reciprocal loop pairs (RLPs)...")
    rlps = cluster_reciprocal_loop_pairs(results_list, w_pos=1.0, w_ang=0.5, cost_max=None)
    print(f"[info]: len rlps {len(rlps)}")
    
    prepare_vectors_for_rlps(rlps)
    augment_vectors_for_pairs_with_map(rlps, min_count=2)  # optional

    # vector selection UI (RLP-wise)
    if cfgs.debug_mode:
        print("=" * 60)
        print("VECTOR SELECTION STAGE CONTROLS (Per RLP Window):")
        print("  Mouse:")
        print("    Left Click on Vector : Toggle state (None -> Revolute -> Prismatic -> None)")
        print("  Keyboard (Applies to ALL vectors in current window):")
        print("    W / S     : Move along X axis")
        print("    D / A     : Move along Y axis")
        print("    Space / Z : Move along Z axis")
        print("    N / B     : Rotate around X axis")
        print("    H / J     : Rotate around Y axis")
        print("    U / I     : Rotate around Z axis")
        print("    Q / Esc   : Finish current window")
        print("=" * 60)

    print("[info] Starting interactive vector selection...")
    visualize_and_select_vectors_for_rlps(rlps, parts, categories, verts, faces, display=cfgs.debug_mode)
    print("[info] Vector selection finished.")

    # save per-link meshes
    print("[info] Saving link meshes...")
    seg_dir = os.path.join(os.path.dirname(cfgs.json_states_dir), "segment_mesh")
    segment_mesh_paths = save_link_meshes_from_labels(verts, faces, vert_label, categories_num, seg_dir)
    joints = build_joints_from_rlps(rlps)
    print("[info] Link meshes saved.")

    # persist states
    print("[info] Persisting states...")
    states.segment = {}
    states.segment.vert_label = vert_label.tolist()
    states.segment.dirs = {}
    states.segment.dirs.mesh = segment_mesh_paths
    states.segment.joints = joints
    print("[info] States persisted.")

    # patch fill & append
    print("[info] Building and appending patches...")
    patch_vertices, patch_faces = build_extrapoints_and_patches(results_list, verts)
    append_patches_to_mesh_files(segment_mesh_paths, patch_vertices, patch_faces)
    print("[info] Patches appended.")
    
    print("[info] stage_segment finished successfully.")
    print("=" * 60)
