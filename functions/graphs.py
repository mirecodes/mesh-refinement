import numpy as np
import trimesh
from collections import deque
from typing import Iterable

# ---------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------
def build_adjacency_graph(faces: np.ndarray, num_verts: int) -> list[list[int]]:
    adj = [set() for _ in range(num_verts)]
    faces = faces.astype(np.int64, copy=False)
    for a, b, c in faces:
        adj[a].add(b); adj[b].add(a)
        adj[b].add(c); adj[c].add(b)
        adj[c].add(a); adj[a].add(c)
    return [list(s) for s in adj]

# ---------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------
def classify_vertices(
    verts: np.ndarray,
    categories: list[list[trimesh.Trimesh]],
    use_signed_dist: bool = True,
    sdf_thresh: float = 0.0,
    batch_size: int = 50000,
) -> np.ndarray:
    V = len(verts)
    vert_label = -np.ones(V, dtype=np.int32)
    best_sd = np.full(V, np.inf, dtype=np.float32) if use_signed_dist else None
    if use_signed_dist:
        from trimesh.proximity import signed_distance as tm_signed_distance

    for cid, meshes in enumerate(categories):
        if not meshes:
            continue
        vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        vmax = -vmin
        for pm in meshes:
            vmin = np.minimum(vmin, pm.vertices.min(axis=0))
            vmax = np.maximum(vmax, pm.vertices.max(axis=0))
        mask_cat = np.all((verts >= vmin) & (verts <= vmax), axis=1)
        cand_idx = np.where(mask_cat)[0]
        if cand_idx.size == 0:
            continue

        if use_signed_dist:
            for pm in meshes:
                pm_vmin = pm.vertices.min(axis=0)
                pm_vmax = pm.vertices.max(axis=0)
                mask_pm = np.all((verts[cand_idx] >= pm_vmin) & (verts[cand_idx] <= pm_vmax), axis=1)
                sub_idx = cand_idx[mask_pm]
                if sub_idx.size == 0:
                    continue
                for s in range(0, sub_idx.size, batch_size):
                    take = sub_idx[s:s+batch_size]
                    P = verts[take]
                    sd = -tm_signed_distance(pm, P).astype(np.float32)  # negate
                    inside = sd <= sdf_thresh
                    if not np.any(inside):
                        continue
                    upd = inside & (sd < best_sd[take])
                    if np.any(upd):
                        best_sd[take[upd]] = sd[upd]
                        vert_label[take[upd]] = cid
        else:
            for pm in meshes:
                pm_vmin = pm.vertices.min(axis=0)
                pm_vmax = pm.vertices.max(axis=0)
                mask_pm = np.all((verts[cand_idx] >= pm_vmin) & (verts[cand_idx] <= pm_vmax), axis=1)
                sub_idx = cand_idx[mask_pm]
                if sub_idx.size == 0:
                    continue
                for s in range(0, sub_idx.size, batch_size):
                    take = sub_idx[s:s+batch_size]
                    P = verts[take]
                    inside = pm.contains(P)
                    if not np.any(inside):
                        continue
                    free = vert_label[take] == -1
                    upd = inside & free
                    if np.any(upd):
                        vert_label[take[upd]] = cid
    return vert_label

# ---------------------------------------------------------------------
# Neighborhood & boundary helpers
# ---------------------------------------------------------------------
def _connected_components_subset(subset_idxs: Iterable[int], adj):
    S = set(int(x) for x in subset_idxs)
    seen = set()
    comps = []
    for s in list(S):
        if s in seen:
            continue
        q = deque([s]); seen.add(s)
        comp = []
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if v in S and v not in seen:
                    seen.add(v); q.append(v)
        comps.append(np.array(comp, dtype=int))
    return comps

def filter_boundary_vertices(adj: list[list[int]], vert_label: np.ndarray, idx_part: int) -> np.ndarray:
    seeds = np.where(vert_label == idx_part)[0]
    if len(seeds) == 0:
        return seeds
    is_boundary = np.zeros(len(adj), dtype=bool)
    for u in seeds:
        for v in adj[u]:
            if vert_label[v] != idx_part:
                is_boundary[u] = True
                break
    boundary = seeds[is_boundary[seeds]]
    return boundary if len(boundary) > 0 else seeds

def filter_proximal_vertices(
    adj: list[list[int]],
    vert_label: np.ndarray,
    idx_part: int,
    max_hops: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    V = len(adj)
    INF = max_hops + 1
    dist = np.full(V, INF, dtype=np.int32)
    seeds = filter_boundary_vertices(adj, vert_label, idx_part)
    q = deque()
    for s in seeds:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        du = dist[u]
        if du + 1 >= max_hops:
            continue
        for v in adj[u]:
            if dist[v] == INF:
                dist[v] = du + 1
                q.append(v)
    valid = dist < max_hops
    idxs = np.nonzero(valid)[0]
    return idxs, dist

# ---------------------------------------------------------------------
# Separators & splitting
# ---------------------------------------------------------------------
def label_by_separator(verts: np.ndarray, separator) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(separator, list) and len(separator) > 0 and len(separator[0]) == 2:
        n, d = np.asarray(separator[0][0], dtype=float), float(separator[0][1])
        s = verts @ n + d
    elif isinstance(separator, tuple) and len(separator) == 2:
        n, d = np.asarray(separator[0], dtype=float), float(separator[1])
        s = verts @ n + d
    elif isinstance(separator, dict) and "clf" in separator and "scaler" in separator:
        scaler = separator["scaler"]; clf = separator["clf"]
        Xs = scaler.transform(verts.astype(float))
        s = clf.decision_function(Xs).astype(float)
    else:
        raise ValueError("separator must be (n, d), [(n, d), ...], or {'clf':..., 'scaler':...}")
    labels = np.where(s >= 0.0, 1, 2).astype(int)
    return s, labels

def split_seed_boundary(
    verts: np.ndarray,
    faces: np.ndarray,
    separator,
    idx_part: int = 1
) -> tuple[np.ndarray, int, np.ndarray]:
    scores, vert_label = label_by_separator(verts, separator)
    adj = build_adjacency_graph(faces, len(verts))
    part_verts = np.where(vert_label == idx_part)[0]
    if part_verts.size == 0:
        return vert_label, -1, np.array([], dtype=int)
    seed_idx = int(part_verts[np.argmin(np.abs(scores[part_verts]))])
    boundary = filter_boundary_vertices(adj, vert_label, idx_part)
    return vert_label, seed_idx, boundary

# ---------------------------------------------------------------------
# Boundary → Joint (hop=1 adjacency only)
# ---------------------------------------------------------------------
def build_joints_hop1(verts: np.ndarray, faces: np.ndarray, vert_label: np.ndarray) -> dict:
    """Build joints using only hop=1 boundary adjacency."""
    V = verts.shape[0]
    adj = build_adjacency_graph(faces.astype(int, copy=False), V)
    part_ids = np.unique(vert_label)
    part_ids = part_ids[(part_ids > 0)]  # ignore background 0 if any

    joints = {}
    part_to_joints = {int(pid): [] for pid in part_ids}
    connections = []
    seen_keys = set()
    next_jid = 1

    for i in part_ids:
        i = int(i)
        B_i = filter_boundary_vertices(adj, vert_label, i)
        if B_i.size == 0:
            continue

        neighbors_of_i = {}
        for u in B_i:
            js = set(int(vert_label[v]) for v in adj[u] if vert_label[v] > 0 and vert_label[v] != i)
            for j in js:
                neighbors_of_i.setdefault(j, []).append(u)

        for j, Ui in neighbors_of_i.items():
            Ui = np.unique(np.asarray(Ui, dtype=int))
            if Ui.size == 0:
                continue
            comps = _connected_components_subset(Ui, adj)
            for comp in comps:
                if comp.size == 0:
                    continue
                centroid = verts[comp].mean(axis=0)
                key = (min(i, j), max(i, j), *np.round(centroid, 5))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                jid = f"J{next_jid}"; next_jid += 1
                label_i = f"joint_{len(part_to_joints[i]) + 1}"
                joints[jid] = {
                    'id': jid,
                    'parts': [i, int(j)],
                    'centroid': centroid.tolist(),
                    'size': int(comp.size),
                    'owner': i,
                }
                part_to_joints[i].append({'label': label_i, 'joint_id': jid, 'other': int(j)})
                part_to_joints[int(j)].append({'label': None, 'joint_id': jid, 'other': i})
                connections.append({'part_a': i, 'joint_label_a': label_i, 'part_b': int(j), 'joint_id': jid})

    return {
        'joints': joints,
        'part_to_joints': part_to_joints,
        'connections': connections,
    }