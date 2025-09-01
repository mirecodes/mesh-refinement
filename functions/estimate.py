import numpy as np
import pymeshlab
import trimesh
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


# -----------------------------------------------------------------------------
# Graph utilities
# -----------------------------------------------------------------------------

def build_adjacency_graph(faces: np.ndarray, num_verts: int) -> list[list[int]]:
    """Build an undirected vertex adjacency list from triangular faces.

    Args:
        faces: (F, 3) int array of face indices.
        num_verts: number of vertices in the original mesh.
    Returns:
        Adjacency list: list of neighbor index lists for each vertex [0..V-1].
    """
    adj = [set() for _ in range(num_verts)]
    faces = faces.astype(np.int64, copy=False)
    for a, b, c in faces:
        adj[a].add(b); adj[b].add(a)
        adj[b].add(c); adj[c].add(b)
        adj[c].add(a); adj[a].add(c)
    return [list(s) for s in adj]


# -----------------------------------------------------------------------------
# Labeling: assign each original vertex to a category of parts
# -----------------------------------------------------------------------------

def classify_vertices(
    verts: np.ndarray,
    categories: list[list[trimesh.Trimesh]],
    use_signed_dist: bool = True,
    sdf_thresh: float = 0.0,
    batch_size: int = 50000,
) -> np.ndarray:
    """Classify original vertices by membership in categories of part meshes.

    Notes on signed distance convention:
    - `trimesh.proximity.signed_distance(mesh, P)` often returns positive for
      inside and negative for outside. We negate it here so that "inside" tends
      to be negative; then inclusion check becomes `sd <= sdf_thresh` (default 0).

    Args:
        verts: (V, 3) original mesh vertices.
        categories: list of categories, each category is a list of trimesh meshes.
        use_signed_dist: if True, use signed distance (robust to non-watertight).
                          if False, use `contains` (requires watertight meshes).
        sdf_thresh: threshold on (negated) signed distance for inclusion.
        batch_size: batch size for distance/contains evaluation.
    Returns:
        vert_label: (V,) int32, category id per vertex; -1 if not included.
    """
    V = len(verts)
    vert_label = -np.ones(V, dtype=np.int32)

    # Track best (most negative) distance per vertex to resolve overlaps
    best_sd = np.full(V, np.inf, dtype=np.float32) if use_signed_dist else None

    if use_signed_dist:
        # Import locally to avoid unused import when not needed
        from trimesh.proximity import signed_distance as tm_signed_distance

    for cid, meshes in enumerate(categories):
        if not meshes:
            continue

        # Category-level AABB prefilter
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
            # For each piece in the category compute (negated) signed distance
            for pm in meshes:
                pm_vmin = pm.vertices.min(axis=0)
                pm_vmax = pm.vertices.max(axis=0)
                mask_pm = np.all((verts[cand_idx] >= pm_vmin) & (verts[cand_idx] <= pm_vmax), axis=1)
                sub_idx = cand_idx[mask_pm]
                if sub_idx.size == 0:
                    continue

                for s in range(0, sub_idx.size, batch_size):
                    take = sub_idx[s:s + batch_size]
                    P = verts[take]
                    # Negate so that inside tends to be negative
                    sd = -tm_signed_distance(pm, P).astype(np.float32)

                    inside = sd <= sdf_thresh
                    if not np.any(inside):
                        continue

                    # Update label if this piece is "more inside" (smaller sd)
                    upd = inside & (sd < best_sd[take])
                    if np.any(upd):
                        best_sd[take[upd]] = sd[upd]
                        vert_label[take[upd]] = cid
        else:
            # Watertight path using `contains`; first-match policy
            for pm in meshes:
                pm_vmin = pm.vertices.min(axis=0)
                pm_vmax = pm.vertices.max(axis=0)
                mask_pm = np.all((verts[cand_idx] >= pm_vmin) & (verts[cand_idx] <= pm_vmax), axis=1)
                sub_idx = cand_idx[mask_pm]
                if sub_idx.size == 0:
                    continue

                for s in range(0, sub_idx.size, batch_size):
                    take = sub_idx[s:s + batch_size]
                    P = verts[take]
                    inside = pm.contains(P)
                    if not np.any(inside):
                        continue
                    free = vert_label[take] == -1
                    upd = inside & free
                    if np.any(upd):
                        vert_label[take[upd]] = cid

    return vert_label


# -----------------------------------------------------------------------------
# Neighborhood queries (boundary seeds + multi-source BFS)
# -----------------------------------------------------------------------------

def filter_boundary_vertices(adj: list[list[int]], vert_label: np.ndarray, idx_part: int) -> np.ndarray:
    """Return boundary vertices of `idx_part` (or all part vertices if boundary is empty)."""
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


def label_by_separator(verts: np.ndarray, separator) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute separating scores and labels from either:
    - plane (n, d): score = n·x + d
    - list of planes [(n, d), ...]: use the first plane (common in linear/polyhedral APIs)
    - svm dict: {'clf': fitted_sklearn_svm, 'scaler': fitted_StandardScaler}
                score = clf.decision_function(scaler.transform(x))
    Returns:
        scores: (V,) float array
        labels: (V,) int array in {1, 2} by sign of score (>=0 -> 1, <0 -> 2)
    """
    # Accept a list of planes by taking the first one
    if isinstance(separator, list) and len(separator) > 0 and len(separator[0]) == 2:
        n, d = np.asarray(separator[0][0], dtype=float), float(separator[0][1])
        s = verts @ n + d
    elif isinstance(separator, tuple) and len(separator) == 2:
        n, d = np.asarray(separator[0], dtype=float), float(separator[1])
        s = verts @ n + d
    elif isinstance(separator, dict) and "clf" in separator and "scaler" in separator:
        scaler = separator["scaler"]
        clf = separator["clf"]
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
    """
    1) Split mesh by separator into labels {1,2}
    2) Seed = vertex in idx_part closest to the separator (min |score|)
    3) Boundary = boundary vertices of idx_part
    Returns:
        vert_label: (V,)
        seed_idx: int
        boundary: (K,)
    """
    scores, vert_label = label_by_separator(verts, separator)
    adj = build_adjacency_graph(faces, len(verts))

    part_verts = np.where(vert_label == idx_part)[0]
    if part_verts.size == 0:
        return vert_label, -1, np.array([], dtype=int)

    seed_idx = int(part_verts[np.argmin(np.abs(scores[part_verts]))])
    boundary = filter_boundary_vertices(adj, vert_label, idx_part)
    return vert_label, seed_idx, boundary


def filter_proximal_vertices(
    adj: list[list[int]],
    vert_label: np.ndarray,
    idx_part: int,
    max_hops: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-source BFS from boundary seeds; return indices with hop < max_hops.

    Note: we keep part vertices in the index set and filter later by labels.
    """
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


# -----------------------------------------------------------------------------
# Trainers: single-plane SVM and polyhedral (multi-plane) classifier
# -----------------------------------------------------------------------------

def train_boundary_svm(
    verts: np.ndarray,
    idx_part: np.ndarray,
    idx_neighbor: np.ndarray,
    method: str = "linear",
    C: float = 1.0,
    gamma: str | float = "scale",
    balance: str = "downsample",   # 'downsample' | 'weights' | 'none'
    random_state: int | None = 0,
):
    """Train a linear or RBF SVM on balanced or weighted samples.

    Args:
        verts: (V, 3) vertex positions.
        idx_part: indices of positive class (target part).
        idx_neighbor: indices of negative class (neighboring parts).
        method: 'linear' or 'rbf'.
        C, gamma: SVM hyperparameters.
        balance: 'downsample' (sample-size balancing), 'weights' (class_weight), 'none'.
        random_state: RNG seed.
    Returns:
        (clf, scaler)
    """
    pos = np.unique(idx_part)
    neg = np.unique(idx_neighbor)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Not enough positive/negative samples for SVM.")

    rng = np.random.default_rng(random_state)

    if balance == "downsample":
        k = min(len(pos), len(neg))
        pos = rng.choice(pos, k, replace=False)
        neg = rng.choice(neg, k, replace=False)

    Xp, Xn = verts[pos], verts[neg]
    X = np.vstack([Xp, Xn])
    y = np.hstack([np.ones(len(Xp)), -np.ones(len(Xn))])

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    class_w = ("balanced" if balance in ("weights", "downsample") else None)
    if method == "linear":
        clf = LinearSVC(C=C, class_weight=class_w, max_iter=5000)
    elif method == "rbf":
        clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_w, random_state=random_state)
    else:
        raise ValueError(f"Undefined SVM method: {method}")

    clf.fit(Xs, y)
    return clf, scaler


def train_polyhedral_planes(
    verts: np.ndarray,
    idx_pos: np.ndarray,
    idx_neg: np.ndarray,
    K: int = 2,
    max_iter: int = 20,
    C: float = 1.0,
    random_state: int = 0,
    reinit_when_empty: bool = True,
) -> dict:
    """Train K linear planes forming a convex polyhedral positive region.

    Alternating scheme: assign negatives to the most-violating plane, then
    retrain each plane with all positives vs the assigned negatives.

    Returns a dict with keys: 'planes' (list[(n,d)]), 'models' (list[LinearSVC]), 'scaler'.
    """
    rng = np.random.default_rng(random_state)

    Xp = verts[np.unique(idx_pos)]
    Xn = verts[np.unique(idx_neg)]
    X = np.vstack([Xp, Xn])
    y = np.hstack([np.ones(len(Xp)), -np.ones(len(Xn))])

    scaler = StandardScaler().fit(X)
    Xp_s = scaler.transform(Xp)
    Xn_s = scaler.transform(Xn)

    # Initialize by a single LinearSVC, then replicate with noise
    base = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
    base.fit(np.vstack([Xp_s, Xn_s]), np.hstack([np.ones(len(Xp_s)), -np.ones(len(Xn_s))]))
    W = np.tile(base.coef_.ravel(), (K, 1)).astype(np.float64)
    b = np.tile(base.intercept_[0], K).astype(np.float64)
    W += 0.05 * rng.standard_normal(W.shape)
    b += 0.05 * rng.standard_normal(b.shape)

    assign = [np.array([], dtype=int) for _ in range(K)]

    for _ in range(max_iter):
        # E-step: assign each negative to the plane with largest score
        S = np.stack([Xn_s @ W[j].ravel() + b[j] for j in range(K)], axis=1)
        jstar = np.argmax(S, axis=1)
        new_assign = [np.where(jstar == j)[0] for j in range(K)]

        # Convergence check
        if all(np.array_equal(a, na) for a, na in zip(assign, new_assign)):
            break
        assign = new_assign

        # M-step: retrain each plane with all positives vs its assigned negatives
        for j in range(K):
            idxN_j = assign[j]
            if idxN_j.size == 0:
                if reinit_when_empty:
                    W[j] += 0.01 * rng.standard_normal(W[j].shape)
                    b[j] += 0.01 * rng.standard_normal(())
                continue

            X_pos = Xp_s
            X_neg = Xn_s[idxN_j]
            X_j = np.vstack([X_pos, X_neg])
            y_j = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

            clf = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
            clf.fit(X_j, y_j)
            W[j] = clf.coef_.ravel()
            b[j] = clf.intercept_[0]

    # Recover planes in original coordinates: n·x + d = 0
    planes = []
    models = []
    for j in range(K):
        w_s = W[j]
        b_s = b[j]
        w = w_s / scaler.scale_
        b0 = b_s - np.dot(w, scaler.mean_)
        norm = np.linalg.norm(w) + 1e-12
        n = w / norm
        d = b0 / norm
        planes.append((n, d))

        m = LinearSVC(C=C, class_weight="balanced", max_iter=1)
        m.coef_ = w_s.reshape(1, -1)
        m.intercept_ = np.array([b_s])
        models.append(m)

    return {"planes": planes, "models": models, "scaler": scaler}


# -----------------------------------------------------------------------------
# Main pipeline entry
# -----------------------------------------------------------------------------

def learn_separator_main(
    ms: pymeshlab.MeshSet,
    categories: list[list[trimesh.Trimesh]],
    idx_part: int,
    max_hops: int = 5,
    method: str = "linear",          # 'linear' | 'rbf' | 'polyhedral'
    C: float = 1.0,
    gamma: str | float = "scale",
    use_signed_dist: bool = True,
    sdf_thresh: float = 0.0,
    balance: str = "downsample",      # 'downsample' | 'weights' | 'none'
    random_state: int | None = 0,
) -> dict:
    """End-to-end learning of a separating boundary for a selected part.

    Returns dict with keys: 'clf', 'scaler', 'method', 'plane', 'idx_pos',
    'idx_neg', 'idx_neighbor', 'dist'.
    - 'plane' is always a list of (n, d):
        * method == 'linear'  -> one plane: [(n, d)]
        * method == 'polyhedral' -> K planes: [(n1, d1), ..., (nK, dK)]
        * method == 'rbf'     -> [] (no linear plane)
    """
    # (0) Original mesh
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    # (1) Adjacency
    adj = build_adjacency_graph(faces, len(verts))

    # (2) Vertex labeling by categories
    if not 0 < idx_part < len(categories):
        raise ValueError("[error]: idx_part must be in [1, len(parts))]")
    vert_label = classify_vertices(verts, categories, use_signed_dist, sdf_thresh)

    # (3) Seeds check
    idx_seeds = np.where(vert_label == idx_part)[0]
    if len(idx_seeds) == 0:
        raise RuntimeError("[error]: Target part has no included vertices. Check watertight/SDF settings.")

    # (4) Neighborhood (hop < K)
    idx_neighbor, dist = filter_proximal_vertices(adj, vert_label, idx_part, max_hops)

    # (5) Positive/negative split
    neighbor_label = vert_label[idx_neighbor]
    idx_pos = idx_neighbor[neighbor_label == idx_part]
    idx_neg = idx_neighbor[(neighbor_label > 0) & (neighbor_label != idx_part)]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        raise RuntimeError("[error]: Not enough positives/negatives in the neighborhood for SVM learning.")

    # (6) Train: polyhedral / single-plane or kernel SVM
    if method == "polyhedral":
        poly = train_polyhedral_planes(verts, idx_pos, idx_neg, K=2, C=C, random_state=random_state)
        return {
            "clf": None,
            "scaler": poly["scaler"],
            "method": method,
            "plane": poly["planes"],
            "idx_pos": np.unique(idx_pos),
            "idx_neg": np.unique(idx_neg),
            "idx_neighbor": idx_neighbor,
            "dist": dist,
        }

    clf, scaler = train_boundary_svm(
        verts, idx_pos, idx_neg,
        method=method, C=C, gamma=gamma, balance=balance, random_state=random_state
    )

    # (7) Recover plane(s)
    plane = []
    if method == "linear":
        w_s = clf.coef_.ravel()
        b_s = clf.intercept_[0]
        w = w_s / scaler.scale_
        b = b_s - np.dot(w, scaler.mean_)
        norm = np.linalg.norm(w) + 1e-12
        n = w / norm
        d = b / norm
        plane = [(n, d)]
    elif method == "rbf":
        # Kernel SVM has no single separating plane
        plane = []

    return {
        "clf": clf,
        "scaler": scaler,
        "method": method,
        "plane": plane,
        "idx_pos": np.unique(idx_pos),
        "idx_neg": np.unique(idx_neg),
        "idx_neighbor": idx_neighbor,
        "dist": dist,
    }