import numpy as np
import pymeshlab
import trimesh
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from functions.graphs import build_adjacency_graph, classify_vertices, filter_proximal_vertices


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