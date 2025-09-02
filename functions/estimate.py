from typing import TypedDict

import numpy as np
import pymeshlab
import trimesh
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from functions.graphs import build_adjacency_graph, classify_vertices, filter_proximal_vertices, \
    filter_boundary_vertices, segregate_loops


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
) -> list[TypedDict]:
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

    # (4) Get boundaries
    boundary_indices = filter_boundary_vertices(adj, vert_label, idx_part)
    loops = segregate_loops(adj, boundary_indices)

    # loops: segregate_loops(adj, boundary_indices)의 결과 (List[np.ndarray])
    # visualize_boundary_loops(
    #     ms, categories, idx_part, loops,
    #     show_original_mesh=True,  # 원본 메쉬 켜기
    #     tube_radius=0.001,
    #     sphere_radius=0.004,
    #     background="white",
    # )


    results = list()

    for order, loop in enumerate(loops):
        # (4) Neighborhood (hop < K)
        idx_neighbor, dist = filter_proximal_vertices(adj, vert_label, loop, idx_part, max_hops)

        # (5) Positive/negative split
        neighbor_label = vert_label[idx_neighbor]
        idx_pos = idx_neighbor[neighbor_label == idx_part]
        idx_neg = idx_neighbor[(neighbor_label > 0) & (neighbor_label != idx_part)]

        # find the most frequent contact neighbor
        idx_nbr_dist1, dist1 = filter_proximal_vertices(adj, vert_label, loop, idx_part, 2)

        prox_nbr_label = vert_label[idx_nbr_dist1]
        idx_prox_nbr = idx_nbr_dist1[(prox_nbr_label > 0) & (prox_nbr_label != idx_part)]
        values, counts = np.unique(vert_label[idx_prox_nbr], return_counts=True)
        freq_prox_nbr = values[np.argmax(counts)]

        # PCA plane normal of the loop points (smallest singular vector)
        loop_idx = np.asarray(loop, dtype=int).ravel()
        pts_loop = verts[loop_idx]
        ctr_loop = pts_loop.mean(axis=0)
        X = pts_loop - ctr_loop
        try:
            _, Svals, Vt = np.linalg.svd(X, full_matrices=False)
            n_pca = Vt[-1]
        except np.linalg.LinAlgError:
            # fallback: covariance eigendecomposition
            C = (X.T @ X) / max(len(X) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(C)
            n_pca = eigvecs[:, np.argmin(eigvals)]
        n_pca = n_pca / (np.linalg.norm(n_pca) + 1e-12)
        pca_normal = n_pca

        # find the loop center
        loop_idx = np.asarray(loop, dtype=int)
        if loop_idx.size == 0:
            raise RuntimeError("[error]: Empty loop indices.")
        center = verts[loop_idx].mean(axis=0)

        if len(idx_pos) == 0 or len(idx_neg) == 0:
            raise RuntimeError("[error]: Not enough positives/negatives in the neighborhood for SVM learning.")

        # (6) Train: polyhedral / single-plane or kernel SVM
        if method == "polyhedral":
            poly = train_polyhedral_planes(verts, idx_pos, idx_neg, K=2, C=C, random_state=random_state)
            results.append( {
                "loop": loop,
                "center": center,
                "pca_normal": pca_normal,
                "clf": None,
                "scaler": poly["scaler"],
                "method": method,
                "plane": poly["planes"],
                "parent": idx_part,
                "child": freq_prox_nbr,
                "idx_pos": np.unique(idx_pos),
                "idx_neg": np.unique(idx_neg),
                "idx_neighbor": idx_neighbor,
                "dist": dist,
            })
        elif method in ("linear", "rbf"):
            clf, scaler = train_boundary_svm(
                verts, idx_pos, idx_neg,
                method=method, C=C, gamma=gamma, balance=balance, random_state=random_state
            )

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

            results.append( {
                "loop": loop,
                "center": center,
                "pca_normal": pca_normal,
                "clf": clf,
                "scaler": scaler,
                "method": method,
                "plane": plane,
                "parent": idx_part,
                "child": freq_prox_nbr,
                "idx_pos": np.unique(idx_pos),
                "idx_neg": np.unique(idx_neg),
                "idx_neighbor": idx_neighbor,
                "dist": dist,
            })
        else:
            raise ValueError("[error]: Unknown method '%s'" % method)

    return results


# TODO: remove temporal functions
# --- Visualization of boundary loops over mesh (vedo) ---
import numpy as np
import vedo

def _trimesh_list_to_vedo_mesh(parts: list[trimesh.Trimesh]) -> vedo.Mesh | None:
    """Convert list of trimesh.Trimesh into a single vedo.Mesh (or None if empty)."""
    if not parts:
        return None
    v_all = []
    f_all = []
    off = 0
    for tm in parts:
        v = np.asarray(tm.vertices, dtype=float)
        f = np.asarray(tm.faces, dtype=int)
        v_all.append(v)
        f_all.append(f + off)
        off += v.shape[0]
    V = np.vstack(v_all)
    F = np.vstack(f_all)
    return vedo.Mesh([V, F])

def visualize_boundary_loops(
    ms: pymeshlab.MeshSet,
    categories: list[list[trimesh.Trimesh]],
    idx_part: int,
    loops: list[np.ndarray],
    *,
    show_original_mesh: bool = True,    # True면 원본 메쉬도 표시
    tube_radius: float = 0.4,           # 루프 튜브 반지름
    sphere_radius: float = 0.8,         # 루프 중심 표시 구 반지름
    background: str = "white",
):
    """
    Visualize boundary loops together with either the original mesh and/or the target part.
    - `loops` : list of arrays of vertex indices (dtype=int)
    """
    # 0) get original mesh vertices/faces from MeshSet(0)
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    # 1) actors container
    actors = []

    # 2) original mesh (semi-transparent)
    if show_original_mesh:
        m_orig = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.25)
        m_orig.lighting("plastic")
        actors.append(m_orig)

    # 3) selected part from categories[idx_part]
    vpart = _trimesh_list_to_vedo_mesh(categories[idx_part])
    if vpart is not None:
        vpart.c("dodgerblue").alpha(0.35).lw(0.5).lighting("plastic")
        actors.append(vpart)

    # 4) draw loops
    cmap = vedo.color_map(range(len(loops)), "Set1")  # distinct colors
    for i, loop in enumerate(loops):
        loop = np.asarray(loop, dtype=int).ravel()
        if loop.size == 0:
            continue

        # 좌표 시퀀스
        pts = verts[loop]

        # 루프가 닫혀 있는지(첫/끝 이웃 여부) 판단
        closed = (loop.size >= 3 and (loop[0] == loop[-1] or loop[0] in set(adjacent for adjacent in [])))
        # 위 closed 여부는 보수적으로 False로 두고, 실제로는 open/closed 모두 그리되,
        # 닫힌 형태로 보고 싶으면 다음 한 줄로 덮어써도 됩니다:
        # closed = (loop.size >= 3 and loop[0] == loop[-1])

        # vedo.Line으로 경계 곡선을 만들고 튜브화
        line = vedo.Line(pts, closed=False).c(cmap[i]).lw(2)
        # Some vedo versions expose Line.tube(...), not .tubes(...)
        if hasattr(line, "tube") and callable(getattr(line, "tube")):
            try:
                tube = line.tube(radius=tube_radius).c(cmap[i])
            except TypeError:
                # Fallback for signatures like tube(r=...)
                tube = line.tube(tube_radius).c(cmap[i])
        else:
            # Final fallback: build a Tube actor from points directly
            tube = vedo.Tube(pts, r=tube_radius, cap=True).c(cmap[i])
        actors.append(tube)

        # 루프 중심 표시
        center = pts.mean(axis=0)
        s = vedo.Sphere(pos=center, r=sphere_radius, res=16).c(cmap[i]).alpha(1.0)

    # 5) axes (version-safe) & show
    plt = vedo.Plotter(bg=background, title="Boundary loops")
    _axes_target = m_orig if show_original_mesh else (vpart or vedo.Mesh([verts, faces]))
    try:
        axes_actor = vedo.Axes(_axes_target, axesType=4, xyGrid=True)
    except TypeError:
        try:
            # Some vedo versions don't accept axesType
            axes_actor = vedo.Axes(_axes_target, xyGrid=True)
        except TypeError:
            # Fallback minimal Axes
            axes_actor = vedo.Axes(_axes_target)
    plt.show(actors + [axes_actor], viewup="z").close()