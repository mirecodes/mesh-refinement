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
    logic: str = "auto",            # 'and' | 'or' | 'auto'
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

    def _em_train_for_logic(_logic: str):
        # Initialize by a single LinearSVC, then replicate with noise
        base = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
        base.fit(np.vstack([Xp_s, Xn_s]), np.hstack([np.ones(len(Xp_s)), -np.ones(len(Xn_s))]))
        W = np.tile(base.coef_.ravel(), (K, 1)).astype(np.float64)
        b = np.tile(base.intercept_[0], K).astype(np.float64)
        W += 0.05 * rng.standard_normal(W.shape)
        b += 0.05 * rng.standard_normal(b.shape)

        # assignments
        assign_neg = [np.array([], dtype=int) for _ in range(K)]
        assign_pos = [np.array([], dtype=int) for _ in range(K)]  # only used for 'or'

        for _ in range(max_iter):
            # E-step
            # scores for negatives
            Sneg = np.stack([Xn_s @ W[j].ravel() + b[j] for j in range(K)], axis=1)  # (Nneg,K)
            jstar_neg = np.argmax(Sneg, axis=1)  # most positive plane
            new_assign_neg = [np.where(jstar_neg == j)[0] for j in range(K)]

            if _logic == "or":
                # positives choose their best supporting plane
                Spos = np.stack([Xp_s @ W[j].ravel() + b[j] for j in range(K)], axis=1)  # (Npos,K)
                jstar_pos = np.argmax(Spos, axis=1)
                new_assign_pos = [np.where(jstar_pos == j)[0] for j in range(K)]
            else:
                new_assign_pos = [np.arange(len(Xp_s)) for _ in range(K)]  # all positives to every plane

            # Convergence check
            if (all(np.array_equal(a, na) for a, na in zip(assign_neg, new_assign_neg)) and
                all(np.array_equal(a, na) for a, na in zip(assign_pos, new_assign_pos))):
                assign_neg, assign_pos = new_assign_neg, new_assign_pos
                break

            assign_neg, assign_pos = new_assign_neg, new_assign_pos

            # M-step
            for j in range(K):
                idxP_j = assign_pos[j]
                idxN_j = assign_neg[j]
                if idxN_j.size == 0 and idxP_j.size == 0:
                    continue
                if idxN_j.size == 0 and reinit_when_empty:
                    # small jitter to avoid dead plane
                    W[j] += 0.01 * rng.standard_normal(W[j].shape)
                    b[j] += 0.01 * rng.standard_normal(())
                    continue

                X_pos = Xp_s if _logic != "or" else Xp_s[idxP_j]  # 'and': all positives; 'or': assigned positives
                X_neg = Xn_s[idxN_j] if idxN_j.size > 0 else np.empty((0, Xp_s.shape[1]), dtype=Xp_s.dtype)
                if X_pos.size == 0:
                    # if no positives assigned (can happen in 'or'), skip update
                    continue
                X_j = np.vstack([X_pos, X_neg])
                y_j = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

                clf = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
                clf.fit(X_j, y_j)
                W[j] = clf.coef_.ravel()
                b[j] = clf.intercept_[0]

        return W, b

    if logic in ("and", "or"):
        W, b = _em_train_for_logic(logic)
        selected_logic = logic
    elif logic == "auto":
        # Train both and pick by accuracy on training set
        W_and, b_and = _em_train_for_logic("and")
        W_or,  b_or  = _em_train_for_logic("or")
        # Evaluate
        Xs_all = np.vstack([Xp_s, Xn_s])
        y_all = np.hstack([np.ones(len(Xp_s)), -np.ones(len(Xn_s))]).astype(int)
        def _acc(W_, b_):
            S_all = np.stack([Xs_all @ W_[j].ravel() + b_[j] for j in range(K)], axis=1)
            pred_and = np.where(S_all.min(axis=1) > 0.0, 1, -1)
            acc_and = float(np.mean(pred_and == y_all))
            pred_or  = np.where(S_all.max(axis=1) > 0.0, 1, -1)
            acc_or  = float(np.mean(pred_or == y_all))
            return acc_and, acc_or
        acc_and, acc_or = _acc(W_and, b_and)
        acc_and2, acc_or2 = _acc(W_or, b_or)
        # Pick the best (consider both logic with their own training)
        candidates = [
            ("and", W_and, b_and, acc_and),
            ("or",  W_and, b_and, acc_or),
            ("and", W_or,  b_or,  acc_and2),
            ("or",  W_or,  b_or,  acc_or2),
        ]
        selected_logic, W, b, selected_score = max(candidates, key=lambda t: t[3])
    else:
        raise ValueError(f"Unknown logic: {logic}")

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

    # ---- Evaluate selected weights with both logics
    Xs_all = np.vstack([Xp_s, Xn_s])
    y_all = np.hstack([np.ones(len(Xp_s)), -np.ones(len(Xn_s))]).astype(int)
    S_all = np.stack([Xs_all @ W[j].ravel() + b[j] for j in range(K)], axis=1)
    acc_and = float(np.mean(np.where(S_all.min(axis=1) > 0.0, 1, -1) == y_all))
    acc_or  = float(np.mean(np.where(S_all.max(axis=1) > 0.0, 1, -1) == y_all))
    selected_score = acc_and if selected_logic == "and" else acc_or

    return {
        "planes": planes,
        "models": models,
        "scaler": scaler,
        "logic": selected_logic,
        "scores": {"and": acc_and, "or": acc_or},
        "score": selected_score,
    }

def margin_score(clf, Xs, y):
    """
    clf: 학습된 SVM (LinearSVC or SVC)
    Xs: scaler로 변환된 feature
    y: {+1, -1} 라벨
    """
    scores = clf.decision_function(Xs)
    margins = y * scores   # >0: 올바른 분류, <0: 오분류
    clipped_margins = np.clip(margins, 0, None) # or np.clip(margins, 0, None)
    # 평균 margin을 점수로 사용 (음수면 패널티)
    return np.mean(margins)


# -----------------------------------------------------------------------------
# Main pipeline entry
# -----------------------------------------------------------------------------

def learn_separator_main(
    ms: pymeshlab.MeshSet,
    categories: list[list[trimesh.Trimesh]],
    idx_part: int,
    adj: list[list[int]],
    vert_label: np.ndarray,
    max_hops: int = 5,
    method: str | list[str] = "all",   # 'linear' | 'rbf' | 'polyhedral' | 'all' | list of these
    C: float = 1.0,
    gamma: str | float = "scale",
    use_signed_dist: bool = True,
    sdf_thresh: float = 0.0,
    balance: str = "downsample",      # 'downsample' | 'weights' | 'none'
    random_state: int | None = 0,
) -> list[TypedDict]:
    """
    End-to-end learning of separating boundaries for a selected part.

    Returns a list of result dicts; for each boundary loop, one or more variants are appended
    depending on `method`:
      - 'linear'      → one plane: [(n, d)]
      - 'polyhedral'  → K planes: [(n1, d1), ..., (nK, dK)]
      - 'rbf'         → [] (no linear plane)
      - 'all' or list → include multiple variants per loop (one entry per method)

    Each result dict contains:
      'loop', 'center', 'pca_normal', 'clf', 'scaler', 'method', 'plane',
      'parent', 'child', 'idx_pos', 'idx_neg', 'idx_neighbor', 'dist'
    """
    # (0) Original mesh
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    # (3) Seeds check
    idx_seeds = np.where(vert_label == idx_part)[0]
    if len(idx_seeds) == 0:
        raise RuntimeError("[error]: Target part has no included vertices. Check watertight/SDF settings.")

    # (4) Get boundaries
    boundary_indices = filter_boundary_vertices(adj, vert_label, idx_part)
    loops = segregate_loops(adj, boundary_indices)

    # visualize_boundary_loops(ms=ms,
    #     categories=categories,
    #     idx_part=idx_part,
    #     loops=loops,
    #     boundary_indices=boundary_indices,  # ← 함께 표시
    #     show_original_mesh=True,
    #     tube_radius=0.010,
    #     sphere_radius=0.005,
    #     boundary_point_size=0.005,
    #     boundary_point_color="red",
    #     background="white",
    # )

    # normalize `method` to a list of methods to run
    if isinstance(method, (list, tuple)):
        methods_to_run = [m for m in method]
    elif method == "all":
        methods_to_run = ["linear", "rbf", "polyhedral"]
    else:
        methods_to_run = [str(method)]

    results = list()

    # (5) Iterate through the entire loops
    for order, loop in enumerate(loops):
        # (A) Find the neighborhood vertices (hop < K)
        idx_neighbor, dist = filter_proximal_vertices(adj, vert_label, loop, idx_part, max_hops)

        # (B) Split positive, negative points
        neighbor_label = vert_label[idx_neighbor]
        idx_pos = idx_neighbor[neighbor_label == idx_part]
        idx_neg = idx_neighbor[(neighbor_label > 0) & (neighbor_label != idx_part)]

        # (C) Find the most frequently contact neighbor
        idx_nbr_dist1, dist1 = filter_proximal_vertices(adj, vert_label, loop, idx_part, 2)
        prox_nbr_label = vert_label[idx_nbr_dist1]
        idx_prox_nbr = idx_nbr_dist1[(prox_nbr_label > 0) & (prox_nbr_label != idx_part)]
        values, counts = np.unique(vert_label[idx_prox_nbr], return_counts=True)
        freq_prox_nbr = values[np.argmax(counts)]

        # (D) Estimate PCA plane normal of the loop points (smallest singular vector)
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

        # (E) Find the loop center
        loop_idx = np.asarray(loop, dtype=int)
        if loop_idx.size == 0:
            raise RuntimeError("[error]: Empty loop indices.")
        center = verts[loop_idx].mean(axis=0)

        if len(idx_pos) == 0 or len(idx_neg) == 0:
            raise RuntimeError("[error]: Not enough positives/negatives in the neighborhood for SVM learning.")

        # (F) Train one or more variants and append all results

        result_linear = {}
        result_rbf = {}
        result_polyhedral = {}

        # evaluation metric
        pos = np.unique(idx_pos)
        neg = np.unique(idx_neg)
        Xp, Xn = verts[pos], verts[neg]
        X = np.vstack([Xp, Xn])
        y = np.hstack([np.ones(len(Xp)), -np.ones(len(Xn))]).astype(int)


        if "linear" in methods_to_run:
            clf, scaler = train_boundary_svm(
                verts, idx_pos, idx_neg,
                method="linear", C=C, gamma=gamma, balance=balance, random_state=random_state
            )
            w_s = clf.coef_.ravel()
            b_s = clf.intercept_[0]
            w = w_s / scaler.scale_
            b = b_s - np.dot(w, scaler.mean_)
            norm = np.linalg.norm(w) + 1e-12
            n = w / norm
            d = b / norm
            plane = [(n, d)]

            # === build evaluation set for this loop (same split used for training) ===
            Xs = scaler.transform(X)
            score = float(margin_score(clf, Xs, y))

            result_linear = {
                "method": "linear",
                "clf": clf,
                "scaler": scaler,
                "plane": plane,
                "score": score,
            }

        if "rbf" in methods_to_run:
            clf, scaler = train_boundary_svm(
                verts, idx_pos, idx_neg,
                method="rbf", C=C, gamma=gamma, balance=balance, random_state=random_state
            )
            Xs = scaler.transform(X)
            score = float(margin_score(clf, Xs, y))

            result_rbf = {
                "method": "rbf",
                "clf": clf,
                "scaler": scaler,
                "plane": [],
                "score": score,
            }

        if "polyhedral" in methods_to_run:
            poly = train_polyhedral_planes(verts, idx_pos, idx_neg, K=2, C=C, random_state=random_state)
            plane = poly["planes"]
            clf = None
            scaler = poly["scaler"]
            score = poly["score"]

            result_polyhedral = {
                "method": "polyhedral",
                "clf": clf,
                "scaler": scaler,
                "plane": plane,
                "score": score,
            }

        results.append({
            # joint information
            "parent": idx_part,
            "child": freq_prox_nbr,
            # loop characteristics
            "loop": loop,
            "center": center,
            "pca_normal": pca_normal,
            # division backgrounds
            "vert_label": vert_label,
            "idx_pos": np.unique(idx_pos),
            "idx_neg": np.unique(idx_neg),
            "idx_neighbor": idx_neighbor,
            "dist": dist,
            # plane information
            "linear": result_linear,
            "rbf": result_rbf,
            "polyhedral": result_polyhedral,
        })

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
    boundary_indices: np.ndarray | None = None,  # ★ 추가: 경계 정점 인덱스
    show_original_mesh: bool = True,
    tube_radius: float = 0.4,
    sphere_radius: float = 0.8,
    boundary_point_size: float = 6.0,           # ★ 추가: 경계 포인트 크기
    boundary_point_color = "black",             # ★ 추가: 경계 포인트 색
    background: str = "white",
):
    """
    Visualize boundary loops; optionally overlay boundary vertex indices as points.
    - loops : list of arrays of vertex indices (each loop = ordered vertex ids)
    - boundary_indices : (B,) array of vertex indices to draw as points (optional)
    """
    # 0) get original mesh geometry
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    actors = []

    # 1) original mesh (semi-transparent)
    if show_original_mesh:
        m_orig = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.25)
        m_orig.lighting("plastic")
        actors.append(m_orig)

    # 2) selected part mesh (if any)
    vpart = _trimesh_list_to_vedo_mesh(categories[idx_part])
    if vpart is not None:
        vpart.c("dodgerblue").alpha(0.35).lw(0.5).lighting("plastic")
        actors.append(vpart)

    if len(loops) < 1: return

    # 3) draw loops (as tubes) + loop centers (small spheres)
    cmap = vedo.color_map(range(len(loops)), "Set1")  # distinct colors
    for i, loop in enumerate(loops):
        loop = np.asarray(loop, dtype=int).ravel()
        if loop.size == 0:
            continue
        pts = verts[loop]
        line = vedo.Line(pts, closed=False).c(cmap[i]).lw(2)
        if hasattr(line, "tube") and callable(getattr(line, "tube")):
            try:
                tube = line.tube(radius=tube_radius).c(cmap[i])
            except TypeError:
                tube = line.tube(tube_radius).c(cmap[i])
        else:
            tube = vedo.Tube(pts, r=tube_radius, cap=True).c(cmap[i])
        actors.append(tube)

        center = pts.mean(axis=0)
        s = vedo.Sphere(pos=center, r=sphere_radius, res=16).c(cmap[i]).alpha(0.5)
        actors.append(s)

    # 4) boundary indices → 확실히 보이도록 Spheres 글리프로
    if boundary_indices is not None and len(boundary_indices) > 0:
        bi = np.asarray(boundary_indices, int)
        bi = bi[(bi >= 0) & (bi < verts.shape[0])]
        if bi.size > 0:
            # 빨간 구로 또렷하게
            dots = vedo.Spheres(verts[bi], r=boundary_point_size, c="red", alpha=1.0)
            # 포인트가 표면에 파묻혀 보여도 구는 확실히 보입니다.
            actors.append(dots)

    # 5) axes & show
    plt = vedo.Plotter(bg=background, title="Boundary loops (+ boundary points)")
    _axes_target = actors[0] if show_original_mesh else (vpart or vedo.Mesh([verts, faces]))
    try:
        axes_actor = vedo.Axes(_axes_target, axesType=4, xyGrid=True)
    except TypeError:
        try:
            axes_actor = vedo.Axes(_axes_target, xyGrid=True)
        except TypeError:
            axes_actor = vedo.Axes(_axes_target)
    plt.show(actors + [axes_actor], viewup="z").close()

