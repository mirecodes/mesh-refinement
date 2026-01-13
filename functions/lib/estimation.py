from typing import TypedDict, List
import numpy as np
import pymeshlab
import trimesh
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from functions.lib.graph import (
    build_adjacency_graph, filter_boundary_vertices, segregate_loops, 
    filter_proximal_vertices, identify_loop_neighbor, extract_boundary_loops_robust
)
from functions.lib.visualization import visualize_k_hop_plane

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
        W_and, b_and = _em_train_for_logic("and")
        W_or,  b_or  = _em_train_for_logic("or")
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
        candidates = [
            ("and", W_and, b_and, acc_and),
            ("or",  W_and, b_and, acc_or),
            ("and", W_or,  b_or,  acc_and2),
            ("or",  W_or,  b_or,  acc_or2),
        ]
        selected_logic, W, b, selected_score = max(candidates, key=lambda t: t[3])
    else:
        raise ValueError(f"Unknown logic: {logic}")

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
    scores = clf.decision_function(Xs)
    margins = y * scores
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
    visualize_results: bool = False,
    neighbor_threshold: float = 0.25, # Added parameter
) -> list[TypedDict]:
    """
    End-to-end learning of separating boundaries for a selected part.
    """
    print(f"[info] Starting learn_separator_main for part {idx_part}...")
    
    # (0) Original mesh
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    # (3) Seeds check
    idx_seeds = np.where(vert_label == idx_part)[0]
    if len(idx_seeds) == 0:
        raise RuntimeError("[error]: Target part has no included vertices. Check watertight/SDF settings.")

    # (4) Get boundaries
    print(f"[info] Finding boundary loops for part {idx_part}...")
    # boundary_indices = filter_boundary_vertices(adj, vert_label, idx_part)
    # loops = segregate_loops(adj, boundary_indices)
    
    # Use extract_boundary_loops_robust instead of segregate_loops
    loops = extract_boundary_loops_robust(verts, faces, idx_seeds)

    print(f"[info] Found {len(loops)} loops for part {idx_part}.")

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
        print(f"[info] Processing loop {order+1}/{len(loops)} (size: {len(loop)})...")
        
        # (A) Find the neighborhood vertices (hop < K)
        idx_neighbor, dist = filter_proximal_vertices(adj, vert_label, loop, idx_part, max_hops)

        # (B) Split positive, negative points
        neighbor_label = vert_label[idx_neighbor]
        idx_pos = idx_neighbor[neighbor_label == idx_part]
        idx_neg = idx_neighbor[(neighbor_label > 0) & (neighbor_label != idx_part)]

        # (C) Find the most frequently contact neighbor(s)
        idx_nbr_dist1, dist1 = filter_proximal_vertices(adj, vert_label, loop, idx_part, 2)
        
        # Use identify_loop_neighbor with threshold
        freq_prox_nbrs = identify_loop_neighbor(
            loop, adj, vert_label, idx_part, neighbor_threshold=neighbor_threshold
        )
        
        # If no neighbors found (should be rare if loop is on boundary), skip or handle
        if not freq_prox_nbrs:
             print(f"[warn] loop {order} has no valid neighbors.")
             continue

        # (D) Estimate PCA plane normal of the loop points
        loop_idx = np.asarray(loop, dtype=int).ravel()
        pts_loop = verts[loop_idx]
        ctr_loop = pts_loop.mean(axis=0)
        X = pts_loop - ctr_loop
        try:
            _, Svals, Vt = np.linalg.svd(X, full_matrices=False)
            n_pca = Vt[-1]
        except np.linalg.LinAlgError:
            # fallback
            C_cov = (X.T @ X) / max(len(X) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(C_cov)
            n_pca = eigvecs[:, np.argmin(eigvals)]
        n_pca = n_pca / (np.linalg.norm(n_pca) + 1e-12)
        pca_normal = n_pca

        # (E) Find the loop center
        loop_idx = np.asarray(loop, dtype=int)
        if loop_idx.size == 0:
            raise RuntimeError("[error]: Empty loop indices.")
        center = verts[loop_idx].mean(axis=0)

        if len(idx_pos) == 0 or len(idx_neg) == 0:
            # raise RuntimeError("[error]: Not enough positives/negatives in the neighborhood for SVM learning.")
            print(f"[warn] loop {order} skipped (not enough pos/neg).")
            continue

        # (F) Train one or more variants
        print(f"[info] Training separators for loop {order+1}...")
        result_linear = {}
        result_rbf = {}
        result_polyhedral = {}

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
        
        # (G) Visualize results if requested
        if visualize_results:
            for method_name in methods_to_run:
                result_data = result_linear if method_name=="linear" else result_rbf if method_name=="rbf" else result_polyhedral
                if result_data and result_data.get("plane"):
                    visualize_k_hop_plane(
                        verts=verts,
                        faces=faces,
                        idx_pos=np.unique(idx_pos),
                        idx_neg=np.unique(idx_neg),
                        planes=result_data["plane"],
                        center=center,
                        title=f"Loop {order}, Method: {method_name}",
                        display=visualize_results,
                        loop_indices=loop_idx # Pass loop indices for visualization
                    )

        # Duplicate result for each identified neighbor
        for nbr in freq_prox_nbrs:
            results.append({
                "parent": idx_part,
                "child": nbr,
                "loop": loop,
                "center": center,
                "pca_normal": pca_normal,
                "vert_label": vert_label,
                "idx_pos": np.unique(idx_pos),
                "idx_neg": np.unique(idx_neg),
                "idx_neighbor": idx_neighbor,
                "dist": dist,
                "linear": result_linear,
                "rbf": result_rbf,
                "polyhedral": result_polyhedral,
            })
    
    print(f"[info] Finished learn_separator_main for part {idx_part}. Generated {len(results)} results.")
    return results
