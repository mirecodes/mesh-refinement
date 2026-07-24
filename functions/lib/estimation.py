from typing import TypedDict, List
import numpy as np
import pymeshlab
import trimesh
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans

from functions.lib.graph import (
    build_adjacency_graph, filter_boundary_vertices, segregate_loops, 
    filter_proximal_vertices, identify_loop_neighbor, merge_loops_by_topology, extract_boundary_loops_robust
)
from functions.lib.visualization import visualize_k_hop_plane, visualize_all_boundary_loops

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
    """Train K linear planes forming a convex ('and') or concave ('or') polyhedral positive region.
    """
    rng = np.random.default_rng(random_state)

    Xp = verts[np.unique(idx_pos)]
    Xn = verts[np.unique(idx_neg)]
    X = np.vstack([Xp, Xn])
    y = np.hstack([np.ones(len(Xp)), -np.ones(len(Xn))])

    scaler = StandardScaler().fit(X)
    Xp_s = scaler.transform(Xp)
    Xn_s = scaler.transform(Xn)

    def _init_weights(_logic: str):
        W = np.zeros((K, Xp_s.shape[1]), dtype=np.float64)
        b = np.zeros(K, dtype=np.float64)
        
        try:
            if _logic == "and" and len(Xn_s) >= K:
                # For AND logic (convex corner), cluster negatives into K directions
                km = KMeans(n_clusters=K, random_state=random_state, n_init=5)
                labels_n = km.fit_predict(Xn_s)
                for j in range(K):
                    sub_neg = Xn_s[labels_n == j]
                    if len(sub_neg) == 0:
                        sub_neg = Xn_s
                    clf_init = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
                    clf_init.fit(np.vstack([Xp_s, sub_neg]), np.hstack([np.ones(len(Xp_s)), -np.ones(len(sub_neg))]))
                    W[j] = clf_init.coef_.ravel()
                    b[j] = clf_init.intercept_[0]
                return W, b
            elif _logic == "or" and len(Xp_s) >= K:
                # For OR logic (concave corner), cluster positives into K branches
                km = KMeans(n_clusters=K, random_state=random_state, n_init=5)
                labels_p = km.fit_predict(Xp_s)
                for j in range(K):
                    sub_pos = Xp_s[labels_p == j]
                    if len(sub_pos) == 0:
                        sub_pos = Xp_s
                    clf_init = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
                    clf_init.fit(np.vstack([sub_pos, Xn_s]), np.hstack([np.ones(len(sub_pos)), -np.ones(len(Xn_s))]))
                    W[j] = clf_init.coef_.ravel()
                    b[j] = clf_init.intercept_[0]
                return W, b
        except Exception:
            pass

        # Fallback single LinearSVC + noise jitter
        base = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
        base.fit(np.vstack([Xp_s, Xn_s]), np.hstack([np.ones(len(Xp_s)), -np.ones(len(Xn_s))]))
        W = np.tile(base.coef_.ravel(), (K, 1)).astype(np.float64)
        b = np.tile(base.intercept_[0], K).astype(np.float64)
        W += 0.05 * rng.standard_normal(W.shape)
        b += 0.05 * rng.standard_normal(b.shape)
        return W, b

    def _em_train_for_logic(_logic: str):
        W, b = _init_weights(_logic)

        assign_neg = [np.array([], dtype=int) for _ in range(K)]
        assign_pos = [np.array([], dtype=int) for _ in range(K)]

        for _ in range(max_iter):
            # E-step
            if _logic == "and":
                # AND logic: All positives belong to every plane
                new_assign_pos = [np.arange(len(Xp_s)) for _ in range(K)]
                # Negatives are assigned to the plane where they are most positive (highest score)
                Sneg = np.stack([Xn_s @ W[j].ravel() + b[j] for j in range(K)], axis=1)  # (Nneg, K)
                jstar_neg = np.argmax(Sneg, axis=1)
                new_assign_neg = [np.where(jstar_neg == j)[0] for j in range(K)]
            else:  # "or"
                # OR logic: Positives choose their best supporting plane (highest score)
                Spos = np.stack([Xp_s @ W[j].ravel() + b[j] for j in range(K)], axis=1)  # (Npos, K)
                jstar_pos = np.argmax(Spos, axis=1)
                new_assign_pos = [np.where(jstar_pos == j)[0] for j in range(K)]
                # Negatives belong to EVERY plane (must be <= 0 for all planes)
                new_assign_neg = [np.arange(len(Xn_s)) for _ in range(K)]

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

                if idxP_j.size == 0 or idxN_j.size == 0:
                    if reinit_when_empty:
                        W[j] += 0.02 * rng.standard_normal(W[j].shape)
                        b[j] += 0.02 * rng.standard_normal(())
                    continue

                X_pos = Xp_s[idxP_j]
                X_neg = Xn_s[idxN_j]
                X_j = np.vstack([X_pos, X_neg])
                y_j = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

                clf = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
                clf.fit(X_j, y_j)
                W[j] = clf.coef_.ravel()
                b[j] = clf.intercept_[0]

        return W, b

    def _eval_model(W_, b_, _logic: str):
        Xs_all = np.vstack([Xp_s, Xn_s])
        y_all = np.hstack([np.ones(len(Xp_s)), -np.ones(len(Xn_s))]).astype(int)
        S_all = np.stack([Xs_all @ W_[j].ravel() + b_[j] for j in range(K)], axis=1)

        if _logic == "and":
            score_comb = S_all.min(axis=1)
        else:
            score_comb = S_all.max(axis=1)

        pred = np.where(score_comb > 0.0, 1, -1)
        pos_mask = (y_all == 1)
        neg_mask = (y_all == -1)

        pos_acc = float(np.mean(pred[pos_mask] == 1)) if np.any(pos_mask) else 0.0
        neg_acc = float(np.mean(pred[neg_mask] == -1)) if np.any(neg_mask) else 0.0
        balanced_acc = 0.5 * (pos_acc + neg_acc)

        margin_avg = float(np.mean(y_all * score_comb))
        total_score = balanced_acc + 1e-4 * margin_avg
        raw_acc = float(np.mean(pred == y_all))
        return total_score, raw_acc, balanced_acc

    if logic in ("and", "or"):
        W, b = _em_train_for_logic(logic)
        selected_logic = logic
    elif logic == "auto":
        W_and, b_and = _em_train_for_logic("and")
        W_or,  b_or  = _em_train_for_logic("or")

        score_and, acc_and, b_acc_and = _eval_model(W_and, b_and, "and")
        score_or,  acc_or,  b_acc_or  = _eval_model(W_or,  b_or,  "or")

        if score_and >= score_or:
            selected_logic, W, b = "and", W_and, b_and
        else:
            selected_logic, W, b = "or", W_or, b_or
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

    sel_total_score, sel_raw_acc, sel_b_acc = _eval_model(W, b, selected_logic)

    return {
        "planes": planes,
        "models": models,
        "scaler": scaler,
        "logic": selected_logic,
        "scores": {"and": sel_raw_acc, "or": sel_raw_acc},
        "score": sel_total_score,
        "balanced_acc": sel_b_acc,
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
    min_size_to_keep: int = 5,
    show_none_loops: bool = False,
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
    print(f"[debug] Raw loops: {len(loops)}")
    for r_idx, r_loop in enumerate(loops):
        print(f"[debug] Raw loop {r_idx} size: {len(r_loop)}")

    # 2. [NEW] Topology 기반 병합 및 필터링
    #    - 서로 2칸(hop) 이내에 있는 루프들은 "같은 경계"로 보고 합칩니다.
    #    - 합쳐진 결과가 15개 점 미만이면 노이즈로 보고 버립니다.
    loops = merge_loops_by_topology(
        adj,
        loops,
        max_hops=2,  # 거리가 2 hop 이내면 병합
        min_size_to_keep=min_size_to_keep  # 병합 후에도 너무 작으면 삭제
    )

    print(f"[info] Found {len(loops)} loops for part {idx_part}.")

    if visualize_results and len(loops) > 0:
        loop_neighbors_list = []
        for loop in loops:
            freq_prox_nbrs = identify_loop_neighbor(
                loop, adj, vert_label, idx_part, neighbor_threshold=neighbor_threshold
            )
            loop_neighbors_list.append(freq_prox_nbrs)
        
        visualize_all_boundary_loops(
            verts=verts,
            faces=faces,
            loops=loops,
            loop_neighbors=loop_neighbors_list,
            parent_part=idx_part,
            vert_label=vert_label,
            title=f"All Boundary Loops for Part {idx_part}",
            display=visualize_results,
            show_none_loops=show_none_loops
        )

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
                        loop_indices=loop_idx,
                        parent_part=idx_part
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
