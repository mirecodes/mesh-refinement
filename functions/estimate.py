import numpy as np
import pymeshlab
import trimesh
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from trimesh.proximity import signed_distance
from collections import deque


def build_adjacency_graph(faces, num_verts):
    adj = [set() for _ in range(num_verts)]
    faces = faces.astype(np.int64)
    for a, b, c in faces:
        adj[a].add(b); adj[b].add(a)
        adj[b].add(c); adj[c].add(b)
        adj[c].add(a); adj[a].add(c)
    return [list(s) for s in adj]


def classify_vertices(verts, categories, use_signed_dist=True, sdf_thresh=0.0, batch_size: int=50000) -> np.ndarray:
    """

    Args:
        verts: (N,3) vertices from the original mesh
        categories: list(list(trimesh.Trimesh)) the segmented parts
        use_signed_dist: if Ture, classify based on signed distance (robust on non-watertight mesh)
                            if False, classify using function .contains in watertight parts
        sdf_thresh: signed distance threshold (considered contained when sdf<0)
        batch_size: batch size to use for classification

    Returns:
        vert_label: (N,) part ids for each vertex, -1 unless included in anywhere
    """

    N = len(verts)
    vert_label = -np.ones(N, dtype=np.int32)

    # 겹침 처리: "더 안쪽"을 선택하기 위해 sd 최솟값(가장 작은 값) 추적
    best_sd = np.full(N, np.inf, dtype=np.float32) if use_signed_dist else None

    if use_signed_dist:
        from trimesh.proximity import signed_distance

    # 각 카테고리(파트)
    for cid, meshes in enumerate(categories):
        if not meshes:
            continue

        # 카테고리 전체 AABB로 1차 프리필터 (조각들이 여러 개일 수 있으니 합집합)
        vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        vmax = -vmin
        for pm in meshes:
            vmin = np.minimum(vmin, pm.vertices.min(axis=0))
            vmax = np.maximum(vmax, pm.vertices.max(axis=0))

        in_box = np.all((verts >= vmin) & (verts <= vmax), axis=1)
        cand_idx = np.where(in_box)[0]
        if cand_idx.size == 0:
            continue

        # 조각별로 포함 여부 갱신
        if use_signed_dist:
            # signed distance로 "가장 안쪽(가장 작은 sd)" 조각을 선택
            for pm in meshes:
                # 조각 AABB 프리필터로 후보 추가 축소
                pm_vmin = pm.vertices.min(axis=0)
                pm_vmax = pm.vertices.max(axis=0)
                mask_pm = np.all((verts[cand_idx] >= pm_vmin) & (verts[cand_idx] <= pm_vmax), axis=1)
                sub_idx = cand_idx[mask_pm]
                if sub_idx.size == 0:
                    continue

                # 배치로 sd 계산
                for s in range(0, sub_idx.size, batch_size):
                    take = sub_idx[s:s+batch_size]
                    P = verts[take]
                    sd = -signed_distance(pm, P).astype(np.float32)

                    # 포함 기준: sd <= sdf_thresh
                    inside = sd <= sdf_thresh
                    if not np.any(inside):
                        continue

                    # 아직 라벨 없거나, 이번 sd가 더 "안쪽"(작음)이면 갱신
                    upd = inside & (sd < best_sd[take])
                    if np.any(upd):
                        best_sd[take[upd]] = sd[upd]
                        vert_label[take[upd]] = cid

        else:
            # watertight 가정: contains로 빠르게 포함 판정 (겹침은 "먼저 맞은 카테고리 유지")
            # 필요 시 마지막 우선으로 바꾸려면 아래 조건을 vert_label[take][mask] == -1로 유지
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
                    # 현재 미배정인 곳만 라벨 부여(겹침 시 '첫 매치 우선')
                    free = vert_label[take] == -1
                    upd = inside & free
                    if np.any(upd):
                        vert_label[take[upd]] = cid

    return vert_label


def filter_boundary_vertices(adj, vert_label, idx_part):
    seeds = np.where(vert_label == idx_part)[0]
    if len(seeds) == 0:
        return seeds
    mask = np.zeros(len(adj), dtype=bool)
    for u in seeds:
        for v in adj[u]:
            if vert_label[v] != idx_part:
                mask[u] = True
                break
    boundary = seeds[mask[np.array(seeds)]]
    return boundary if len(boundary) > 0 else seeds # return entire vertices when boundary is empty

def filter_proximal_vertices(adj, vert_label, idx_part, max_hops=5):
    num_verts = len(adj)
    INF = max_hops + 1
    dist = np.full(num_verts, INF, dtype=np.int32)

    # seed = boundary vertices
    seeds = filter_boundary_vertices(adj, vert_label, idx_part)

    # multi-source BFS
    deq = deque()
    for s in seeds:
        dist[s] = 0
        deq.append(s)

    while deq:
        u = deq.popleft()
        du = dist[u]
        if du + 1 >= max_hops:      # hop<max_hops 까지만 필요
            continue
        for v in adj[u]:
            if dist[v] == INF:
                dist[v] = du + 1
                deq.append(v)

    # exclude the part vertices
    # part_mask = (vert_label == idx_part)
    # dist[part_mask] = -1

    # the filtered index in the limit
    valid = (dist < max_hops) # (dist != -1) & (dist < max_hops)
    idxs = np.nonzero(valid)[0]
    return idxs, dist


def train_boundary_svm(
        verts: np.ndarray,
        idx_part: np.ndarray,
        idx_neighbor: np.ndarray,
        method: str = "linear",
        C: float = 1.0,
        gamma: str | float = "scale",
        balance: str = "downsample",
        random_state: int | None = 0
):
    """

    Args:
        verts: (V, 3) vertices
        idx_part: indices of the part
        idx_neighbor: indices of the neighbor
        method: SVM training methods: 'linear' | 'rbf'
        C: hyperparameter
        gamma: hyperparameter
        balance: 'downsample' | 'weights'
        random_state: random seed
    Returns:
        clf, scaler
    """

    pos = np.unique(idx_part) # target part -> +
    neg = np.unique(idx_neighbor) # neighbor part -> -
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Not enough positive/negative samples for SVM.")

    # balance between the classes
    rng = np.random.default_rng(0)

    if balance == 'downsample':
        k = min(len(pos), len(neg))
        pos = rng.choice(pos, k, replace=False)
        neg = rng.choice(neg, k, replace=False)

    # prepare the training set
    Xp, Xn= verts[pos], verts[neg]
    X  = np.vstack([Xp, Xn])
    y  = np.hstack([np.ones(len(Xp)), -np.ones(len(Xn))])

    # normalization
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # train the svm
    if method == "linear":
        clf = LinearSVC(C=C, class_weight=("balanced" if balance in ("weights", "downsample") else None), max_iter=5000)
    elif method == "rbf":
        clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight=("balanced" if balance in ("weights", "downsample") else None), random_state=random_state)
    else:
        raise ValueError(f"Undefined svm algorithm: {method}")
    clf.fit(Xs, y)
    return clf, scaler


def learn_separator_main(
    ms: pymeshlab.MeshSet,
    categories: list[list[trimesh.Trimesh]],
    idx_part: int,
    max_hops: int = 5,
    method: str = "linear",
    C: float = 1.0,
    gamma: str | float = "scale",
    use_signed_dist: bool = True,
    sdf_thresh: float = 0.0,
    balance: str = 'downsample',
    random_state: int | None = 0
):
    """

    Args:
        ms: meshset
        categories: list of parts included in same category
        idx_part: indices of the part
        max_hops: the maximum number of hops
        method: SVM training methods: 'linear' | 'rbf'
        C: hyperparameter
        gamma: hyperparameter
        use_signed_dist: if Ture, classify based on signed distance
        sdf_thresh: signed distance threshold (considered contained when sdf<0)
        balance: 'downsample' | 'weights' | 'none'
        random_state: random seed

    Returns:
        { 'clf, 'scalar', 'method', 'plane', 'idx_pos', 'idx_neg', 'idx_neighbor', 'dist'}
    """

    # obtain set (V, E) form original mesh
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    # obtain adjacency graph
    adj = build_adjacency_graph(faces, len(verts))

    # vertice labeling based on parts
    if not 0 < idx_part < len(categories):
        raise ValueError("[error]: idx_part must be in [1, len(parts))]")
    vert_label = classify_vertices(verts, categories, use_signed_dist, sdf_thresh)



    # verify seeds
    idx_seeds = np.where(vert_label == idx_part)[0]
    if len(idx_seeds) == 0:
        raise RuntimeError("[error]: Target part has no included vertices. Check watertight/SDF settings.")

    # find boundary neighbors where hop<k (BFS)
    idx_neighbor, dist = filter_proximal_vertices(adj, vert_label, idx_part, max_hops)

    # bind pos / neg points
    neighbor_label = vert_label[idx_neighbor]
    idx_pos = idx_neighbor[neighbor_label == idx_part]
    idx_neg = idx_neighbor[(neighbor_label > 0) & (neighbor_label != idx_part)]

    if len(idx_pos) == 0 or len(idx_neg) == 0:
        raise RuntimeError("[error]: Not enough positives/negatives in the neighborhood for SVM learning.")

    # train SVM

    if method == "polyhedral":
        poly = train_polyhedral_planes(verts, idx_pos, idx_neg, K=2, C=C)
        planes = poly["planes"]
        return {
            "clf": None,
            "scaler": poly["scaler"],
            "method": method,
            "plane": planes,  # 여러 평면
            "idx_pos": np.unique(idx_pos),
            "idx_neg": np.unique(idx_neg),
            "idx_neighbor": idx_neighbor,
            "dist": dist
        }

    clf, scaler = train_boundary_svm(
        verts, idx_pos, idx_neg,
        method, C, gamma, balance, random_state
    )

    # when svm uses linear method, reconstruct the plane
    plane = None
    if method == 'linear':
        w_s = clf.coef_.ravel()
        b_s = clf.intercept_[0]
        w = w_s / scaler.scale_
        b = b_s - np.dot(w, scaler.mean_)
        n = w / np.linalg.norm(w)
        d = b / np.linalg.norm(w)  # 평면: n·x + d = 0
        plane = (n, d)


    return {
        "clf": clf,
        "scaler": scaler,
        "method": method,
        "plane": plane,
        "idx_pos": np.unique(idx_pos),
        "idx_neg": np.unique(idx_neg),
        "idx_neighbor": idx_neighbor,
        "dist": dist
    }

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def train_polyhedral_planes(
    verts, idx_pos, idx_neg,
    K=2,                    # 평면 개수(2~3 권장)
    max_iter=20,            # 교대 최적화 반복 수
    C=1.0,                  # SVM 규제
    random_state=0,
    reinit_when_empty=True  # 특정 평면에 음성이 비면 재초기화
):
    """
    verts: (N,3)
    idx_pos, idx_neg: 원본 버텍스 인덱스(양/음)
    return: dict with {'planes': [(n1,d1),...], 'models': [clf_j], 'scaler': scaler}
    """
    rng = np.random.default_rng(random_state)

    # 데이터 구성 (+ 표준화)
    Xp = verts[np.unique(idx_pos)]
    Xn = verts[np.unique(idx_neg)]
    X  = np.vstack([Xp, Xn])
    y  = np.hstack([np.ones(len(Xp)), -np.ones(len(Xn))])

    scaler = StandardScaler().fit(X)
    Xp_s = scaler.transform(Xp)
    Xn_s = scaler.transform(Xn)

    # 초기화: K개의 평면 파라미터 (LinearSVC로 러프하게 시작)
    # 간단히: 전체 데이터로 하나 돌리고, 랜덤 노이즈로 K개 복제
    base = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state).fit(
        np.vstack([Xp_s, Xn_s]), np.hstack([np.ones(len(Xp_s)), -np.ones(len(Xn_s))])
    )
    W = np.tile(base.coef_.ravel(), (K,1)).astype(np.float64)
    b = np.tile(base.intercept_[0], K).astype(np.float64)
    W += 0.05 * rng.standard_normal(W.shape)
    b += 0.05 * rng.standard_normal(b.shape)

    # 음성 배정 인덱스(리스트의 리스트)
    assign = [np.array([], dtype=int) for _ in range(K)]

    for it in range(max_iter):
        # --- E: 음성 배정 (가장 큰 s_j(x)=w_j^T x + b_j) ---
        S = np.stack([Xn_s @ W[j].ravel() + b[j] for j in range(K)], axis=1)  # (N_neg, K)
        jstar = np.argmax(S, axis=1)  # 최대 위반 평면
        new_assign = [np.where(jstar == j)[0] for j in range(K)]

        # 수렴 체크
        if it > 0 and all(np.array_equal(a, na) for a, na in zip(assign, new_assign)):
            break
        assign = new_assign

        # --- M: 각 평면별로 SVM 재학습 (양성 전체 vs 해당 평면의 음성) ---
        for j in range(K):
            idxN_j = assign[j]
            if idxN_j.size == 0:
                if reinit_when_empty:
                    # 음성이 한 명도 없으면 약간 회전시켜 유지 (완전 소실 방지)
                    W[j] += 0.01 * rng.standard_normal(W[j].shape)
                    b[j] += 0.01 * rng.standard_normal(())
                continue

            X_pos = Xp_s
            X_neg = Xn_s[idxN_j]
            X_j   = np.vstack([X_pos, X_neg])
            y_j   = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])

            clf = LinearSVC(C=C, class_weight="balanced", max_iter=3000, random_state=random_state)
            clf.fit(X_j, y_j)
            # 저장(스케일 복원은 나중에 한 번에)
            W[j] = clf.coef_.ravel()
            b[j] = clf.intercept_[0]

    # 표준화 역변환으로 (n_j, d_j) 복원: n = W_j / scale, d = b - n·mean
    planes = []
    models = []
    for j in range(K):
        w_s = W[j]
        b_s = b[j]
        w = w_s / scaler.scale_
        b0 = b_s - np.dot(w, scaler.mean_)
        n = w / (np.linalg.norm(w) + 1e-12)
        d = b0 / (np.linalg.norm(w) + 1e-12)
        planes.append((n, d))
        # 원하면 모델 객체로도 보관 (예측/의사결정값 용도)
        m = LinearSVC(C=C, class_weight="balanced", max_iter=1)  # dummy shell
        m.coef_ = w_s.reshape(1, -1)
        m.intercept_ = np.array([b_s])
        models.append(m)

    return {"planes": planes, "models": models, "scaler": scaler}