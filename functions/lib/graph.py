import numpy as np
import trimesh
from collections import deque, defaultdict
from typing import Iterable, List, Tuple, Dict
from functions.lib.geometry import normalize

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
    k_vote: int = 3,                      # <- 추가: 최근접 이웃 개수
) -> np.ndarray:
    # TODO: Improve the classification algorithm
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

    # ------------------------------
    # (NEW) 최근접 k개 라벨-정점 투표 보정
    # ------------------------------
    unlabeled = np.where(vert_label == -1)[0]
    if unlabeled.size == 0:
        return vert_label

    labeled = np.where(vert_label >= 0)[0]
    if labeled.size == 0:
        # 라벨 있는 정점이 전혀 없으면 보정 불가
        return vert_label

    k = int(min(k_vote, labeled.size))

    def vote_assign(u_idx: np.ndarray, nn_idx: np.ndarray, nn_dist: np.ndarray) -> np.ndarray:
        """
        u_idx: (M,) 미분류 정점 인덱스
        nn_idx: (M,k) labeled 배열 내 인덱스(로컬 인덱스)
        nn_dist: (M,k) 거리
        반환: (M,) 할당 라벨
        """
        # 로컬 인덱스를 글로벌 정점 인덱스로 변환
        neigh_global = labeled[nn_idx]             # (M,k)
        neigh_labels = vert_label[neigh_global]    # (M,k)
        out = np.full(u_idx.shape[0], -1, dtype=np.int32)

        for m in range(u_idx.shape[0]):
            labs = neigh_labels[m]
            dists = nn_dist[m]
            # 다수결
            vals, cnts = np.unique(labs, return_counts=True)
            # 최빈도 후보들
            maxc = np.max(cnts)
            cand = vals[cnts == maxc]
            if cand.size == 1:
                out[m] = int(cand[0])
            else:
                # 동표면 후보들 중 거리 합이 최소인 라벨 선택
                best_lab = None
                best_sum = np.inf
                for L in cand:
                    s = float(np.sum(dists[labs == L]))
                    if s < best_sum:
                        best_sum = s
                        best_lab = int(L)
                out[m] = best_lab if best_lab is not None else int(cand[0])
        return out

    # 우선 SciPy KDTree 시도
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts[labeled])
        # query 결과가 k==1이면 1D, 그 외 2D → 2D로 통일
        dists, idxs = tree.query(verts[unlabeled], k=k, workers=-1)
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        assign = vote_assign(unlabeled, idxs, dists)
        vert_label[unlabeled] = assign
        return vert_label
    except Exception:
        # SciPy가 없거나 실패하면 배치 브루트포스(메모리 안전)
        B = 8192  # 배치 크기
        for s in range(0, unlabeled.size, B):
            take = unlabeled[s:s+B]
            U = verts[take]                  # (b,3)
            L = verts[labeled]              # (L,3)
            # 거리 제곱 (메모리 고려해 배치로 충분히 작게)
            # b x L 행렬에서 각 행마다 k개 최솟값의 인덱스를 얻는다
            # numpy.argpartition 사용
            # (b, L) 거리 계산
            # (U - L)^2 = |U|^2 + |L|^2 - 2 U·L
            UU = np.sum(U*U, axis=1, keepdims=True)         # (b,1)
            LL = np.sum(L*L, axis=1, keepdims=True).T       # (1,L)
            d2 = UU + LL - 2.0 * (U @ L.T)                  # (b,L)
            # k개 이웃의 로컬 인덱스
            part = np.argpartition(d2, kth=k-1, axis=1)[:, :k]  # (b,k) 순서는 정렬X
            # 해당 거리 추출 후 정렬
            row_idx = np.arange(part.shape[0])[:, None]
            dsel = d2[row_idx, part]
            order = np.argsort(dsel, axis=1)
            nn_idx = part[row_idx, order]                    # (b,k)
            nn_d2  = dsel[row_idx, order]
            assign = vote_assign(take, nn_idx, np.sqrt(np.maximum(nn_d2, 0.0)))
            vert_label[take] = assign
        return vert_label

# ---------------------------------------------------------------------
# Neighborhood & boundary helpers
# ---------------------------------------------------------------------
def connected_components_subset(subset_idxs: Iterable[int], adj):
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
    seeds: np.ndarray, # loop
    idx_part: int,
    max_hops: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    V = len(adj)
    INF = max_hops + 1
    dist = np.full(V, INF, dtype=np.int32)
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

def segregate_loops(adj: List[List[int]], boundary_indices: np.ndarray) -> List[np.ndarray]:
    """
    Extract boundary loops robustly:
      1) Build boundary subgraph (+ optional 2-core peel)
      2) Enumerate simple cycles (no edge consumption)
      3) Select long, low-overlap cycles (greedy by length & new-edge coverage)
    Return loops sorted by descending length.
    """
    # -------------------- local toggles --------------------
    PEEL_CORE          = True     # remove deg<2 iteratively
    DETERMINISTIC      = True     # sorted neighbors
    MIN_LEN            = 4        # drop tiny loops (e.g., triangles)
    MIN_NEW_EDGE_FRAC  = 0.5      # keep a cycle only if ≥50% edges are new vs. already selected
    # -------------------------------------------------------

    B = set(map(int, np.asarray(boundary_indices, dtype=int).ravel()))
    if not B:
        return []

    # boundary-only neighbors & undirected edges
    bneigh = {u: [] for u in B}
    edges: set[tuple[int, int]] = set()
    for u in B:
        for v in adj[u]:
            if v in B and v != u:
                bneigh[u].append(v)
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))
    if not edges:
        return []

    # (optional) 2-core peel to remove spurs
    if PEEL_CORE:
        deg = {u: len(bneigh[u]) for u in B}
        alive = {u: True for u in B}
        q = deque([u for u in B if deg[u] < 2])
        while q:
            u = q.popleft()
            if not alive[u]:
                continue
            alive[u] = False
            for v in list(bneigh[u]):
                e = (u, v) if u < v else (v, u)
                edges.discard(e)
                if u in bneigh[v]:
                    bneigh[v].remove(u)
                deg[v] -= 1
                if alive[v] and deg[v] == 1:
                    q.append(v)
            bneigh[u].clear()
        coreV = {u for u in B if alive[u]}
        if not coreV or not edges:
            return []
    else:
        coreV = B

    # core adjacency from remaining edges
    core_adj = defaultdict(list)
    for a, b in list(edges):
        if a in coreV and b in coreV:
            core_adj[a].append(b)
            core_adj[b].append(a)
        else:
            edges.discard((a, b))
    if not edges:
        return []
    if DETERMINISTIC:
        for u in core_adj:
            core_adj[u].sort()

    def ekey(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    # ---------- 1) enumerate simple cycles (Paton-style, undirected) ----------
    nodes = sorted(core_adj.keys())
    seen_cycles: set[Tuple[int, ...]] = set()
    cycles: list[list[int]] = []

    def canonical_cycle(cyc: list[int]) -> Tuple[int, ...]:
        # remove duplicate end if present
        if len(cyc) >= 2 and cyc[0] == cyc[-1]:
            cyc = cyc[:-1]
        m = min(cyc)
        idxs = [i for i, v in enumerate(cyc) if v == m]
        best = None
        for i0 in idxs:
            # forward
            fwd = cyc[i0:] + cyc[:i0]
            # backward
            bwd = list(reversed(cyc[i0:] + cyc[:i0]))
            cand = tuple(fwd) if tuple(fwd) < tuple(bwd) else tuple(bwd)
            if best is None or cand < best:
                best = cand
        return best

    for u in nodes:
        for v in core_adj[u]:
            if v <= u:  # enforce v>u to reduce duplicates
                continue
            stack = [(u, v, [u, v])]
            while stack:
                root, cur, path = stack.pop()
                # neighbors with id >= root to keep canonical growth
                for w in core_adj[cur]:
                    if w == path[-2]:
                        continue  # don't immediately go back
                    if w == root:
                        if len(path) >= 3:
                            cyc = canonical_cycle(path[:])
                            if cyc not in seen_cycles:
                                seen_cycles.add(cyc)
                                cycles.append(list(cyc))
                        continue
                    # grow only if w >= root and new
                    if w >= root and (w not in path):
                        stack.append((root, w, path + [w]))

    if not cycles:
        return []

    # ---------- 2) select cycles (prefer long, low-overlap) ----------
    # Filter tiny cycles first
    cycles = [c for c in cycles if len(c) >= MIN_LEN]

    # Sort by length desc
    cycles.sort(key=len, reverse=True)

    selected: list[np.ndarray] = []
    used_edges: set[Tuple[int, int]] = set()

    for cyc in cycles:
        cyc_edges = {ekey(a, b) for a, b in zip(cyc, cyc[1:] + cyc[:1])}
        if not cyc_edges:
            continue
        new_edges = cyc_edges - used_edges
        # keep only if it brings enough new edges (prevents tiny loop replacing a big one)
        if len(new_edges) / len(cyc_edges) >= MIN_NEW_EDGE_FRAC:
            selected.append(np.asarray(cyc, dtype=int))
            used_edges |= cyc_edges

    # final sort by length desc
    selected.sort(key=lambda a: -len(a))
    return selected

def identify_loop_neighbor(
    loop: np.ndarray,
    adj: List[List[int]],
    vert_label: np.ndarray,
    parent_label: int
) -> int:
    """
    Identify the neighbor part ID for a given boundary loop.
    Returns the most frequent neighbor label (excluding parent and <=0).
    """
    counts = defaultdict(int)
    for u in loop:
        for v in adj[u]:
            l_v = int(vert_label[v])
            # 0(배경)이나 자기 자신(parent)은 제외하고 카운트
            if l_v > 0 and l_v != parent_label:
                counts[l_v] += 1
    if not counts:
        return -1
    return max(counts, key=counts.get)


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
            comps = connected_components_subset(Ui, adj)
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


def _build_cost_matrix(A: List[Dict], B: List[Dict], w_pos: float, w_ang: float) -> np.ndarray:
    m, n = len(A), len(B)
    C = np.zeros((m, n), dtype=float)
    for i, a in enumerate(A):
        ca = np.asarray(a["center"], dtype=float)
        na = normalize(np.asarray(a["pca_normal"], dtype=float))
        for j, b in enumerate(B):
            cb = np.asarray(b["center"], dtype=float)
            nb = normalize(np.asarray(b["pca_normal"], dtype=float))
            dpos = np.linalg.norm(ca - cb)
            cos_abs = abs(float(np.dot(na, nb)))
            dang = 1.0 - cos_abs
            C[i, j] = w_pos * dpos + w_ang * dang
    return C

def _hungarian_or_greedy(C: np.ndarray) -> List[Tuple[int, int]]:
    """Try SciPy Hungarian; fallback to simple greedy (deterministic)."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        # Greedy: repeatedly pick the smallest remaining entry
        m, n = C.shape
        pairs = []
        used_r = set(); used_c = set()
        # flatten with indices
        flat = [(C[i, j], i, j) for i in range(m) for j in range(n)]
        flat.sort(key=lambda t: t[0])
        for _, i, j in flat:
            if i in used_r or j in used_c:
                continue
            pairs.append((i, j))
            used_r.add(i); used_c.add(j)
            if len(used_r) == min(m, n):
                break
        return pairs

def cluster_reciprocal_loop_pairs(
    results: List[Dict],
    *,
    w_pos: float = 1.0,
    w_ang: float = 0.5,
    cost_max: float | None = None,  # 매칭 허용 상한(선택)
    verbose: bool = False
) -> List[Dict]:
    """
    results: learn_separator_main per-loop 결과 리스트.
      각 원소는 최소한 아래 키를 가짐:
        - 'parent': int, 'child': int
        - 'center': (3,), 'pca_normal': (3,)
        - (원하면 'loop' 등 원본 식별자를 그대로 유지)
    반환: RLP(상호 루프쌍) 리스트. 각 항목 구조:
       {
        'parts': (i, j),                     # 정렬된 파트쌍 (min, max)
        'a': a_dict, 'b': b_dict,            # 대응된 원본 루프(방향 i->j, j->i)
        'center': (3,),                      # 평균 center
        'normal': (3,),                      # 방향 정합 후 평균 정규화
        'cost': float,                       # 매칭 비용
       }
    """
    if verbose:
        print(f"[info] cluster_reciprocal_loop_pairs: Clustering {len(results)} loop results...")

    # 1) 파트쌍 단위로 그룹핑 (무순서 쌍)
    buckets: Dict[Tuple[int, int], Dict[str, List[Dict]]] = {}
    for it in results:
        p, q = int(it["parent"]), int(it["child"])
        if p == q:
            # 자기-루프는 건너뜀
            continue
        key = (p, q) if p < q else (q, p)
        dir_key = "A" if (p < q) else "B"   # A: p->q (작은->큰), B: q->p (큰->작은)
        if key not in buckets:
            buckets[key] = {"A": [], "B": []}
        buckets[key][dir_key].append(it)

    if verbose:
        print(f"[info]   Found {len(buckets)} potential part pairs.")

    RLPs: List[Dict] = []

    # 2) 각 파트쌍마다 최소 비용 매칭
    for idx, ((i, j), grp) in enumerate(buckets.items()):
        A = grp["A"]  # i->j
        B = grp["B"]  # j->i
        
        if verbose:
            print(f"[info]   Processing pair {idx+1}/{len(buckets)}: parts {i} <-> {j}")
            print(f"[info]     Found {len(A)} loops for {i}->{j} and {len(B)} loops for {j}->{i}")
            
        pairs_to_process = [] # list of (a, b, cost)
        
        if A and B:
            C = _build_cost_matrix(A, B, w_pos=w_pos, w_ang=w_ang)
            matched_indices = _hungarian_or_greedy(C)
            for ia, ib in matched_indices:
                pairs_to_process.append((A[ia], B[ib], float(C[ia, ib])))
            
        elif A and not B:
            if verbose:
                print(f"[info]     One-sided (A only): Creating {len(A)} virtual B loops.")
            for a in A:
                # Create virtual B
                b = a.copy()
                b["parent"] = a["child"]
                b["child"] = a["parent"]
                # b["center"] remains same
                if "pca_normal" in a:
                    b["pca_normal"] = -np.asarray(a["pca_normal"])
                b["linear"] = {}
                b["polyhedral"] = {}
                b["loop"] = []
                
                pairs_to_process.append((a, b, 0.0)) # Cost 0 or penalty?
                
        elif B and not A:
            if verbose:
                print(f"[info]     One-sided (B only): Creating {len(B)} virtual A loops.")
            for b in B:
                # Create virtual A
                a = b.copy()
                a["parent"] = b["child"]
                a["child"] = b["parent"]
                if "pca_normal" in b:
                    a["pca_normal"] = -np.asarray(b["pca_normal"])
                a["linear"] = {}
                a["polyhedral"] = {}
                a["loop"] = []
                
                pairs_to_process.append((a, b, 0.0))

        count_new_rlps = 0
        # 3) 페어 생성 (코스트 상한 필터)
        for a, b, cost in pairs_to_process:
            if (cost_max is not None) and (cost > cost_max):
                continue
            
            ca = np.asarray(a["center"], dtype=float)
            cb = np.asarray(b["center"], dtype=float)
            na = normalize(np.asarray(a.get("pca_normal", [0,0,1]), dtype=float))
            nb = normalize(np.asarray(b.get("pca_normal", [0,0,1]), dtype=float))
            
            # normal 방향 정합
            if float(np.dot(na, nb)) < 0.0:
                nb = -nb
            n_pair = normalize(na + nb) if np.linalg.norm(na + nb) > 1e-9 else na
            c_pair = 0.5 * (ca + cb)
            RLPs.append({
                "parts": (i, j),
                "a": a, "b": b,
                "center": c_pair,
                "normal": n_pair,
                "cost": cost,
            })
            count_new_rlps += 1
            
        if verbose:
            print(f"[info]     Synthesized {count_new_rlps} RLPs for pair {i} <-> {j}")
            
    if verbose:
        print(f"[info] cluster_reciprocal_loop_pairs: Done. Found {len(RLPs)} RLPs.")

    # 4) 정렬(코스트 오름차순) 또는 길이 등 다른 기준도 가능
    RLPs.sort(key=lambda d: d["cost"])
    return RLPs
