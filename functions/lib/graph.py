import math

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


import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set

import numpy as np
from typing import List, Set


def merge_loops_by_topology(
        adj: List[List[int]],
        loops: List[np.ndarray],
        max_hops: int = 2,
        min_size_to_keep: int = 10  # 병합되지 못한 루프 중 너무 작은 것은 버림
) -> List[np.ndarray]:
    """
    여러 개의 루프(점들의 리스트)를 입력받아,
    그래프 상(Edge Hop)에서 가까운 루프끼리 하나로 '병합'합니다.

    Args:
        adj: 전체 메쉬의 Adjacency list
        loops: extract_boundary_loops_robust 결과물 (List[np.array])
        max_hops: 몇 칸 이내면 같은 덩어리로 볼 것인가 (보통 2~3)
        min_size_to_keep: 병합 후에도 독립적으로 남은 루프가 이보다 작으면 노이즈로 간주하고 삭제

    Returns:
        List[np.ndarray]: 병합된 루프 점들의 리스트.
                          (주의: 병합된 루프는 더 이상 순차적인 경로(Path)가 아니라 점들의 집합(Cloud)이 됩니다.
                           하지만 PCA/SVM 학습에는 순서가 상관없으므로 문제없습니다.)
    """
    if not loops:
        return []

    num_loops = len(loops)

    # 1. 각 루프의 '영향권(Frontier)' 미리 계산 (Set 변환)
    #    매번 BFS를 돌면 느리므로, Set 연산으로 처리
    loop_sets = [set(l) for l in loops]
    expanded_sets = []

    for l_idx, current_set in enumerate(loop_sets):
        # 0-hop (자기 자신)
        frontier = current_set.copy()

        # Expand k-hops
        current_layer = current_set
        for _ in range(max_hops):
            next_layer = set()
            for u in current_layer:
                # adj[u]의 이웃들을 추가
                for v in adj[u]:
                    if v not in frontier:
                        frontier.add(v)
                        next_layer.add(v)
            if not next_layer: break
            current_layer = next_layer

        expanded_sets.append(frontier)

    # 2. 루프 간 인접 행렬 생성 (Meta-Graph)
    #    loop_adjacency[i][j] = True if loop i is close to loop j
    meta_adj = [[] for _ in range(num_loops)]

    for i in range(num_loops):
        for j in range(i + 1, num_loops):
            # i의 확장 영역이 j와 겹치거나, j의 확장 영역이 i와 겹치면 연결
            # (교집합이 있는지 확인)
            if not loop_sets[j].isdisjoint(expanded_sets[i]) or \
                    not loop_sets[i].isdisjoint(expanded_sets[j]):
                meta_adj[i].append(j)
                meta_adj[j].append(i)

    # 3. 연결된 컴포넌트 찾기 (Connected Components)
    visited = [False] * num_loops
    merged_results = []

    for i in range(num_loops):
        if visited[i]:
            continue

        # BFS/DFS로 연결된 루프 그룹 찾기
        component_indices = []
        stack = [i]
        visited[i] = True

        while stack:
            curr = stack.pop()
            component_indices.append(curr)
            for neighbor in meta_adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        # 4. 그룹 내 병합 (Merge Vertices)
        #    여러 루프의 점들을 하나로 합침
        if len(component_indices) == 1:
            # 병합할 것 없음
            final_loop = loops[component_indices[0]]
        else:
            # 여러 개 병합
            arrays_to_merge = [loops[idx] for idx in component_indices]
            final_loop = np.concatenate(arrays_to_merge)
            # 중복 제거 (혹시 겹친다면)
            final_loop = np.unique(final_loop)

        # 5. 크기 필터링 (너무 작은 노이즈 덩어리는 제외)
        if len(final_loop) >= min_size_to_keep:
            merged_results.append(final_loop)

    # 크기순 정렬 (메인 루프가 먼저 오도록)
    merged_results.sort(key=len, reverse=True)

    return merged_results

def extract_boundary_loops_robust(
        verts: np.ndarray,
        faces: np.ndarray,
        part_indices: np.ndarray,
        min_len: int = 4,
        open_loop_threshold: int = 10,
        min_new_edge_frac: float = 0.25  # <--- [ADDED] 겹침 비율 임계값
) -> List[np.ndarray]:
    """
    Mesh의 Face Winding Order를 사용하여 경계 루프를 추출하고,
    '정보량(새로운 엣지 비율)'을 기준으로 유효한 루프만 선별합니다.
    """
    # ---------------------------------------------------------
    # 1. Prepare Boundary Edges (Directed)
    # ---------------------------------------------------------
    mask = np.isin(faces, part_indices).all(axis=1)
    sub_faces = faces[mask]

    if sub_faces.size == 0:
        return []

    edges = np.concatenate([
        sub_faces[:, [0, 1]],
        sub_faces[:, [1, 2]],
        sub_faces[:, [2, 0]]
    ], axis=0)

    # [IMPORTANT] Memory layout fix for view()
    edges = np.ascontiguousarray(edges)

    dtype = [('u', edges.dtype), ('v', edges.dtype)]
    structured_edges = edges.view(dtype=dtype).squeeze()
    unique_edges, counts = np.unique(structured_edges, return_counts=True)

    # (u, v) exists but (v, u) does not => Boundary Edge
    all_directed_edges = set(zip(unique_edges['u'], unique_edges['v']))
    boundary_edges = []
    for u, v in all_directed_edges:
        if (v, u) not in all_directed_edges:
            boundary_edges.append((u, v))

    if not boundary_edges:
        return []

    adj_map = defaultdict(list)
    for u, v in boundary_edges:
        adj_map[u].append(v)

    # ---------------------------------------------------------
    # 2. Walk Paths (Collect Candidates)
    # ---------------------------------------------------------
    visited_edges_during_walk = set()
    candidates: List[Tuple[np.ndarray, bool]] = []  # (path, is_closed)

    start_nodes = list(adj_map.keys())

    for start_node in start_nodes:
        # Check validity
        if not adj_map[start_node]: continue

        # 이미 방문한 엣지에서 시작하는 경우 스킵 (탐색 중복 방지)
        # 단, 완벽한 중복 방지는 아래 Filtering 단계에서 수행하므로 여기선 느슨하게 체크
        valid_start = False
        for neighbor in adj_map[start_node]:
            if (start_node, neighbor) not in visited_edges_during_walk:
                valid_start = True
                break
        if not valid_start:
            continue

        curr = start_node
        path = [curr]

        while True:
            neighbors = adj_map[curr]
            found_next = False
            for nxt in neighbors:
                if (curr, nxt) not in visited_edges_during_walk:
                    visited_edges_during_walk.add((curr, nxt))
                    curr = nxt
                    path.append(curr)
                    found_next = True
                    break

            # Case A: Dead End (Open Loop)
            if not found_next:
                if len(path) >= open_loop_threshold:
                    candidates.append((np.array(path, dtype=int), False))
                break

            # Case B: Cycle Detected (Closed Loop)
            if curr == path[0]:
                loop_nodes = np.array(path[:-1], dtype=int)
                if len(loop_nodes) >= min_len:
                    candidates.append((loop_nodes, True))
                break

            # Safety brake
            if len(path) > len(boundary_edges) + 1:
                break

    # ---------------------------------------------------------
    # 3. Filter Candidates (Selection Logic like segregate_loops)
    # ---------------------------------------------------------
    # Sort by length descending (긴 루프 우선)
    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    selected_loops: List[np.ndarray] = []
    globally_used_edges: Set[Tuple[int, int]] = set()

    def get_edge_key(u: int, v: int) -> Tuple[int, int]:
        """Undirected edge key for overlap check"""
        return (u, v) if u < v else (v, u)

    for path, is_closed in candidates:
        # Build edges for this path
        current_edges = set()

        if is_closed:
            # 닫힌 루프: (0,1), (1,2), ..., (last, 0)
            for i in range(len(path)):
                u, v = path[i], path[(i + 1) % len(path)]
                current_edges.add(get_edge_key(u, v))
        else:
            # 열린 루프: (0,1), (1,2), ..., (n-2, n-1)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                current_edges.add(get_edge_key(u, v))

        if not current_edges:
            continue

        # Check Overlap
        new_edges = current_edges - globally_used_edges

        # [CHECK] 새로운 정보의 비율 확인
        ratio = len(new_edges) / len(current_edges)

        if ratio >= min_new_edge_frac:
            selected_loops.append(path)
            globally_used_edges.update(current_edges)

    return selected_loops


def segregate_loops(adj: List[List[int]], boundary_indices: np.ndarray) -> List[np.ndarray]:
    """
    Extract boundary loops robustly with safety limits to prevent infinite loops.
    """
    # -------------------- local toggles --------------------
    PEEL_CORE = True  # remove deg<2 iteratively
    DETERMINISTIC = True  # sorted neighbors
    MIN_LEN = 4  # drop tiny loops
    MIN_NEW_EDGE_FRAC = 0.5  # overlap filtering

    # [FIX] 무한 루프 방지를 위한 안전장치 추가
    MAX_CYCLES = math.inf  # 찾을 사이클의 최대 개수 (충분히 큰 값)
    MAX_DFS_ITER = math.inf  # DFS 탐색 최대 반복 횟수
    # -------------------------------------------------------

    B = set(map(int, np.asarray(boundary_indices, dtype=int).ravel()))
    if not B:
        return []

    # 1. Build Boundary Subgraph
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

    # 2. (Optional) 2-Core Peel
    if PEEL_CORE:
        deg = {u: len(bneigh[u]) for u in B}
        alive = {u: True for u in B}
        q = deque([u for u in B if deg[u] < 2])
        while q:
            u = q.popleft()
            if not alive[u]: continue
            alive[u] = False
            for v in list(bneigh[u]):
                e = (u, v) if u < v else (v, u)
                edges.discard(e)
                if u in bneigh[v]: bneigh[v].remove(u)
                deg[v] -= 1
                if alive[v] and deg[v] == 1:
                    q.append(v)
            bneigh[u].clear()
        coreV = {u for u in B if alive[u]}
        if not coreV or not edges:
            return []
    else:
        coreV = B

    # 3. Core Adjacency Construction
    core_adj = defaultdict(list)
    for a, b in list(edges):
        if a in coreV and b in coreV:
            core_adj[a].append(b)
            core_adj[b].append(a)

    if DETERMINISTIC:
        for u in core_adj:
            core_adj[u].sort()

    def ekey(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    # -------------------------------------------------------------------------
    # [MODIFIED] 4. Enumerate cycles with Safety Limits
    # -------------------------------------------------------------------------
    nodes = sorted(core_adj.keys())
    seen_cycles: set[Tuple[int, ...]] = set()
    cycles: list[list[int]] = []

    dfs_counter = 0  # 총 탐색 횟수 카운터

    def canonical_cycle(cyc: list[int]) -> Tuple[int, ...]:
        if len(cyc) >= 2 and cyc[0] == cyc[-1]:
            cyc = cyc[:-1]
        m = min(cyc)
        idxs = [i for i, v in enumerate(cyc) if v == m]
        best = None
        for i0 in idxs:
            fwd = cyc[i0:] + cyc[:i0]
            bwd = list(reversed(cyc[i0:] + cyc[:i0]))  # Optimize: create list once
            cand = tuple(fwd) if tuple(fwd) < tuple(bwd) else tuple(bwd)
            if best is None or cand < best:
                best = cand
        return best

    stop_search = False

    for u in nodes:
        if stop_search: break  # 전체 제한 도달 시 중단

        for v in core_adj[u]:
            if v <= u: continue  # undirected unique order

            # Stack: (root, current_node, path_so_far)
            stack = [(u, v, [u, v])]

            while stack:
                # [CHECK 1] DFS 반복 횟수 제한 (너무 깊거나 넓은 탐색 방지)
                dfs_counter += 1
                if dfs_counter > MAX_DFS_ITER:
                    stop_search = True
                    break

                # [CHECK 2] 찾은 사이클 개수 제한
                if len(cycles) >= MAX_CYCLES:
                    stop_search = True
                    break

                root, cur, path = stack.pop()

                # Limit Path Depth (Optional optimization for speed)
                # if len(path) > 500: continue

                for w in core_adj[cur]:
                    if w == path[-2]:
                        continue  # don't go back immediately

                    if w == root:
                        # Cycle Found
                        if len(path) >= 3:
                            # canonical 변환 비용을 줄이기 위해 길이 체크 먼저
                            if len(path) >= MIN_LEN:
                                cyc = canonical_cycle(path[:])
                                if cyc not in seen_cycles:
                                    seen_cycles.add(cyc)
                                    cycles.append(list(cyc))
                        continue

                    # Grow path: w >= root (canonical ordering constraint)
                    if w >= root and (w not in path):
                        stack.append((root, w, path + [w]))

            if stop_search: break

    if not cycles:
        return []

    # -------------------------------------------------------------------------
    # 5. Filter & Select Cycles (Longest, Low Overlap)
    # -------------------------------------------------------------------------
    cycles.sort(key=len, reverse=True)  # Longest first

    selected: list[np.ndarray] = []
    used_edges: set[Tuple[int, int]] = set()

    for cyc in cycles:
        # [MODIFIED] Path를 닫힌 루프로 만들기 위해 마지막 연결부 포함하여 엣지 생성
        # zip(cyc, cyc[1:] + cyc[:1]) -> (0,1), (1,2), ..., (last, 0)
        cyc_edges = {ekey(a, b) for a, b in zip(cyc, cyc[1:] + [cyc[0]])}

        if not cyc_edges:
            continue

        new_edges = cyc_edges - used_edges

        # 분모가 0이 되는 것을 방지하고, 새로운 정보가 많은지 확인
        if len(cyc_edges) > 0:
            ratio = len(new_edges) / len(cyc_edges)
            if ratio >= MIN_NEW_EDGE_FRAC:
                selected.append(np.asarray(cyc, dtype=int))
                used_edges |= cyc_edges

    selected.sort(key=lambda a: -len(a))
    return selected

def identify_loop_neighbor(
    loop: np.ndarray,
    adj: List[List[int]],
    vert_label: np.ndarray,
    parent_label: int,
    neighbor_threshold: float = 0.25
) -> List[int]:
    """
    Identify the neighbor part ID(s) for a given boundary loop.
    Returns a list of neighbor labels that exceed the threshold ratio.
    If no neighbor exceeds the threshold, returns the most frequent one.
    """
    counts = defaultdict(int)
    total_neighbors = 0
    for u in loop:
        for v in adj[u]:
            l_v = int(vert_label[v])
            # 0(배경)이나 자기 자신(parent)은 제외하고 카운트
            if l_v > 0 and l_v != parent_label:
                counts[l_v] += 1
                total_neighbors += 1
    
    if not counts:
        return []

    # Find neighbors exceeding threshold
    valid_neighbors = []
    if total_neighbors > 0:
        for label, count in counts.items():
            if count / total_neighbors >= neighbor_threshold:
                valid_neighbors.append(label)
    
    # If no neighbor exceeds threshold, fallback to the most frequent one
    if not valid_neighbors:
        most_frequent = max(counts, key=counts.get)
        valid_neighbors.append(most_frequent)
        
    return valid_neighbors


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
        lin_A = a.get("linear", {})
        if lin_A and lin_A.get("plane") and len(lin_A["plane"]) > 0:
            na = normalize(np.asarray(lin_A["plane"][0][0], dtype=float))
        else:
            na = normalize(np.asarray(a["pca_normal"], dtype=float))
            
        for j, b in enumerate(B):
            cb = np.asarray(b["center"], dtype=float)
            lin_B = b.get("linear", {})
            if lin_B and lin_B.get("plane") and len(lin_B["plane"]) > 0:
                nb = normalize(np.asarray(lin_B["plane"][0][0], dtype=float))
            else:
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
    # 1) 파트쌍 단위로 그룹핑 (무순서 쌍)
    buckets: Dict[Tuple[int, int], Dict[str, List[Dict]]] = {}
    for it in results:
        # it["child"] might be a list now, so we need to handle that.
        # However, learn_separator_main usually returns single child per loop result.
        # If we modified identify_loop_neighbor to return list, we need to check where it is used.
        # It seems learn_separator_main calls identify_loop_neighbor.
        # We need to check learn_separator_main implementation.
        
        # Assuming results are flattened and duplicated if multiple children were found
        p = int(it["parent"])
        q = int(it["child"])

        if p == q:
            # 자기-루프는 건너뜀
            continue
        key = (p, q) if p < q else (q, p)
        dir_key = "A" if (p < q) else "B"   # A: p->q (작은->큰), B: q->p (큰->작은)
        if key not in buckets:
            buckets[key] = {"A": [], "B": []}
        buckets[key][dir_key].append(it)

    RLPs: List[Dict] = []

    # 2) 각 파트쌍마다 최소 비용 매칭
    for (i, j), grp in buckets.items():
        A = grp["A"]  # i->j
        B = grp["B"]  # j->i
        
        # [INFO] 각 방향별 루프 개수 출력
        print(f"[info] Pair ({i}, {j}): found {len(A)} loops (i->j) and {len(B)} loops (j->i).")
        
        if not A or not B:
            continue  # 한쪽 방향만 있으면 매칭 불가
        
        # If multiple neighbors were detected, we might have multiple entries for the same loop.
        # We should handle this. For now, we treat them as separate candidates.
        # However, if we have multiple candidates for the same loop, we might want to pick the best one.
        # But here we are matching between A and B sets.
        # If a loop in A has multiple potential neighbors in B, it will appear multiple times in A list with different 'child' values.
        # But 'child' is part of the key (i, j). So for a specific (i, j) pair, 
        # the list A contains loops from part i that think they are neighbors to part j.
        # So duplicates shouldn't be an issue here because we are iterating over specific (i, j) pairs.
        
        # However, if a loop was duplicated because it had multiple neighbors (e.g. j and k),
        # it will appear in bucket (i, j) and bucket (i, k).
        # This is fine, a loop can be part of multiple interfaces if it touches multiple parts.
        # But wait, a single loop usually defines a single boundary.
        # If it touches multiple parts, it might be better to split it?
        # But here we just allow it to be used in multiple RLPs.
        
        # But wait, if we use the same loop for multiple RLPs, we might generate multiple joints at the same location.
        # The user asked: "if multiple neighbors are detected, copy that many times so it can be used in cluster_reciprocal_loop_pairs".
        # This is exactly what we did in learn_separator_main by appending multiple results.
        # And since cluster_reciprocal_loop_pairs iterates over (i, j) buckets, 
        # each copy will go to its respective bucket.
        # So the logic seems correct.

        C = _build_cost_matrix(A, B, w_pos=w_pos, w_ang=w_ang)
        pairs = _hungarian_or_greedy(C)
        
        # [INFO] 매칭된 RLP 개수 출력
        print(f"[info] Pair ({i}, {j}): matched {len(pairs)} RLPs.")

        # 3) 페어 생성 (코스트 상한 필터)
        for ia, ib in pairs:
            cost = float(C[ia, ib])
            if (cost_max is not None) and (cost > cost_max):
                continue
            a = A[ia]; b = B[ib]
            ca = np.asarray(a["center"], dtype=float)
            cb = np.asarray(b["center"], dtype=float)
            
            lin_A = a.get("linear", {})
            if lin_A and lin_A.get("plane") and len(lin_A["plane"]) > 0:
                na = normalize(np.asarray(lin_A["plane"][0][0], dtype=float))
            else:
                na = normalize(np.asarray(a["pca_normal"], dtype=float))

            lin_B = b.get("linear", {})
            if lin_B and lin_B.get("plane") and len(lin_B["plane"]) > 0:
                nb = normalize(np.asarray(lin_B["plane"][0][0], dtype=float))
            else:
                nb = normalize(np.asarray(b["pca_normal"], dtype=float))
                
            # normal 방향 정합 (90도 이상 벌어지면 뒤집어서 일치시킴)
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

    # 4) 정렬(코스트 오름차순) 또는 길이 등 다른 기준도 가능
    RLPs.sort(key=lambda d: d["cost"])
    return RLPs
