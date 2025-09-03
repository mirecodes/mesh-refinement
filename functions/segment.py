import json
from functools import partial

import numpy as np
import pymeshlab
import trimesh
import vedo
from json_handler import JsonHandler
from collections import deque
import os

from functions.estimate import learn_separator_main
from functions.graphs import cluster_reciprocal_loop_pairs, classify_vertices

# --- small helpers to simplify branching & reuse ---

import numpy as np

def build_links_from_labels(vert_label: np.ndarray, categories_present: list[int]) -> list[dict]:
    """
    vert_label[v] == k  -> 정점 v가 카테고리 k에 속함.
    categories_present 에 포함된 category만 links로 생성.
    반환: [{'id': k, 'vertices': [v0, v1, ...]}, ...]
    """
    links = []
    for k in sorted(set(categories_present)):
        if k <= 0:
            continue
        vidx = np.where(vert_label == k)[0].astype(int).tolist()
        links.append({"id": int(k), "vertices": vidx})
    return links


def build_links_from_mapping(cat_to_vert_indices: dict[int, np.ndarray | list[int]]) -> list[dict]:
    """
    vert_label이 없고, 카테고리 -> 정점인덱스 매핑을 이미 갖고 있을 때 사용.
    """
    links = []
    for k in sorted(cat_to_vert_indices.keys()):
        if k <= 0:
            continue
        vidx = np.asarray(cat_to_vert_indices[k], dtype=int).ravel().tolist()
        links.append({"id": int(k), "vertices": vidx})
    return links


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n


def build_joints_from_rlps(rlps: list[dict]) -> list[dict]:
    """
    rlps 항목 구조 가정:
      - rlp['a']['parent'] : int  (link id)
      - rlp['a']['child']  : int  (link id)
      - rlp['center']      : (3,) center (origin)
      - rlp['vectors']     : list of dicts
            each vec: {
                'center': (3,),
                'n': (3,),           # 또는 'n_poly'
                'state': 'None' | 'Linear' | 'Revolute',
                'order': int | None  # 첫 선택 순서
            }

    선택 규칙:
      - state가 'Linear' 또는 'Revolute' 인 벡터만 사용
      - 여러 개면 order 오름차순으로 모두 기록(필요시 하나만 쓰려면 [:1])
    반환 예:
      [{'parent': id_p, 'child': id_c,
        'axis': {'origin': [x,y,z], 'n': [nx,ny,nz]},
        'type': 'Linear' or 'Revolute'}, ...]
    """
    joints = []
    for rlp in rlps:
        p = int(rlp["a"]["parent"])
        c = int(rlp["a"]["child"])

        vecs = rlp.get("vectors", []) or []
        # state가 선택된 것만, order 기준 정렬
        chosen = [
            v for v in vecs
            if isinstance(v, dict) and v.get("state") in ("Linear", "Revolute")
        ]
        # order가 None이면 큰 값으로 보내고, 숫자면 숫자 정렬
        def _ord_key(v):
            o = v.get("order", None)
            return (1, 1e18) if o is None else (0, int(o))
        chosen.sort(key=_ord_key)

        # 선택된 벡터 각각을 하나의 joint로 기록 (여러 축을 허용)
        for v in chosen:
            ori = np.asarray(v.get("center", rlp.get("center", [0,0,0])), dtype=float).ravel()
            nvec = v.get("n")
            if nvec is None:
                nvec = v.get("n_poly")  # poly 교선 방향이 있으면 사용
            nvec = _normalize(np.asarray(nvec, dtype=float).ravel())
            if not np.isfinite(nvec).all() or np.linalg.norm(nvec) < 1e-12:
                continue
            joints.append({
                "parent": p,
                "child":  c,
                "axis": {
                    "origin": ori.astype(float).tolist(),
                    "n":      nvec.astype(float).tolist(),
                },
                "type":   v.get("state")  # 'Linear' or 'Revolute'
            })
    return joints


def recolor_parts(vmeshes, belongings, mode, rand_colors):
    if mode == 0:
        for i, vmesh in enumerate(vmeshes):
            vmesh.c(rand_colors[i]).alpha(0.5)
    else:
        for i, vmesh in enumerate(vmeshes):
            if belongings[i] == mode:
                vmesh.c("red").alpha(0.8)
            else:
                vmesh.c("lightblue").alpha(0.4)


def on_left_click(event, vmeshes, belongings, prevs, shared, rand_colors, plt):
    vmesh = event.actor
    mode = shared['mode']
    if vmesh is None or mode == 0:
        return
    if vmesh in vmeshes[1:]:
        idx = vmesh.idx_part
        prevs[idx] = belongings[idx]
        belongings[idx] = mode
        recolor_parts(vmeshes, belongings, mode, rand_colors)
        plt.render()

def on_right_click(event, vmeshes, belongings, prevs, shared, rand_colors, plt):
    vmesh = event.actor
    mode = shared['mode']
    if vmesh is None or mode == 0:
        return
    if vmesh in vmeshes[1:]:
        idx = vmesh.idx_part
        belongings[idx] = prevs[idx]
        recolor_parts(vmeshes, belongings, mode, rand_colors)
        plt.render()


def on_tab(event, vmeshes, belongings, prevs, shared, categories_num, rand_colors, plt):
    if event.keypress in ('\t', 'Tab'):
        shared['mode'] = (shared['mode']+1) % (categories_num+1)
        mode = shared['mode']
        recolor_parts(vmeshes, belongings, mode, rand_colors)
        plt.render()
        if mode == 0:
            print(f"[info]: defualt mode")
        else:
            print(f"[info]: selection mode #{mode}")

def on_key(event, vmeshes, belongings, prevs, shared, rand_colors, categories_num, plt):
    """KeyPress dispatcher: ESC/q to exit; Tab cycles selection."""
    k = getattr(event, "keypress", None)
    if k in ("q", "Q", "Esc", "Escape", "\x1b"):
        try:
            plt.close()
        except Exception:
            pass
        try:
            vedo.close()  # ensure all windows close
        except Exception:
            pass
        return
    # keep your Tab behavior
    if k in ('\t', 'Tab'):
        on_tab(event, vmeshes, belongings, prevs, shared, categories_num, rand_colors, plt)

# --- vector interaction handler (reusable) ---
def on_vector_click(event, *, vec_actors, rlp, click_counter):
    """
    Cycle a clicked vector actor through states:
    'None'(gray) -> 'Linear'(yellow) -> 'Revolute'(blue) -> 'None'.
    Records first-selection order into rlp['vectors'][k]['order'].
    """
    act = getattr(event, "actor", None)
    if act is None or act not in vec_actors:
        return
    st = getattr(act, "vec_state", "None")
    idx = getattr(act, "vec_idx", None)
    if idx is None:
        return

    if st == "None":
        # None -> Linear (yellow), assign order
        try:
            act.c("yellow").alpha(1.0)
        except Exception:
            pass
        setattr(act, "vec_state", "Linear")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            click_counter['count'] = int(click_counter.get('count', 0)) + 1
            rlp['vectors'][idx]['state'] = "Linear"
            rlp['vectors'][idx]['order'] = int(click_counter['count'])
    elif st == "Linear":
        # Linear -> Revolute (blue), keep order
        try:
            act.c("blue").alpha(1.0)
        except Exception:
            pass
        setattr(act, "vec_state", "Revolute")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "Revolute"
    else:
        # Revolute -> None (gray), clear order
        try:
            act.c("gray").alpha(0.5)
        except Exception:
            pass
        setattr(act, "vec_state", "None")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "None"
            rlp['vectors'][idx]['order'] = None

    # re-render
    try:
        event.plotter.render()
    except Exception:
        pass


def stage_segment(cfgs, ms: pymeshlab.MeshSet):
    # TODO: Remove temporal attributes
    categories_num = 3
    categories = [[] for _ in range(categories_num+1)]

    # Read json states
    states = JsonHandler(cfgs.json_states_dir)

    # Load the parts
    parts = []
    for i in range(states.decompose.length):
        tri = trimesh.load(states.decompose.dirs.mesh[i])
        parts.append(tri)

    # Prepare for the visualization
    vmeshes = []
    rand_colors = []

    # Load meshes from pymeshlab
    ms.set_current_mesh(0)
    current_mesh = ms.current_mesh()
    verts = current_mesh.vertex_matrix()
    faces = current_mesh.face_matrix().astype(np.int64)

    vmesh = vedo.Mesh([verts, faces])
    vmeshes.append(vmesh)
    rand_colors.append((255, 255, 255))

    for i, part in enumerate(parts, start=1):
        tri = part.copy()
        vmesh = vedo.Mesh([tri.vertices, tri.faces])
        rand_color = np.random.rand(3)
        vmesh.c(rand_color).alpha(0.5)
        vmesh.idx_part = i
        vmeshes.append(vmesh)
        rand_colors.append(rand_color)

    # parts information
    belongings = [1 for _ in range(len(vmeshes))]; belongings[0] = 0
    prevs = [1 for _ in range(len(vmeshes))]; prevs[0] = 0
    shared = {"mode": 0}

    # Compose GUI
    plt = vedo.Plotter(title="Part selection")

    # --- Disable vedo's built-in keyboard shortcuts (KeyPress/Release/Char) ---
    try:
        iren = plt.interactor  # vtkRenderWindowInteractor
        for _ev in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
            try:
                iren.RemoveObservers(_ev)
            except Exception:
                pass
    except Exception as _e:
        print("[warn] Could not remove default keyboard observers:", _e)

    # partial 로 인자 고정
    cb_left = partial(
        on_left_click,
        vmeshes=vmeshes, belongings=belongings, prevs=prevs,
        shared=shared, rand_colors=rand_colors, plt=plt,
    )
    cb_right = partial(
        on_right_click,
        vmeshes=vmeshes, belongings=belongings, prevs=prevs,
        shared=shared, rand_colors=rand_colors, plt=plt,
    )
    cb_key_dispatch = partial(
        on_key,
        vmeshes=vmeshes, belongings=belongings, prevs=prevs,
        shared=shared, rand_colors=rand_colors, categories_num=categories_num, plt=plt,
    )

    plt.add_callback("LeftButtonPress", cb_left)
    plt.add_callback("RightButtonPress", cb_right)
    plt.add_callback("KeyPress", cb_key_dispatch)

    recolor_parts(vmeshes, belongings, 0, rand_colors)
    plt.show(vmeshes, interactive=True)

    # after the selection has ended
    for i in range(1, len(belongings)):
        categories[belongings[i]].append(parts[i-1])

    # TODO: remove index of part selection
    method = 'all' # fixed to 'linear' so far
    results = {}

    # -------- Phase 1: GET SEPARATION PLANES (no visualization here) --------
    for idx_part in range(1, categories_num+1):
        out = learn_separator_main(
            ms,
            categories,
            idx_part,
            max_hops=10,
            method=method,
            C=1.0,
            gamma='scale',
            use_signed_dist=True,
            sdf_thresh=0.0,
            balance='None'
        )
        results[idx_part] = out

    # print(results)
    results_list = [item for v in results.values() for item in (v if isinstance(v, list) else [v])]

    rlps = cluster_reciprocal_loop_pairs(results_list, w_pos=1.0, w_ang=0.5, cost_max=None)

    # Calculate Vectors
    for i, rlp in enumerate(rlps):
        # Extract centers
        center = rlp['center']

        # Extract normals
        A = rlp['a']['linear']; B = rlp['b']['linear']
        if A.get('plane') and len(A['plane']) > 0 and B.get('plane') and len(B['plane']) > 0:
            nA = np.asarray(A['plane'][0][0], dtype=float)
            nB = np.asarray(B['plane'][0][0], dtype=float)
            dA = float(A['plane'][0][1]);
            dB = float(B['plane'][0][1])
        else:
            nA = np.asarray(A['pca_normal'], dtype=float)
            nB = np.asarray(B['pca_normal'], dtype=float)
            dA = dB = 0.0
        # Align directions before averaging
        if float(np.dot(nA, nB)) < 0.0:
            nB = -nB
        n = np.mean(np.vstack([nA, nB]), axis=0)
        n = n / (np.linalg.norm(n) + 1e-12)
        d = 0.5 * (dA + dB)
        rlps[i]['vectors'] = []
        rlps[i]['vectors'].append({'center': center, 'n': n, 'd': d})

        # Extract intersections
        A = rlp['a']['polyhedral']; B = rlp['b']['polyhedral']
        if A.get('plane') and len(A['plane']) > 0:
            planes = A.get('plane')
            planes_num = len(planes)
            for u in range(planes_num):
                for v in range(u+1, planes_num):
                    n = np.cross(planes[u][0], planes[v][0])
                    rlps[i]['vectors'].append({'center': center, 'n': n, 'd': np.zeros(3)})

        # Align directions before averaging



    # -------- Phase 2: VISUALIZATION (per RLP) --------
    # For each reciprocal loop pair: draw base mesh, two connected parts with different colors, and vectors[i]
    for i, rlp in enumerate(rlps):
        # base mesh
        mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.35)
        try:
            mesh_actor.pickable(False)
        except Exception:
            pass

        # two parts (parent/child) in distinct colors
        p_idx = int(rlp["a"]["parent"])
        q_idx = int(rlp["a"]["child"])
        # safety: ensure indices are valid and distinct
        part_ids = [pid for pid in (p_idx, q_idx) if pid != 0 and 1 <= pid <= len(parts)]
        part_ids = list(dict.fromkeys(part_ids))  # unique, keep order

        part_colors = [(0.85, 0.2, 0.2), (0.2, 0.4, 0.85)]
        part_actors = []
        for j, pid in enumerate(part_ids):
            for tri in categories[pid]:
                actor = vedo.Mesh([tri.vertices, tri.faces]).c(part_colors[min(j, 1)]).alpha(0.25)
                try:
                    actor.pickable(False)
                except Exception:
                    pass
                part_actors.append(actor)

        # corresponding vectors (center + direction) for this RLP, visualize all stored vectors
        vec_actors = []
        vecs = rlp.get('vectors', [])
        if isinstance(vecs, dict):
            vecs = [vecs]
        # scale once
        seg_len = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) * 0.2
        # color palette for multiple vectors
        colors = ["orange", "green", "magenta", "cyan", "yellow", "purple"]
        for k, v in enumerate(vecs):
            if v is None:
                continue
            c = np.asarray(v.get("center", rlp.get("center", [0,0,0])), dtype=float)
            n = np.asarray(v.get("n"), dtype=float)
            if n.ndim == 0 or not np.isfinite(n).all():
                continue
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            p0 = c
            p1 = c + seg_len * n
            col = colors[k % len(colors)]
            # build a thicker, pickable vector actor
            a = None
            # choose a thickness proportional to scene size
            r_shaft = seg_len * 0.02

            try:
                # Arrow with explicit radii improves picking
                a = vedo.Arrow(p0, p1, shaftRadius=r_shaft).c(col).alpha(1.0)
            except TypeError:
                try:
                    # Fallback: real tube geometry (better than thin line for picking)
                    a = vedo.Tube([p0, p1], r=r_shaft, cap=True).c(col).alpha(1.0)
                except Exception:
                    # Last resort: a thicker line
                    a = vedo.Line(p0, p1).c(col).alpha(1.0).lw(8)
            try:
                a.pickable(True)
            except Exception:
                pass
            vec_actors.append(a)

        # initialize vector states: all gray ('None')
        for k, a in enumerate(vec_actors):
            try:
                a.c("gray").alpha(1.0)
            except Exception:
                pass
            # attach state to actor and mirror to rlp['vectors'][k]
            setattr(a, "vec_state", "None")
            setattr(a, "vec_idx", k)
            if 'vectors' in rlp and k < len(rlp['vectors']):
                rlp['vectors'][k]['state'] = "None"
                rlp['vectors'][k]['order'] = None

        show_actors = [mesh_actor, *part_actors, *vec_actors]

        # per-RLP click order counter
        click_counter = {'count': 0}

        vedo.settings.use_depth_peeling = True
        plt_rlp = vedo.Plotter(title=f"RLP #{i}: parts {p_idx} ↔ {q_idx}", axes=1)

        # bind arguments via partial for a clean, uniform callback signature
        cb_vec_click = partial(on_vector_click, vec_actors=vec_actors, rlp=rlp, click_counter=click_counter)
        plt_rlp.add_callback("LeftButtonPress", cb_vec_click)

        plt_rlp.show(show_actors, interactive=True).close()

    # Save intermediate state into the json file
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)

    links = [[] for _ in range(categories_num+1)]
    vert_label = classify_vertices(verts, categories, use_signed_dist=True)
    for index, cat in enumerate(vert_label):
        links[cat].append(index)
    joints = build_joints_from_rlps(rlps)

    states.segment = {}
    states.segment.links = links
    states.segment.joints = joints
