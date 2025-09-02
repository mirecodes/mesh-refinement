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
from functions.graphs import cluster_reciprocal_loop_pairs

# --- small helpers to simplify branching & reuse ---

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


def stage_segment(cfgs, ms: pymeshlab.MeshSet):
    # TODO: Remove temporal attributes
    categories_num = 2
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
            try:
                a = vedo.Arrow(p0, p1).c(col).alpha(1.0)
            except TypeError:
                a = vedo.Line(p0, p1).c(col).alpha(1.0)
            vec_actors.append(a)

        show_actors = [mesh_actor, *part_actors, *vec_actors]

        vedo.settings.use_depth_peeling = True
        vedo.show(
            show_actors,
            f"RLP #{i}: parts {p_idx} ↔ {q_idx}",
            axes=1,
            interactive=True
        )
