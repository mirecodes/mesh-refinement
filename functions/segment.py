import json
from functools import partial

import numpy as np
import pymeshlab
import trimesh
import vedo
from json_handler import JsonHandler
from collections import deque
import os

from functions.estimate import (learn_separator_main, )
from functions.graphs import (build_adjacency_graph, split_seed_boundary, classify_vertices, build_joints_hop1, )

# --- small helpers to simplify branching & reuse ---
def _is_multi_plane(plane):
    return isinstance(plane, (list, tuple)) and len(plane) > 0 and isinstance(plane[0], (list, tuple, np.ndarray))

def _bbox_for_parts(category_parts, pad_ratio=0.02):
    """(bb_min, bb_max) for parts with padding; None if empty."""
    if not category_parts:
        return None
    pmins = [pm.vertices.min(axis=0) for pm in category_parts]
    pmaxs = [pm.vertices.max(axis=0) for pm in category_parts]
    bb_min = np.min(np.vstack(pmins), axis=0)
    bb_max = np.max(np.vstack(pmaxs), axis=0)
    pad = pad_ratio * float(np.linalg.norm(bb_max - bb_min))
    return bb_min - pad, bb_max + pad

def _prepare_separator_and_center(plane, out, verts, local_center):
    """
    Returns:
      separator_for_boundary: (n,d) for plane or {'clf','scaler'} for SVM
      center_for_loop: plane center near boundary (None for non-linear SVM)
    """
    if not plane:
        return {"clf": out["clf"], "scaler": out["scaler"]}, None
    if _is_multi_plane(plane):
        n0, d0 = plane[0]
    else:
        n0, d0 = plane
    n0 = np.asarray(n0, float); d0 = float(d0)
    center = _best_center_near_boundary(n0, d0, verts, out, fallback_point=local_center)
    return (n0, d0), center

def make_plane_actor(n_arr, d_scalar, verts, out, fallback_center, plane_size):
    """Create a vedo.Plane actor near the boundary with fixed size."""
    n_arr = np.asarray(n_arr, dtype=float)
    d_scalar = float(d_scalar)
    center = _best_center_near_boundary(n_arr, d_scalar, verts, out, fallback_point=fallback_center)
    try:
        return vedo.Plane(pos=center, normal=n_arr, s=float(plane_size)).c("yellow").alpha(0.5)
    except TypeError:  # older vedo
        actor = vedo.Plane(pos=center, normal=n_arr).c("yellow").alpha(0.5)
        actor.scale([float(plane_size), float(plane_size), 1.0], origin=center)
        return actor

def _make_plane_actors(plane, make_actor):
    """Create vedo plane actors from a single (n,d) or list of (n,d)."""
    if plane is None:
        return []
    items = plane if _is_multi_plane(plane) else [plane]
    actors = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            n_i, d_i = item
            actors.append(make_actor(n_i, d_i))
    return actors

# --- optional: non-linear SVM isosurface visualization ---
try:
    from skimage.measure import marching_cubes
except Exception as _e:
    marching_cubes = None

def show_svm_isosurface(clf, scaler, verts, faces, title="SVM surface", grid=64, bbox=None):
    """
    Visualize f(x)=0 of an SVM (typically RBF) as an isosurface using marching cubes.
    - clf: trained sklearn SVM with decision_function
    - scaler: StandardScaler used during training
    - verts, faces: original mesh for context (vedo Mesh)
    - grid: resolution per axis (64~128 recommended)
    - bbox: optional (min, max) tuple to restrict volume; if None uses verts bbox
    """
    if marching_cubes is None:
        print("[warn] scikit-image not available; cannot render isosurface. pip install scikit-image")
        return

    import numpy as np
    import vedo

    # 1) bounding box
    if bbox is None:
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
    else:
        bb_min, bb_max = bbox
    xs = np.linspace(bb_min[0], bb_max[0], grid)
    ys = np.linspace(bb_min[1], bb_max[1], grid)
    zs = np.linspace(bb_min[2], bb_max[2], grid)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")

    # 2) decision function on grid (batched)
    coords = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    coords_s = scaler.transform(coords)
    F = np.empty(coords_s.shape[0], dtype=np.float32)
    bs = 200000
    for s in range(0, len(coords_s), bs):
        F[s:s+bs] = clf.decision_function(coords_s[s:s+bs])
    F = F.reshape(grid, grid, grid)

    # 3) extract isosurface f(x)=0
    dx, dy, dz = xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]
    v, f, n, val = marching_cubes(F, level=0.0, spacing=(dx, dy, dz))
    # shift to world coords
    v[:, 0] += bb_min[0]
    v[:, 1] += bb_min[1]
    v[:, 2] += bb_min[2]

    # 4) render
    mesh_actor  = vedo.Mesh([verts, faces]).c("white").alpha(0.25)
    surf_actor  = vedo.Mesh([v, f]).c("yellow").alpha(0.6)
    vedo.show([mesh_actor, surf_actor], title, axes=1, interactive=True)

# --- helpers to size/position planes near the boundary ---

def _project_point_to_plane(p, n, d):
    """Orthogonally project point p onto plane n·x + d = 0."""
    s = float(np.dot(n, p) + d)
    return p - s * n

def _best_center_near_boundary(n, d, verts, out, fallback_point):
    """Choose a plane center near the learned boundary using neighbor verts.
    - If idx_neighbor exists, pick the vertex with minimal |n·x + d| and project it.
    - Otherwise, project the fallback point (typically local bbox center).
    """
    idx_neighbor = out.get("idx_neighbor", None)
    if idx_neighbor is not None and len(idx_neighbor) > 0:
        X = verts[np.asarray(idx_neighbor, dtype=int)]
        s = X @ n + float(d)
        k = int(np.argmin(np.abs(s)))
        return X[k] - s[k] * n
    return _project_point_to_plane(fallback_point, n, d)

def _make_boundary_lines(verts, faces, separator, idx_part, plane_center=None, color="cyan", lw=3):
    """
    Return only ONE boundary loop for idx_part.
    Seed rule: boundary vertex (label==idx_part) closest to plane_center.
    If plane_center is None, fallback to the largest boundary component.
    """
    # labels via split_seed_boundary (works with plane or SVM separator)
    labels, _, _ = split_seed_boundary(verts, faces, separator, idx_part=idx_part)
    labels = labels.astype(int)

    # collect boundary edges where exactly one endpoint is idx_part
    faces_i = faces.astype(int)
    edge_set = set()
    boundary_vertices = set()
    for a, b, c in faces_i:
        for u, v in ((a, b), (b, c), (c, a)):
            if labels[u] != labels[v] and (labels[u] == idx_part or labels[v] == idx_part):
                e = (u, v) if u < v else (v, u)
                if e not in edge_set:
                    edge_set.add(e)
                    boundary_vertices.add(u); boundary_vertices.add(v)
    if not edge_set:
        return None

    # adjacency graph over boundary vertices
    adj = {int(i): [] for i in boundary_vertices}
    for u, v in edge_set:
        adj[u].append(v); adj[v].append(u)

    # connected components of boundary graph
    visited = set()
    components = []
    for s in list(boundary_vertices):
        if s in visited:
            continue
        stack = [s]; comp = set()
        visited.add(s)
        while stack:
            x = stack.pop()
            comp.add(x)
            for y in adj.get(x, []):
                if y not in visited:
                    visited.add(y)
                    stack.append(y)
        components.append(comp)

    # choose component
    if plane_center is not None:
        # pick seed: closest boundary vertex that belongs to idx_part
        cand = np.array([i for i in boundary_vertices if labels[i] == idx_part], dtype=int)
        if cand.size == 0:
            return None
        d2 = np.sum((verts[cand] - plane_center[None, :])**2, axis=1)
        seed = int(cand[int(np.argmin(d2))])
        # find the component containing the seed
        chosen = next((comp for comp in components if seed in comp), None)
        if chosen is None:
            return None
    else:
        # fallback: choose the largest component
        chosen = max(components, key=lambda c: len(c))

    # build segments only for the chosen component
    segs = [[verts[u], verts[v]] for (u, v) in edge_set if (u in chosen and v in chosen)]
    if len(segs) == 0:
        return None
    return vedo.Lines(segs).c(color).lw(lw)

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
    faces = current_mesh.face_matrix()

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
    cb_key = partial(
        on_tab,
        vmeshes=vmeshes, belongings=belongings, prevs=prevs,
        shared=shared, rand_colors=rand_colors, categories_num=categories_num, plt=plt,
    )

    plt.add_callback("LeftButtonPress", cb_left)
    plt.add_callback("RightButtonPress", cb_right)
    plt.add_callback("KeyPress",         cb_key)

    recolor_parts(vmeshes, belongings, 0, rand_colors)
    plt.show(vmeshes, interactive=True)

    # after the selection has ended
    for i in range(1, len(belongings)):
        categories[belongings[i]].append(parts[i-1])

    # TODO: remove index of part selection
    idx_selected_parts = [1, 2]
    svm_results = {}
    calc_results = {}  # store per-part computed artifacts for later visualization

    # -------- Phase 1: CALCULATION (no visualization here) --------
    for idx_part in idx_selected_parts:
        out = learn_separator_main(
            ms,
            categories,
            idx_part,
            max_hops=10,
            method='linear',
            C=1.0,
            gamma='scale',
            use_signed_dist=True,
            sdf_thresh=0.0,
            balance='None'
        )
        svm_results[idx_part] = out

        plane = out.get('plane', None)
        # local bbox and plane sizing
        bbox_local = _bbox_for_parts(categories[idx_part], pad_ratio=0.05)
        if bbox_local is not None:
            bb_min_local, bb_max_local = bbox_local
        else:
            bb_min_local, bb_max_local = verts.min(axis=0), verts.max(axis=0)
        local_center = 0.5 * (bb_min_local + bb_max_local)
        local_diag = float(np.linalg.norm(bb_max_local - bb_min_local))
        plane_size_local = max(local_diag, 1e-8) * 1.25

        separator_for_boundary, center_for_loop = _prepare_separator_and_center(plane, out, verts, local_center)

        if out.get('method', None) == 'linear':
            # Use the first plane (n0, d0). Visualize a normal starting at local_center along +n0.
            if _is_multi_plane(plane):
                n0, d0 = plane[0]
            else:
                n0, d0 = plane
            n0 = np.asarray(n0, dtype=float)
            n0 /= (np.linalg.norm(n0) + 1e-12)
            # Start at local_center and go in +n0 direction
            p0 = np.asarray(local_center, dtype=float)
            seg_len = float(plane_size_local) * 0.5
            p1 = p0 + seg_len * n0
            normal_info = {"p0": p0, "p1": p1}
        else:
            normal_info = None

        # persist everything needed for visualization
        calc_results[idx_part] = dict(
            out=out,
            plane=plane,
            local_center=local_center,
            plane_size=plane_size_local,
            separator=separator_for_boundary,
            center_for_loop=center_for_loop,
            normal=normal_info,
        )

    # -------- Phase 1.5: BUILD JOINTS (before visualization) --------
    try:
        # Global labeling once (uses same settings as separator; adjust if needed)
        vert_label = classify_vertices(
            verts,
            categories,
            use_signed_dist=True,
            sdf_thresh=0.0,
        )
        joint_index = build_joints_hop1(verts, faces, vert_label)

        # Save joints.json alongside states
        # out_path = getattr(cfgs, 'json_states_dir')
        # with open(out_path, 'w') as f:
        #     json.dump(joint_index, f, indent=2)
        # print(f"[info]: joints saved to {out_path}; #joints={len(joint_index['joints'])}")
        # for pid, lst in joint_index['part_to_joints'].items():
        #     if not lst:
        #         continue
        #     labels = ', '.join([f"{it['label']}→P{it['other']}({it['joint_id']})" if it['label'] else f"→P{it['other']}({it['joint_id']})" for it in lst])
        #     print(f"[info]: part {pid}: {labels}")
    except Exception as e:
        print(f"[warn]: joint build failed before visualization: {e}")



    # -------- Phase 2: VISUALIZATION (consumes cached results) --------
    for idx_part in idx_selected_parts:
        res = calc_results[idx_part]
        out = res["out"]
        plane = res["plane"]
        local_center = res["local_center"]
        plane_size_local = res["plane_size"]
        separator_for_boundary = res["separator"]
        center_for_loop = res["center_for_loop"]
        normal_info = res.get("normal", None)

        # context actors
        mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.25)
        part_actors = [vedo.Mesh([pm.vertices, pm.faces]).c("red").alpha(0.75) for pm in categories[idx_part]]

        if not plane:
            print(f"[info]: part #{idx_part}: non-linear SVM → rendering isosurface (f(x)=0)")
            bbox = _bbox_for_parts(categories[idx_part], pad_ratio=0.02)
            boundary_actor = _make_boundary_lines(verts, faces, separator_for_boundary, idx_part, color="cyan", lw=3)
            if boundary_actor is not None:
                vedo.show([mesh_actor, boundary_actor], f"Boundary edges for part #{idx_part}", axes=1, interactive=False)
            show_svm_isosurface(out["clf"], out["scaler"], verts, faces,
                                title=f"SVM surface for part #{idx_part}", grid=64, bbox=bbox)
        else:
            plane_actor_fn = partial(
                make_plane_actor,
                verts=verts,
                out=out,
                fallback_center=local_center,
                plane_size=plane_size_local,
            )
            plane_actors = _make_plane_actors(plane, plane_actor_fn)

            # boundary edge overlay using chosen separator
            boundary_actor = _make_boundary_lines(verts, faces, separator_for_boundary, idx_part, plane_center=center_for_loop, color="cyan", lw=3)
            extra_actors = [boundary_actor] if boundary_actor is not None else []

            # Visualize the normal (arrow) from local_center along +n0
            if normal_info is not None:
                try:
                    normal_actor = vedo.Arrow(normal_info["p0"], normal_info["p1"]).c("orange").lw(2).alpha(1.0)
                except TypeError:
                    normal_actor = vedo.Line(normal_info["p0"], normal_info["p1"]).c("orange").lw(3).alpha(1.0)
                extra_actors.append(normal_actor)

            vedo.show(
                [mesh_actor, *part_actors, *plane_actors, *extra_actors],
                f"SVM plane(s) for part #{idx_part}",
                axes=1,
                interactive=True
            )