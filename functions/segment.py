from functools import partial

import numpy as np
import pymeshlab
import trimesh
import vedo
from json_handler import JsonHandler

from functions.estimate import learn_separator_main

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
def _local_bbox_of_category(category_parts, pad_ratio=0.05):
    """Return (bb_min, bb_max) of a list of trimesh parts with small padding."""
    if not category_parts:
        return None, None
    pmins = [pm.vertices.min(axis=0) for pm in category_parts]
    pmaxs = [pm.vertices.max(axis=0) for pm in category_parts]
    bb_min = np.min(np.vstack(pmins), axis=0)
    bb_max = np.max(np.vstack(pmaxs), axis=0)
    size = bb_max - bb_min
    bb_min = bb_min - pad_ratio * size
    bb_max = bb_max + pad_ratio * size
    return bb_min, bb_max

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

    # svm training
    # TODO: remove index of part selection
    idx_selected_parts = [1, 2]
    svm_results = {}

    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    plane_size = float(np.linalg.norm(bb_max - bb_min))

    for idx_part in idx_selected_parts:
        try:
            out = learn_separator_main(
                ms,
                categories,
                idx_part,
                max_hops=10,
                method='polyhedral',
                C=1.0,
                gamma='scale',
                use_signed_dist=True,
                sdf_thresh=0.0,
                balance='None'
            )
            svm_results[idx_part] = out

            plane = out.get('plane', None)

            # context actors
            mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.25)
            part_actors = [vedo.Mesh([pm.vertices, pm.faces]).c("red").alpha(0.75) for pm in categories[idx_part]]

            # local bbox and plane sizing
            bb_min_local, bb_max_local = _local_bbox_of_category(categories[idx_part], pad_ratio=0.05)
            if bb_min_local is not None:
                local_center = 0.5 * (bb_min_local + bb_max_local)
                local_diag = float(np.linalg.norm(bb_max_local - bb_min_local))
                plane_size_local = max(local_diag, 1e-8) * 1.25
            else:
                bb_min_global = verts.min(axis=0)
                bb_max_global = verts.max(axis=0)
                local_center = 0.5 * (bb_min_global + bb_max_global)
                plane_size_local = float(np.linalg.norm(bb_max_global - bb_min_global)) * 1.25

            def _make_plane_actor(n_arr, d_scalar):
                n_arr = np.asarray(n_arr, dtype=float)
                d_scalar = float(d_scalar)
                center = _best_center_near_boundary(n_arr, d_scalar, verts, out, fallback_point=local_center)
                try:
                    return vedo.Plane(pos=center, normal=n_arr, s=plane_size_local).c("yellow").alpha(0.5)
                except TypeError:
                    actor = vedo.Plane(pos=center, normal=n_arr).c("yellow").alpha(0.5)
                    actor.scale([plane_size_local, plane_size_local, 1.0], origin=center)
                    return actor

            if plane is None:
                print(f"[info]: part #{idx_part}: non-linear SVM → rendering isosurface (f(x)=0)")
                # Optional: restrict bbox to selected category parts for speed
                if len(categories[idx_part]) > 0:
                    pmins = [pm.vertices.min(axis=0) for pm in categories[idx_part]]
                    pmaxs = [pm.vertices.max(axis=0) for pm in categories[idx_part]]
                    bb_min_local = np.min(np.vstack(pmins), axis=0)
                    bb_max_local = np.max(np.vstack(pmaxs), axis=0)
                    pad = 0.02 * np.linalg.norm(bb_max_local - bb_min_local)
                    bb_min_local -= pad
                    bb_max_local += pad
                    bbox = (bb_min_local, bb_max_local)
                else:
                    bbox = None
                show_svm_isosurface(out["clf"], out["scaler"], verts, faces,
                                    title=f"SVM surface for part #{idx_part}", grid=64, bbox=bbox)
            else:
                plane_actors = []
                # plane may be a single (n,d) or an iterable of (n,d)
                if (isinstance(plane, (list, tuple))
                    and len(plane) > 0
                    and isinstance(plane[0], (list, tuple, np.ndarray))):
                    for item in plane:
                        if len(item) != 2:
                            continue
                        n_i, d_i = item
                        plane_actors.append(_make_plane_actor(n_i, d_i))
                else:
                    n, d = plane
                    plane_actors.append(_make_plane_actor(n, d))

                vedo.show(
                    [mesh_actor, *part_actors, *plane_actors],
                    f"SVM plane(s) for part #{idx_part}",
                    axes=1,
                    interactive=True
                )

        except Exception as e:
            print(f"[error]: part #{idx_part}: {e}")