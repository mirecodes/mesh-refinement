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
            if plane is not None:
                n, d = plane
                center = -d * n
                print(n, d)

                mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.25)
                part_actors = list()
                for part in categories[idx_part]:
                    part_actors.append(vedo.Mesh([part.vertices, part.faces]).c("red").alpha(0.75))
                # part_actor = vedo.Mesh([parts[idx_part].vertices, parts[idx_part].faces]).c("red").alpha(0.8)
                # Fallback for older/newer API variants: create unit plane then scale it about center
                plane_actor = vedo.Plane(pos=center, normal=n).c("yellow").alpha(0.5)
                plane_actor.scale([plane_size, plane_size, 1.0], origin=center)

                vedo.show(
                    [mesh_actor, part_actors, plane_actor],
                    f"SVM plane for part #{idx_part}",
                    axes=1,
                    interactive=True
                )
            else:
                print(f"[info]: part #{idx_part}: non-linear SVM → rendering isosurface (f(x)=0)")
                # Optional: restrict bbox to selected category parts for speed
                if len(categories[idx_part]) > 0:
                    pmins = []
                    pmaxs = []
                    for pm in categories[idx_part]:
                        pmins.append(pm.vertices.min(axis=0))
                        pmaxs.append(pm.vertices.max(axis=0))
                    bb_min_local = np.min(np.vstack(pmins), axis=0)
                    bb_max_local = np.max(np.vstack(pmaxs), axis=0)
                    # small padding
                    pad = 0.02 * np.linalg.norm(bb_max_local - bb_min_local)
                    bb_min_local -= pad
                    bb_max_local += pad
                    bbox = (bb_min_local, bb_max_local)
                else:
                    bbox = None
                show_svm_isosurface(out["clf"], out["scaler"], verts, faces,
                                    title=f"SVM surface for part #{idx_part}", grid=64, bbox=bbox)

        except Exception as e:
            print(f"[error]: part #{idx_part}: {e}")