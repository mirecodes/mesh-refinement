import numpy as np
import vedo
import trimesh
from typing import List, Dict, Any
from functools import partial
import pymeshlab

from functions.lib.geometry import normalize, rotmat

# =============================================================================
# Helper: Convert Trimesh list to Vedo
# =============================================================================
def trimesh_list_to_vedo_mesh(parts: List[trimesh.Trimesh]) -> vedo.Mesh | None:
    """Convert list of trimesh.Trimesh into a single vedo.Mesh (or None if empty)."""
    if not parts:
        return None
    v_all = []
    f_all = []
    off = 0
    for tm in parts:
        v = np.asarray(tm.vertices, dtype=float)
        f = np.asarray(tm.faces, dtype=int)
        v_all.append(v)
        f_all.append(f + off)
        off += v.shape[0]
    V = np.vstack(v_all)
    F = np.vstack(f_all)
    return vedo.Mesh([V, F])

# =============================================================================
# Visualization — part selection callbacks
# =============================================================================
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
        shared['mode'] = (shared['mode'] + 1) % (categories_num + 1)
        mode = shared['mode']
        recolor_parts(vmeshes, belongings, mode, rand_colors)
        plt.render()
        print(f"[info]: {'default mode' if mode==0 else f'selection mode #{mode}'}")


def on_key(event, vmeshes, belongings, prevs, shared, rand_colors, categories_num, plt):
    """ESC/q to quit; Tab to cycle selection mode."""
    k = getattr(event, "keypress", None)
    if k in ("q", "Q", "Esc", "Escape", "\x1b"):
        plt.close()
        vedo.close()
        return
    if k in ('\t', 'Tab'):
        on_tab(event, vmeshes, belongings, prevs, shared, categories_num, rand_colors, plt)

def select_parts_interactive(parts: List[trimesh.Trimesh], verts: np.ndarray, faces: np.ndarray, categories_num: int, display: bool = True) -> tuple[List[List[trimesh.Trimesh]], Any, Any]:
    """Interactive part selection UI. Returns (categories, belongings, rand_colors)."""
    if not display:
        # Default behavior if no display: assign based on something or just return empty?
        # Assuming if no display, we shouldn't fail but interactive selection is impossible.
        # This function is inherently interactive. If display=False, maybe skip?
        print("[warn]: select_parts_interactive called with display=False. Skipping UI.")
        # Fallback or error? For now return empty or simple grouping?
        # But this returns 'categories'. We can't proceed without categories.
        # If this is called, user probably expects interaction.
        # But for automation, we might need a way to bypass.
        # For now, just return empty categories and let the caller handle it.
        return [[] for _ in range(categories_num + 1)], [], []

    vmeshes, rand_colors = [], []
    base_actor = vedo.Mesh([verts, faces])
    vmeshes.append(base_actor); rand_colors.append((255, 255, 255))
    for i, part in enumerate(parts, start=1):
        actor = vedo.Mesh([part.vertices, part.faces])
        color = np.random.rand(3); actor.c(color).alpha(0.5)
        actor.idx_part = i
        vmeshes.append(actor); rand_colors.append(color)

    belongings = [1 for _ in range(len(vmeshes))]; belongings[0] = 0
    prevs = [1 for _ in range(len(vmeshes))]; prevs[0] = 0
    shared = {"mode": 0}

    plt = vedo.Plotter(title="Part selection")
    plt.add_callback("LeftButtonPress", partial(on_left_click,  vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, plt=plt))
    plt.add_callback("RightButtonPress", partial(on_right_click, vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, plt=plt))
    plt.add_callback("KeyPress",        partial(on_key,        vmeshes=vmeshes, belongings=belongings, prevs=prevs, shared=shared, rand_colors=rand_colors, categories_num=categories_num, plt=plt))
    recolor_parts(vmeshes, belongings, 0, rand_colors)
    import time
    import functions
    gui_start = time.time()
    plt.show(vmeshes, interactive=True)
    functions.total_gui_time += time.time() - gui_start
    
    categories = [[] for _ in range(categories_num + 1)]
    for i in range(1, len(belongings)):
        categories[belongings[i]].append(parts[i - 1])
        
    return categories, belongings, rand_colors


# =============================================================================
# Vector selection (per-RLP)
# =============================================================================
def on_vector_click(event, *, vec_actors, rlp, click_counter):
    """Toggle None → Revolute → Prismatic → None; record first selection order."""
    act = getattr(event, "actor", None)
    if act is None or act not in vec_actors:
        return
    st = getattr(act, "vec_state", "None")
    idx = getattr(act, "vec_idx", None)
    if idx is None:
        return

    if st == "None":
        act.c("yellow").alpha(1.0)
        setattr(act, "vec_state", "Revolute")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            click_counter['count'] = int(click_counter.get('count', 0)) + 1
            rlp['vectors'][idx]['state'] = "Revolute"
            rlp['vectors'][idx]['order'] = int(click_counter['count'])
    elif st == "Revolute":
        act.c("blue").alpha(1.0)
        setattr(act, "vec_state", "Prismatic")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "Prismatic"
    else:
        act.c("gray").alpha(0.9)
        setattr(act, "vec_state", "None")
        if 'vectors' in rlp and idx < len(rlp['vectors']):
            rlp['vectors'][idx]['state'] = "None"
            rlp['vectors'][idx]['order'] = None
    event.plotter.render() if hasattr(event, "plotter") else None


def rebuild_tube_actor(plt,
                        *,
                        old_actor,
                        center: np.ndarray,
                        n: np.ndarray,
                        seg_len: float,
                        radius: float,
                        state: str,
                        vi: int,
                        vec_actors: list,
                        idx2actor: dict):
    """Recreate Tube, replace in scene, keep state & index."""
    p0 = np.asarray(center, dtype=float)
    p1 = p0 + seg_len * normalize(np.asarray(n, dtype=float))
    new_actor = vedo.Tube([p0, p1], r=radius, cap=True).lighting('off')
    new_actor.c("yellow" if state == "Revolute" else "blue" if state == "Prismatic" else "gray")
    new_actor.alpha(1.0 if state in ("Revolute", "Prismatic") else 0.9)
    setattr(new_actor, "vec_state", state)
    setattr(new_actor, "vec_idx", vi)
    plt.add(new_actor)
    plt.remove(old_actor)
    # sync collections
    j = vec_actors.index(old_actor) if old_actor in vec_actors else None
    if j is not None:
        vec_actors[j] = new_actor
    idx2actor[vi] = new_actor
    return new_actor


def on_rlp_key(event, *,
               plt,
               rlp: dict,
               vec_indices: List[int],
               idx2actor: dict,
               vec_actors: list,
               seg_len: float,
               radius: float,
               angle_step_deg: float = 3.0,
               trans_step: float = 1.0):
    """
    Keyboard controls (apply to ALL vectors in this RLP window):
      Move:  W(+X), S(-X), D(+Y), A(-Y), Space/Z(+Z), X(-Z)
      Rotate: I(+Z), U(-Z), J(+Y), H(-Y), N(+X), B(-X)
      Exit: q / esc
    """
    kraw = (getattr(event, "keyPressed", None)
            or getattr(event, "key", None)
            or getattr(event, "symbol", None)
            or getattr(event, "keypress", "")
            or "")
    k = str(kraw).lower()

    if k in ("q", "esc", "escape"):
        plt.close()
        vedo.close()
        return

    # translation
    d = np.zeros(3, dtype=float)
    if   k == "w": d[:] = (+trans_step, 0.0, 0.0)
    elif k == "s": d[:] = (-trans_step, 0.0, 0.0)
    elif k == "d": d[:] = (0.0, +trans_step, 0.0)
    elif k == "a": d[:] = (0.0, -trans_step, 0.0)
    elif k in (" ", "space"): d[:] = (0.0, 0.0, +trans_step)
    elif k == "z": d[:] = (0.0, 0.0, -trans_step)

    # rotation
    R = None
    th = np.deg2rad(angle_step_deg)
    if   k == "i": R = rotmat([0,0,1], +th)
    elif k == "u": R = rotmat([0,0,1], -th)
    elif k == "j": R = rotmat([0,1,0], +th)
    elif k == "h": R = rotmat([0,1,0], -th)
    elif k == "n": R = rotmat([1,0,0], +th)
    elif k == "b": R = rotmat([1,0,0], -th)

    did_anything = False
    vecs = rlp.get("vectors", []) or []

    for vi in vec_indices:
        if vi < 0 or vi >= len(vecs):
            continue
        v = vecs[vi]

        # update data
        if np.any(d):
            c = np.asarray(v.get("center", rlp.get("center", [0,0,0])), dtype=float)
            c2 = c + d
            v["center"] = c2.tolist()
            if "center" in rlp and isinstance(rlp["center"], (list, tuple)):
                rlp["center"] = (np.asarray(rlp["center"], dtype=float) + d).tolist()
            did_anything = True

        if R is not None and ("n" in v) and v["n"] is not None:
            n = normalize(np.asarray(v["n"], dtype=float))
            n2 = normalize(R @ n)
            v["n"] = n2.tolist()
            did_anything = True

        # update view (rebuild tube)
        if did_anything and vi in idx2actor:
            actor = idx2actor[vi]
            c_now = np.asarray(v.get("center", rlp.get("center", [0,0,0])), dtype=float)
            n_now = normalize(np.asarray(v.get("n"), dtype=float)) if v.get("n") is not None else None
            if n_now is not None and np.isfinite(n_now).all() and np.linalg.norm(n_now) > 1e-12:
                rebuild_tube_actor(
                    plt,
                    old_actor=actor,
                    center=c_now,
                    n=n_now,
                    seg_len=seg_len,
                    radius=radius,
                    state=getattr(actor, "vec_state", v.get("state", "None")),
                    vi=vi,
                    vec_actors=vec_actors,
                    idx2actor=idx2actor
                )

    if did_anything:
        plt.render()


def visualize_and_select_vectors_for_rlps(rlps: List[dict],
                                          parts: List[trimesh.Trimesh],
                                          categories: List[List[trimesh.Trimesh]],
                                          verts: np.ndarray,
                                          faces: np.ndarray,
                                          display: bool = True) -> None:
    """
    One window per RLP:
      - show base mesh + the two parts
      - draw all vectors (Tube)
      - click to toggle type
      - keyboard to move/rotate all vectors of THIS RLP
    """
    if not display:
        return

    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    seg_len = scene_diag * 0.2 if np.isfinite(scene_diag) and scene_diag > 0 else 1.0
    trans_step = scene_diag * 0.02 if np.isfinite(scene_diag) and scene_diag > 0 else 1.0
    radius = seg_len * 0.02

    for i, rlp in enumerate(rlps):
        mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.35)
        if hasattr(mesh_actor, "pickable"):
            mesh_actor.pickable(False)

        p_idx = int(rlp["a"]["parent"])
        q_idx = int(rlp["a"]["child"])
        part_ids = [pid for pid in (p_idx, q_idx) if pid != 0 and 1 <= pid <= len(parts)]
        part_ids = list(dict.fromkeys(part_ids))

        part_colors = [(0.85, 0.2, 0.2), (0.2, 0.4, 0.85)]
        part_actors = []
        for j, pid in enumerate(part_ids):
            for tri in categories[pid]:
                actor = vedo.Mesh([tri.vertices, tri.faces]).c(part_colors[min(j, 1)]).alpha(0.25)
                if hasattr(actor, "pickable"):
                    actor.pickable(False)
                part_actors.append(actor)

        # vector actors
        vecs = rlp.get('vectors', []) or []
        vec_actors = []
        idx2actor = {}   # vi -> actor
        valid_indices = []

        for vi, v in enumerate(vecs):
            c = np.asarray(v.get("center", rlp.get("center", [0, 0, 0])), dtype=float)
            n = normalize(np.asarray(v.get("n"), dtype=float))
            if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                continue
            p0, p1 = c, c + seg_len * n
            a = vedo.Tube([p0, p1], r=radius, cap=True).c("gray").alpha(1.0).lighting('off')
            st = v.get("state", "None")
            setattr(a, "vec_state", st)
            setattr(a, "vec_idx", vi)
            vec_actors.append(a)
            idx2actor[vi] = a
            valid_indices.append(vi)

        plt_rlp = vedo.Plotter(title=f"RLP #{i}: parts {p_idx} ↔ {q_idx}", axes=0)

        # Disable default VTK keybindings
        iren = plt_rlp.interactor
        for _ev in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
            iren.RemoveObservers(_ev)

        # mouse: toggle
        plt_rlp.add_callback(
            "LeftButtonPress",
            partial(on_vector_click, vec_actors=vec_actors, rlp=rlp, click_counter={'count': 0})
        )
        # keyboard: move/rotate
        key_cb = partial(
            on_rlp_key,
            plt=plt_rlp,
            rlp=rlp,
            vec_indices=valid_indices,
            idx2actor=idx2actor,
            vec_actors=vec_actors,
            seg_len=seg_len,
            radius=radius,
            angle_step_deg=1.5,
            trans_step=trans_step
        )
        plt_rlp.add_callback("KeyPress",  key_cb)
        plt_rlp.add_callback("CharEvent", key_cb)

        import time
        import functions
        gui_start = time.time()
        plt_rlp.show([mesh_actor, *part_actors, *vec_actors], interactive=True).close()
        functions.total_gui_time += time.time() - gui_start


# =============================================================================
# Visualization of boundary loops
# =============================================================================
def visualize_boundary_loops(
    ms: pymeshlab.MeshSet,
    categories: list[list[trimesh.Trimesh]],
    idx_part: int,
    loops: list[np.ndarray],
    *,
    boundary_indices: np.ndarray | None = None,
    show_original_mesh: bool = True,
    tube_radius: float = 0.4,
    sphere_radius: float = 0.8,
    boundary_point_size: float = 6.0,
    boundary_point_color = "black",
    background: str = "white",
    display: bool = True,
):
    """
    Visualize boundary loops; optionally overlay boundary vertex indices as points.
    """
    if not display:
        return

    # 0) get original mesh geometry
    ms.set_current_mesh(0)
    verts = ms.current_mesh().vertex_matrix()
    faces = ms.current_mesh().face_matrix().astype(np.int64)

    actors = []

    # 1) original mesh (semi-transparent)
    if show_original_mesh:
        m_orig = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.25)
        m_orig.lighting("plastic")
        actors.append(m_orig)

    # 2) selected part mesh (if any)
    vpart = trimesh_list_to_vedo_mesh(categories[idx_part])
    if vpart is not None:
        vpart.c("dodgerblue").alpha(0.35).lw(0.5).lighting("plastic")
        actors.append(vpart)

    if len(loops) < 1: 
        if display: 
             plt = vedo.Plotter(bg=background, title="Boundary loops (Empty)")
             plt.show(actors, viewup="z").close()
        return

    # 3) draw loops (as tubes) + loop centers (small spheres)
    cmap = vedo.color_map(range(len(loops)), "Set1")  # distinct colors
    for i, loop in enumerate(loops):
        loop = np.asarray(loop, dtype=int).ravel()
        if loop.size == 0:
            continue
        pts = verts[loop]
        line = vedo.Line(pts, closed=False).c(cmap[i]).lw(2)
        if hasattr(line, "tube") and callable(getattr(line, "tube")):
            try:
                tube = line.tube(radius=tube_radius).c(cmap[i])
            except TypeError:
                tube = line.tube(tube_radius).c(cmap[i])
        else:
            tube = vedo.Tube(pts, r=tube_radius, cap=True).c(cmap[i])
        actors.append(tube)

        center = pts.mean(axis=0)
        s = vedo.Sphere(pos=center, r=sphere_radius, res=16).c(cmap[i]).alpha(0.5)
        actors.append(s)

    # 4) boundary indices
    if boundary_indices is not None and len(boundary_indices) > 0:
        bi = np.asarray(boundary_indices, int)
        bi = bi[(bi >= 0) & (bi < verts.shape[0])]
        if bi.size > 0:
            dots = vedo.Spheres(verts[bi], r=boundary_point_size, c="red", alpha=1.0)
            actors.append(dots)

    # 5) axes & show
    plt = vedo.Plotter(bg=background, title="Boundary loops (+ boundary points)")
    plt.show(actors, viewup="z").close()


def visualize_k_hop_plane(
    verts: np.ndarray,
    faces: np.ndarray,
    idx_pos: np.ndarray,
    idx_neg: np.ndarray,
    planes: list, # list of (n, d)
    center: np.ndarray,
    title: str = "k-hop neighbors and plane",
    display: bool = True,
    loop_indices: np.ndarray | None = None, # Added parameter
):
    """
    Visualize k-hop neighbors and the estimated plane.
    """
    if not display:
        return

    actors = []

    # 1) original mesh (semi-transparent)
    mesh = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.2)
    actors.append(mesh)

    # 2) Positive and negative neighbor vertices (more transparent)
    if idx_pos.size > 0:
        pos_pts = vedo.Spheres(verts[idx_pos], r=0.0025, c="lightblue", res=8).alpha(0.3) # Reduced size
        actors.append(pos_pts)
    if idx_neg.size > 0:
        neg_pts = vedo.Spheres(verts[idx_neg], r=0.0025, c="salmon", res=8).alpha(0.3) # Reduced size
        actors.append(neg_pts)
        
    # 3) Boundary loop points (stronger)
    if loop_indices is not None and loop_indices.size > 0:
        loop_pts = vedo.Spheres(verts[loop_indices], r=0.006, c="red", res=8).alpha(1.0) # Reduced size but larger than neighbors
        actors.append(loop_pts)

    # 4) Estimated plane(s)
    plane_colors = ["green", "cyan", "magenta", "yellow"]
    for i, (n, d) in enumerate(planes):
        plane_pos = center - (np.dot(center, n) + d) * n
        plane_actor = vedo.Plane(pos=plane_pos, normal=n, s=(1.0, 1.0)).c(plane_colors[i % len(plane_colors)]).alpha(0.2) # Increased transparency
        actors.append(plane_actor)

    # 5) Show plot
    plt = vedo.Plotter(bg="white", title=title)
    plt.show(actors, viewup="z").close()


def visualize_rlp_clustering(verts: np.ndarray,
                             faces: np.ndarray,
                             parts: List[trimesh.Trimesh],
                             categories: List[List[trimesh.Trimesh]],
                             results_list: List[dict],
                             rlps: List[dict]) -> None:
    """
    Detailed visualization of RLP clustering showing linear and polyhedral intersection
    matches separately for each generated RLP.
    """
    print("[info] Starting RLP clustering visualization...")
    
    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    arrow_len = scene_diag * 0.15 if np.isfinite(scene_diag) and scene_diag > 0 else 1.0
    radius_arrow = arrow_len * 0.025
    radius_sphere = arrow_len * 0.05
    
    # Custom arrow settings: thicker shaft, larger head
    shaft_rad = radius_arrow * 2.5
    head_rad = radius_arrow * 5.5
    head_len = arrow_len * 0.25
    
    for idx_r, rlp in enumerate(rlps):
        pair = rlp.get('parts')
        if not pair:
            continue
        i, j = pair
        a = rlp.get("a")
        b = rlp.get("b")
        if not a or not b:
            continue
            
        print(f"[info] Visualizing RLP #{idx_r} (parts {i} ↔ {j}). Close window to proceed.")
        
        # 1. Collect and prepare k-hop vertices for this RLP
        part_i_verts = set()
        part_j_verts = set()
        
        if 'idx_pos' in a: part_i_verts.update(a['idx_pos'])
        if 'idx_neg' in a: part_j_verts.update(a['idx_neg'])
        if 'idx_pos' in b: part_j_verts.update(b['idx_pos'])
        if 'idx_neg' in b: part_i_verts.update(b['idx_neg'])
        
        # Create base elements shared across windows
        mesh_actor = vedo.Mesh([verts, faces]).c("white").alpha(0.08)
        if hasattr(mesh_actor, "pickable"):
            mesh_actor.pickable(False)
            
        spheres_i = None
        if part_i_verts:
            pts_i = verts[np.array(list(part_i_verts), dtype=int)]
            spheres_i = vedo.Spheres(pts_i, r=scene_diag*0.003, c="red").alpha(0.1) # 90% transparent (alpha=0.1)
            if hasattr(spheres_i, "pickable"): spheres_i.pickable(False)
            
        spheres_j = None
        if part_j_verts:
            pts_j = verts[np.array(list(part_j_verts), dtype=int)]
            spheres_j = vedo.Spheres(pts_j, r=scene_diag*0.003, c="blue").alpha(0.1) # 90% transparent (alpha=0.1)
            if hasattr(spheres_j, "pickable"): spheres_j.pickable(False)
            
        ca = np.asarray(a["center"], dtype=float)
        cb = np.asarray(b["center"], dtype=float)
        c_pair = np.asarray(rlp["center"], dtype=float)
        
        # Connection line between matching loop centers
        conn_line = vedo.Line(ca, cb).c("orange").lw(3)
        
        # -------------------------------------------------------------
        # Window 1: Linear Normal RLP
        # -------------------------------------------------------------
        lin_actors = [mesh_actor, conn_line]
        if spheres_i: lin_actors.append(spheres_i)
        if spheres_j: lin_actors.append(spheres_j)
        
        # Red arrow for part i -> j loop A
        lin_A = a.get('linear', {})
        if lin_A.get('plane') and len(lin_A['plane']) > 0:
            nA = np.asarray(lin_A['plane'][0][0], dtype=float)
        else:
            nA = np.asarray(a.get('pca_normal', [0,0,1]), dtype=float)
        nA = normalize(nA)
        arrow_a = vedo.Arrow(ca, ca + arrow_len * nA, c="red", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
        lbl_a = vedo.Text3D(f"Linear Normal (Part {i})", pos=ca + np.array([0, 0, radius_sphere*1.5]), s=radius_sphere*0.4, c="red")
        lin_actors.extend([arrow_a, lbl_a])
        
        # Blue arrow for part j -> i loop B (aligned with nA)
        lin_B = b.get('linear', {})
        if lin_B.get('plane') and len(lin_B['plane']) > 0:
            nB = np.asarray(lin_B['plane'][0][0], dtype=float)
        else:
            nB = np.asarray(b.get('pca_normal', [0,0,1]), dtype=float)
        nB = normalize(nB)
        if np.dot(nA, nB) < 0.0:
            nB = -nB
        arrow_b = vedo.Arrow(cb, cb + arrow_len * nB, c="blue", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
        lbl_b = vedo.Text3D(f"Linear Normal (Part {j})", pos=cb + np.array([0, 0, radius_sphere*1.5]), s=radius_sphere*0.4, c="blue")
        lin_actors.extend([arrow_b, lbl_b])
        
        # Hot-pink arrow for merged Linear RLP
        n_pair = normalize(rlp["normal"])
        arrow_rlp = vedo.Arrow(c_pair, c_pair + arrow_len * n_pair, c="magenta", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
        sphere_rlp = vedo.Sphere(pos=c_pair, r=radius_sphere*1.2).c("magenta")
        lbl_rlp = vedo.Text3D(f"Merged Linear RLP\nCost: {rlp['cost']:.3f}", pos=c_pair - np.array([0, 0, radius_sphere*2.0]), s=radius_sphere*0.45, c="magenta")
        lin_actors.extend([arrow_rlp, sphere_rlp, lbl_rlp])
        
        plt1 = vedo.Plotter(title=f"RLP #{idx_r} (Linear Normal Match): parts {i} ↔ {j}", axes=0)
        plt1.show(lin_actors, interactive=True).close()
        
        # -------------------------------------------------------------
        # Window 2: Polyhedral Intersection RLP
        # -------------------------------------------------------------
        poly_A = a.get('polyhedral', {})
        poly_B = b.get('polyhedral', {})
        
        lines_A = []
        if poly_A.get('plane') and len(poly_A['plane']) > 0:
            planes = poly_A['plane']; m = len(planes)
            for u in range(m):
                for v in range(u + 1, m):
                    n_cross = np.cross(planes[u][0], planes[v][0])
                    if np.linalg.norm(n_cross) > 1e-6:
                        lines_A.append(normalize(n_cross))
                        
        lines_B = []
        if poly_B.get('plane') and len(poly_B['plane']) > 0:
            planes = poly_B['plane']; m = len(planes)
            for u in range(m):
                for v in range(u + 1, m):
                    n_cross = np.cross(planes[u][0], planes[v][0])
                    if np.linalg.norm(n_cross) > 1e-6:
                        lines_B.append(normalize(n_cross))
                        
        # Skip polyhedral window if no polyhedral lines found on either side
        if not lines_A and not lines_B:
            continue
            
        poly_actors = [mesh_actor, conn_line]
        if spheres_i: poly_actors.append(spheres_i)
        if spheres_j: poly_actors.append(spheres_j)
        
        # Draw red polyhedral cross lines for A
        for idx_a, la in enumerate(lines_A):
            arrow_la = vedo.Arrow(ca, ca + arrow_len * la, c="red", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
            lbl_la = vedo.Text3D(f"Poly Line A#{idx_a}", pos=ca + la * arrow_len * 0.5 + [0,0,radius_sphere], s=radius_sphere*0.35, c="red")
            poly_actors.extend([arrow_la, lbl_la])
            
        # Draw blue polyhedral cross lines for B
        for idx_b, lb in enumerate(lines_B):
            arrow_lb = vedo.Arrow(cb, cb + arrow_len * lb, c="blue", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
            lbl_lb = vedo.Text3D(f"Poly Line B#{idx_b}", pos=cb + lb * arrow_len * 0.5 + [0,0,radius_sphere], s=radius_sphere*0.35, c="blue")
            poly_actors.extend([arrow_lb, lbl_lb])
            
        # Perform matching of lines_A and lines_B (same logic as prepare_vectors_for_rlps)
        candidates_match = []
        for idx_a, la in enumerate(lines_A):
            for idx_b, lb in enumerate(lines_B):
                dot_val = abs(np.dot(la, lb))
                candidates_match.append((dot_val, idx_a, idx_b))
        candidates_match.sort(key=lambda x: x[0], reverse=True)
        
        matched_A = set()
        matched_B = set()
        cos_30 = 0.866025
        merged_lines = []
        
        for dot_val, idx_a, idx_b in candidates_match:
            if dot_val < cos_30:
                break
            if idx_a in matched_A or idx_b in matched_B:
                continue
            la = lines_A[idx_a]
            lb = lines_B[idx_b]
            if np.dot(la, lb) < 0.0:
                lb = -lb
            l_merged = normalize(la + lb)
            merged_lines.append(l_merged)
            matched_A.add(idx_a)
            matched_B.add(idx_b)
            
        # Add unmatched lines
        for idx_a, la in enumerate(lines_A):
            if idx_a not in matched_A:
                merged_lines.append(la)
        for idx_b, lb in enumerate(lines_B):
            if idx_b not in matched_B:
                merged_lines.append(lb)
                
        # Draw final hot-pink merged polyhedral lines (Window 2 actors)
        for idx_m, line in enumerate(merged_lines):
            # Intersection line (merged)
            arrow_m = vedo.Arrow(c_pair, c_pair + arrow_len * line, c="magenta", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
            lbl_m = vedo.Text3D(f"Merged Poly Line #{idx_m}", pos=c_pair + line * arrow_len * 0.7 + [0,0,radius_sphere], s=radius_sphere*0.4, c="magenta")
            poly_actors.extend([arrow_m, lbl_m])
            
        plt2 = vedo.Plotter(title=f"RLP #{idx_r} (Polyhedral Intersection Match): parts {i} ↔ {j}", axes=0)
        plt2.show(poly_actors, interactive=True).close()
        
        # -------------------------------------------------------------
        # Window 3: 3rd Perpendicular Coordinate Frame
        # -------------------------------------------------------------
        frame_actors = [mesh_actor]
        if spheres_i: frame_actors.append(spheres_i)
        if spheres_j: frame_actors.append(spheres_j)
        
        # 1. Main plane normal (merged) in magenta
        arrow_norm = vedo.Arrow(c_pair, c_pair + arrow_len * n_pair, c="magenta", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
        lbl_norm = vedo.Text3D("1. Plane Normal", pos=c_pair + n_pair * arrow_len * 0.7 + [0,0,radius_sphere], s=radius_sphere*0.4, c="magenta")
        frame_actors.extend([arrow_norm, lbl_norm])
        
        # 2. First Intersection line & its perpendicular 3rd recommended vector in magenta
        has_frame = False
        if len(merged_lines) > 0:
            line = merged_lines[0]
            # 2. Merged Intersection line in magenta
            arrow_m = vedo.Arrow(c_pair, c_pair + arrow_len * line, c="magenta", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
            lbl_m = vedo.Text3D("2. Intersection Line", pos=c_pair + line * arrow_len * 0.7 + [0,0,radius_sphere*1.5], s=radius_sphere*0.4, c="magenta")
            frame_actors.extend([arrow_m, lbl_m])
            
            # 3. 3rd Recommended Vector (perpendicular to both normal and intersection line) in magenta
            n_third = np.cross(n_pair, line)
            if np.linalg.norm(n_third) > 1e-6:
                n_third = normalize(n_third)
                arrow_t = vedo.Arrow(c_pair, c_pair + arrow_len * n_third, c="magenta", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
                lbl_t = vedo.Text3D("3. Perp Recommended Axis", pos=c_pair + n_third * arrow_len * 0.7 - [0,0,radius_sphere*1.5], s=radius_sphere*0.4, c="magenta")
                frame_actors.extend([arrow_t, lbl_t])
                has_frame = True
                
        if has_frame:
            plt3 = vedo.Plotter(title=f"RLP #{idx_r} (3rd Perpendicular Coordinate Frame): parts {i} ↔ {j}", axes=0)
            plt3.show(frame_actors, interactive=True).close()


def visualize_all_boundary_loops(
    verts: np.ndarray,
    faces: np.ndarray,
    loops: list,
    loop_neighbors: list,
    parent_part: int,
    title: str = "All Boundary Loops",
    display: bool = True,
    show_none_loops: bool = False,
):
    """
    Visualize all boundary loops for a part with different colors and labels indicating neighbors.
    """
    if not display:
        return

    actors = []

    # 1) Base mesh (semi-transparent)
    mesh = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.2)
    actors.append(mesh)

    # Beautiful pastel colors: (R, G, B) on 0-255 scale
    pastel_rgbs = [
        (255, 120, 120),  # Pastel Red
        (120, 220, 120),  # Pastel Green
        (245, 220, 100),  # Pastel Yellow
        (120, 180, 255),  # Pastel Blue
        (200, 150, 255),  # Pastel Purple
        (255, 180, 120),  # Pastel Orange
        (255, 150, 200),  # Pastel Pink
        (120, 220, 220),  # Pastel Teal
        (150, 255, 200),  # Pastel Mint
    ]
    
    # Calculate scene diagonal for scaling text sizes
    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    sphere_r = scene_diag * 0.005 if np.isfinite(scene_diag) and scene_diag > 0 else 0.005

    for i, loop in enumerate(loops):
        neighbors = loop_neighbors[i]
        
        # Optionally skip showing loops with no neighbors (connecting to None/N)
        if not neighbors and not show_none_loops:
            continue

        color_rgb = pastel_rgbs[i % len(pastel_rgbs)]
        color_vedo = '#%02x%02x%02x' % color_rgb
        loop_idx = np.asarray(loop, dtype=int)
        
        # Display loop vertices as dots
        loop_pts = vedo.Spheres(verts[loop_idx], r=sphere_r, c=color_vedo, res=8).alpha(1.0)
        actors.append(loop_pts)
        
        # Print information to the terminal with a colored square block (ANSI TrueColor)
        neighbors_str = ",".join(map(str, neighbors)) if neighbors else "N"
        r, g, b = color_rgb
        color_block = f"\033[38;2;{r};{g};{b}m■\033[0m"
        print(f"[info] Cycle {i+1} ({color_block}): Part {parent_part} -> {neighbors_str}")

    # Show plot
    plt = vedo.Plotter(bg="white", title=title)
    plt.show(actors, viewup="z").close()


def visualize_segmented_parts(
    verts: np.ndarray,
    faces: np.ndarray,
    vert_label: np.ndarray,
    title: str = "Segmented Parts Preview",
    display: bool = True,
):
    """
    Visualize the entire mesh with vertices colored by their segmented part ID.
    Highly transparent (alpha=0.4) to show the interior and structure.
    """
    if not display:
        return

    # Distinct colormap for categories (0: background, 1..N: parts)
    colormap = np.array([
        [220, 220, 220],  # 0: Light gray
        [240, 50, 50],    # 1: Red
        [50, 144, 240],   # 2: Blue
        [50, 204, 50],    # 3: Green
        [240, 163, 10],   # 4: Orange
        [150, 50, 240],   # 5: Purple
        [10, 218, 240],   # 6: Cyan
        [240, 50, 180],   # 7: Magenta
        [240, 228, 10],   # 8: Yellow
        [140, 80, 30],    # 9: Brown
        [50, 240, 163],   # 10: Lime/Teal
    ], dtype=np.uint8)

    num_colors = len(colormap)
    # Map labels to colors
    rgb_colors = np.array([colormap[int(l) % num_colors] for l in vert_label], dtype=np.uint8)

    mesh = vedo.Mesh([verts, faces])
    mesh.pointcolors = rgb_colors
    mesh.alpha(0.4)  # High transparency as requested

    plt = vedo.Plotter(bg="white", title=title)
    plt.show([mesh], viewup="z").close()
