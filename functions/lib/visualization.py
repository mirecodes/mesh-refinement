import numpy as np
import vedo
import trimesh
from typing import List, Dict, Any
from functools import partial
import pymeshlab

from functions.lib.geometry import normalize, rotmat

# =============================================================================
# Pastel Palette (Specified Sequence)
# =============================================================================
PASTEL_HEX = [
    "#66C5CC",
    "#F6CF71",
    "#F89C74",
    "#DCB0F2",
    "#87C55F",
    "#9EB9F3",
    "#FE88B1",
    "#C9DB74",
    "#8BE0A4",
    "#B497E7",
    "#B3B3B3",
]

PASTEL_RGB_UINT8 = np.array([
    [102, 197, 204],  # #66C5CC
    [246, 207, 113],  # #F6CF71
    [248, 156, 116],  # #F89C74
    [220, 176, 242],  # #DCB0F2
    [135, 197, 95],   # #87C55F
    [158, 185, 243],  # #9EB9F3
    [254, 136, 177],  # #FE88B1
    [201, 219, 116],  # #C9DB74
    [139, 224, 164],  # #8BE0A4
    [180, 151, 231],  # #B497E7
    [179, 179, 179],  # #B3B3B3
], dtype=np.uint8)

PASTEL_RGB_FLOAT = (PASTEL_RGB_UINT8 / 255.0).tolist()


def get_standard_arrow_params(verts: np.ndarray):
    """
    Unified arrow size & thickness parameters across Plane estimation and RLP visualizations.
    """
    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    arrow_len = scene_diag * 0.25 if np.isfinite(scene_diag) and scene_diag > 0 else 1.5
    radius_arrow = arrow_len * 0.035
    radius_sphere = arrow_len * 0.06
    shaft_rad = radius_arrow * 3.5
    head_rad = radius_arrow * 7.5
    head_len = arrow_len * 0.30
    return arrow_len, shaft_rad, head_rad, head_len, radius_sphere


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
    vmeshes.append(base_actor); rand_colors.append(PASTEL_RGB_FLOAT[-1])
    for i, part in enumerate(parts, start=1):
        actor = vedo.Mesh([part.vertices, part.faces])
        color = PASTEL_RGB_FLOAT[(i - 1) % len(PASTEL_RGB_FLOAT)]
        actor.c(color).alpha(0.5)
        actor.idx_part = i
        vmeshes.append(actor); rand_colors.append(color)

    belongings = [1 for _ in range(len(vmeshes))]; belongings[0] = 0
    prevs = [1 for _ in range(len(vmeshes))]; prevs[0] = 0
    shared = {"mode": 0}

    plt = vedo.Plotter(title="Part selection", size=(1800, 1800))
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
    """Toggle None → Revolute → Prismatic → None for cylinder Tube vectors."""
    act = getattr(event, "actor", None)
    if act is None or act not in vec_actors:
        return
    st = getattr(act, "vec_state", "None")
    rlp_dict = getattr(act, "rlp_dict", rlp)
    v_idx = getattr(act, "v_idx_in_rlp", getattr(act, "vec_idx", None))

    if st == "None":
        act.c("yellow").alpha(1.0)
        setattr(act, "vec_state", "Revolute")
        if rlp_dict and 'vectors' in rlp_dict and v_idx is not None and v_idx < len(rlp_dict['vectors']):
            click_counter['count'] = int(click_counter.get('count', 0)) + 1
            rlp_dict['vectors'][v_idx]['state'] = "Revolute"
            rlp_dict['vectors'][v_idx]['order'] = int(click_counter['count'])
    elif st == "Revolute":
        act.c("blue").alpha(1.0)
        setattr(act, "vec_state", "Prismatic")
        if rlp_dict and 'vectors' in rlp_dict and v_idx is not None and v_idx < len(rlp_dict['vectors']):
            rlp_dict['vectors'][v_idx]['state'] = "Prismatic"
    else:
        act.c("gray").alpha(0.85)
        setattr(act, "vec_state", "None")
        if rlp_dict and 'vectors' in rlp_dict and v_idx is not None and v_idx < len(rlp_dict['vectors']):
            rlp_dict['vectors'][v_idx]['state'] = "None"
            rlp_dict['vectors'][v_idx]['order'] = None
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
    """Recreate Tube (no arrow head, 2/3 thickness, gray/yellow/blue by state)."""
    p0 = np.asarray(center, dtype=float) - (seg_len * 0.5) * normalize(np.asarray(n, dtype=float))
    p1 = np.asarray(center, dtype=float) + (seg_len * 0.5) * normalize(np.asarray(n, dtype=float))
    tube_color = "yellow" if state == "Revolute" else "blue" if state == "Prismatic" else "gray"
    tube_r = radius * (2.0 / 3.0)
    new_actor = vedo.Tube([p0, p1], r=tube_r, cap=True).c(tube_color).lighting('off')
    new_actor.alpha(1.0 if state in ("Revolute", "Prismatic") else 0.85)
    setattr(new_actor, "vec_state", state)
    setattr(new_actor, "vec_idx", vi)
    setattr(new_actor, "rlp_dict", getattr(old_actor, "rlp_dict", None))
    setattr(new_actor, "v_idx_in_rlp", getattr(old_actor, "v_idx_in_rlp", vi))
    plt.add(new_actor)
    plt.remove(old_actor)
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
                                          vert_label: np.ndarray | None = None,
                                          display: bool = True) -> None:
    """
    Joint selection stage per Part Pair (i, j):
      - Color FULL mesh area for Part i and Part j using vert_label in sequence pastel colors (alpha=0.4)
      - Display ALL RLP vectors (plus pair additional intersection vectors) for Pair (i, j) together
      - Vectors rendered as slender cylinder Tubes (no arrow head, ~1/3 thickness)
      - Colors: Gray (None) -> Yellow (Revolute) -> Blue (Prismatic) -> Gray (None)
    """
    if not display or not rlps:
        return

    from collections import defaultdict

    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    arrow_len, shaft_rad, head_rad, head_len, radius_sphere = get_standard_arrow_params(verts)
    seg_len = arrow_len
    # Double tube thickness (diameter 2x larger than previous 0.0035)
    tube_r = scene_diag * 0.007 if np.isfinite(scene_diag) and scene_diag > 0 else 0.010
    radius = tube_r
    trans_step = scene_diag * 0.02 if np.isfinite(scene_diag) and scene_diag > 0 else 1.0

    # Group RLPs by unique part pair (p_idx, q_idx)
    pair_map = defaultdict(list)
    for rlp in rlps:
        p_idx = int(rlp["a"]["parent"])
        q_idx = int(rlp["a"]["child"])
        pair_key = (min(p_idx, q_idx), max(p_idx, q_idx))
        pair_map[pair_key].append(rlp)

    num_pastel = len(PASTEL_RGB_UINT8)
    bg_color = np.array([220, 220, 220], dtype=np.uint8)

    for (p_idx, q_idx), group_rlps in pair_map.items():
        if p_idx == 0 and q_idx == 0:
            continue

        # 1. Color full mesh area for Part p_idx and Part q_idx (matching Segmented Parts Preview)
        rgb_colors = np.zeros((len(verts), 3), dtype=np.uint8)
        if vert_label is not None and len(vert_label) == len(verts):
            for v_idx_i, l_val in enumerate(vert_label):
                l_int = int(l_val)
                if l_int == p_idx:
                    rgb_colors[v_idx_i] = PASTEL_RGB_UINT8[(p_idx - 1) % num_pastel]
                elif l_int == q_idx:
                    rgb_colors[v_idx_i] = PASTEL_RGB_UINT8[(q_idx - 1) % num_pastel]
                else:
                    rgb_colors[v_idx_i] = bg_color
        else:
            rgb_colors = np.tile(bg_color, (len(verts), 1))
            part_p_verts = set()
            part_q_verts = set()
            for rlp in group_rlps:
                a = rlp.get("a", {})
                b = rlp.get("b", {})
                p_curr = int(a.get("parent", p_idx))
                if p_curr == p_idx:
                    if 'idx_pos' in a: part_p_verts.update(a['idx_pos'])
                    if 'idx_neg' in b: part_p_verts.update(b['idx_neg'])
                    if 'idx_pos' in b: part_q_verts.update(b['idx_pos'])
                    if 'idx_neg' in a: part_q_verts.update(a['idx_neg'])
                else:
                    if 'idx_pos' in a: part_q_verts.update(a['idx_pos'])
                    if 'idx_neg' in b: part_q_verts.update(b['idx_neg'])
                    if 'idx_pos' in b: part_p_verts.update(b['idx_pos'])
                    if 'idx_neg' in a: part_p_verts.update(a['idx_neg'])

            if part_q_verts:
                rgb_colors[np.array(list(part_q_verts), dtype=int)] = PASTEL_RGB_UINT8[(q_idx - 1) % num_pastel]
            if part_p_verts:
                rgb_colors[np.array(list(part_p_verts), dtype=int)] = PASTEL_RGB_UINT8[(p_idx - 1) % num_pastel]

        mesh_actor = vedo.Mesh([verts, faces])
        mesh_actor.pointcolors = rgb_colors
        mesh_actor.alpha(0.4)
        if hasattr(mesh_actor, "pickable"):
            mesh_actor.pickable(False)

        # 2. Collect ALL vectors from ALL RLPs in this group (including pair additional intersection vectors)
        vec_actors = []
        idx2actor = {}
        valid_indices = []
        global_vi = 0

        for rlp in group_rlps:
            vecs = rlp.get('vectors', []) or []
            for v_in_rlp_idx, v in enumerate(vecs):
                c = np.asarray(v.get("center", rlp.get("center", [0, 0, 0])), dtype=float)
                n = normalize(np.asarray(v.get("n"), dtype=float))
                if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
                    continue

                st = v.get("state", "None")
                tube_color = "yellow" if st == "Revolute" else "blue" if st == "Prismatic" else "gray"

                p0 = c - (arrow_len * 0.5) * n
                p1 = c + (arrow_len * 0.5) * n

                a_actor = vedo.Tube([p0, p1], r=tube_r, cap=True).c(tube_color).lighting('off')
                a_actor.alpha(1.0 if st in ("Revolute", "Prismatic") else 0.85)
                setattr(a_actor, "vec_state", st)
                setattr(a_actor, "vec_idx", global_vi)
                setattr(a_actor, "rlp_dict", rlp)
                setattr(a_actor, "v_idx_in_rlp", v_in_rlp_idx)

                vec_actors.append(a_actor)
                idx2actor[global_vi] = a_actor
                valid_indices.append(global_vi)
                global_vi += 1

        plt_rlp = vedo.Plotter(title=f"Joint Selection: Parts {p_idx} ↔ {q_idx}", axes=0, size=(1800, 1800))

        # Disable default VTK keybindings
        iren = plt_rlp.interactor
        for _ev in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
            iren.RemoveObservers(_ev)

        # mouse: toggle
        plt_rlp.add_callback(
            "LeftButtonPress",
            partial(on_vector_click, vec_actors=vec_actors, rlp=group_rlps[0], click_counter={'count': 0})
        )

        # keyboard: move/rotate
        first_rlp = group_rlps[0]
        key_cb = partial(
            on_rlp_key,
            plt=plt_rlp,
            rlp=first_rlp,
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

        print(f"[info] Joint Selection for Pair Parts {p_idx} ↔ {q_idx} ({len(vec_actors)} vector candidates total):")
        print("  \033[37m[Gray Tube]\033[0m Default Candidate (None)")
        print("  \033[33m[Yellow Tube]\033[0m Click 1: Revolute Joint")
        print("  \033[34m[Blue Tube]\033[0m Click 2: Prismatic Joint")

        import time
        import functions
        gui_start = time.time()
        plt_rlp.show([mesh_actor, *vec_actors], interactive=True).close()
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
        part_color = PASTEL_RGB_FLOAT[(idx_part - 1) % len(PASTEL_RGB_FLOAT)]
        vpart.c(part_color).alpha(0.35).lw(0.5).lighting("plastic")
        actors.append(vpart)

    if len(loops) < 1: 
        if display: 
             plt = vedo.Plotter(bg=background, title="Boundary loops (Empty)", size=(1800, 1800))
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
            dots = vedo.Spheres(verts[bi], r=boundary_point_size, c="#B3B3B3", alpha=1.0)
            actors.append(dots)

    # 5) axes & show
    plt = vedo.Plotter(bg=background, title="Boundary loops (+ boundary points)", size=(1800, 1800))
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
    loop_indices: np.ndarray | None = None,
    parent_part: int = 1,
):
    """
    Visualize k-hop local geometry with black wireframe edges on submesh,
    base mesh in bright light gray, boundary loops as bubble spheres,
    and estimated planes in light gray.
    """
    if not display:
        return

    actors = []

    # 1) Base original mesh (bright light gray, clean without heavy edges)
    mesh_base = vedo.Mesh([verts, faces]).c("lightgray").alpha(0.18)
    actors.append(mesh_base)

    num_pastel = len(PASTEL_RGB_UINT8)
    pos_rgb = PASTEL_RGB_UINT8[(parent_part - 1) % num_pastel]
    pos_color = '#%02x%02x%02x' % tuple(pos_rgb)
    neg_rgb = np.array([179, 179, 179], dtype=np.uint8)

    # 2) Extract and render k-hop local geometry faces with black wireframe edges
    khop_verts_set = set()
    if idx_pos is not None and idx_pos.size > 0:
        khop_verts_set.update(idx_pos.tolist())
    if idx_neg is not None and idx_neg.size > 0:
        khop_verts_set.update(idx_neg.tolist())

    if khop_verts_set:
        in_khop_mask = np.isin(faces[:, 0], list(khop_verts_set)) | \
                       np.isin(faces[:, 1], list(khop_verts_set)) | \
                       np.isin(faces[:, 2], list(khop_verts_set))
        khop_faces = faces[in_khop_mask]

        if len(khop_faces) > 0:
            base_rgb = np.array([230, 230, 230], dtype=np.uint8)
            rgb_colors = np.tile(base_rgb, (len(verts), 1))

            if idx_neg is not None and idx_neg.size > 0:
                rgb_colors[idx_neg] = neg_rgb
            if idx_pos is not None and idx_pos.size > 0:
                rgb_colors[idx_pos] = pos_rgb

            khop_mesh = vedo.Mesh([verts, khop_faces])
            khop_mesh.pointcolors = rgb_colors
            khop_mesh.alpha(0.7)
            khop_mesh.lw(0.8).lc("black")  # Black wireframe edges ONLY on k-hop local geometry submesh
            actors.append(khop_mesh)

    # 3) Boundary loop points as bubble spheres (기존 bubble 형태로 표시)
    if loop_indices is not None and loop_indices.size > 0:
        loop_pts = vedo.Spheres(verts[loop_indices], r=0.006, c=pos_color, res=8).alpha(1.0)
        actors.append(loop_pts)

    # 4) Estimated plane(s) - both planes in light gray (#B3B3B3)
    plane_color = "#B3B3B3"
    for i, (n, d) in enumerate(planes):
        plane_pos = center - (np.dot(center, n) + d) * n
        plane_actor = vedo.Plane(pos=plane_pos, normal=n, s=(1.0, 1.0)).c(plane_color).alpha(0.25)
        actors.append(plane_actor)

    # 5) Vector visualization according to plane estimation type (Unified size & colors):
    # Single plane -> Normal vector (Orange); Polyhedral planes -> Intersection vector (DodgerBlue)
    arrow_len, shaft_rad, head_rad, head_len, radius_sphere = get_standard_arrow_params(verts)

    if len(planes) == 1:
        n_single = normalize(np.asarray(planes[0][0], dtype=float))
        arrow_norm = vedo.Arrow(center, center + arrow_len * n_single, c="orange", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
        actors.append(arrow_norm)
        print(f"[info] {title}")
        print("  \033[38;2;255;127;14m[Normal Vector - Orange]\033[0m Single Plane Normal")
    elif len(planes) > 1:
        n1 = normalize(np.asarray(planes[0][0], dtype=float))
        n2 = normalize(np.asarray(planes[1][0], dtype=float))
        n_inter = np.cross(n1, n2)
        if np.linalg.norm(n_inter) > 1e-6:
            n_inter = normalize(n_inter)
            arrow_inter = vedo.Arrow(center, center + arrow_len * n_inter, c="dodgerblue", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
            actors.append(arrow_inter)
            print(f"[info] {title}")
            print("  \033[38;2;30;144;255m[Intersection Vector - Blue]\033[0m Polyhedral Planes Intersection Vector")

    # 6) Show plot
    plt = vedo.Plotter(bg="white", title=title, size=(1800, 1800))
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
    
    arrow_len, shaft_rad, head_rad, head_len, radius_sphere = get_standard_arrow_params(verts)
    num_pastel = len(PASTEL_RGB_FLOAT)
    
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
        
        # Color base mesh directly with vertex pointcolors (matching segment result preview)
        num_pastel = len(PASTEL_RGB_UINT8)
        bg_color = np.array([220, 220, 220], dtype=np.uint8)
        color_i_rgb = PASTEL_RGB_UINT8[(i - 1) % num_pastel]
        color_j_rgb = PASTEL_RGB_UINT8[(j - 1) % num_pastel]

        rgb_colors = np.tile(bg_color, (len(verts), 1))

        part_j_verts = set()
        if 'idx_pos' in b: part_j_verts.update(b['idx_pos'])
        if 'idx_neg' in a: part_j_verts.update(a['idx_neg'])
        if part_j_verts:
            rgb_colors[np.array(list(part_j_verts), dtype=int)] = color_j_rgb

        part_i_verts = set()
        if 'idx_pos' in a: part_i_verts.update(a['idx_pos'])
        if 'idx_neg' in b: part_i_verts.update(b['idx_neg'])
        if part_i_verts:
            rgb_colors[np.array(list(part_i_verts), dtype=int)] = color_i_rgb

        mesh_actor = vedo.Mesh([verts, faces])
        mesh_actor.pointcolors = rgb_colors
        mesh_actor.alpha(0.4)
        if hasattr(mesh_actor, "pickable"):
            mesh_actor.pickable(False)

        base_part_actors = [mesh_actor]
            
        ca = np.asarray(a["center"], dtype=float)
        cb = np.asarray(b["center"], dtype=float)
        c_pair = np.asarray(rlp["center"], dtype=float)
        
        # Connection line between matching loop centers
        conn_line = vedo.Line(ca, cb).c("orange").lw(3)
        
        # -------------------------------------------------------------
        # Window 1: Linear Normal RLP (Normal vectors -> Orange)
        # -------------------------------------------------------------
        lin_actors = [*base_part_actors, conn_line]
        
        lin_A = a.get('linear', {})
        if lin_A.get('plane') and len(lin_A['plane']) > 0:
            nA = np.asarray(lin_A['plane'][0][0], dtype=float)
        else:
            nA = np.asarray(a.get('pca_normal', [0,0,1]), dtype=float)
        nA = normalize(nA)
        arrow_a = vedo.Arrow(ca, ca + arrow_len * nA, c="orange", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
        lin_actors.append(arrow_a)
        
        lin_B = b.get('linear', {})
        if lin_B.get('plane') and len(lin_B['plane']) > 0:
            nB = np.asarray(lin_B['plane'][0][0], dtype=float)
        else:
            nB = np.asarray(b.get('pca_normal', [0,0,1]), dtype=float)
        nB = normalize(nB)
        if np.dot(nA, nB) < 0.0:
            nB = -nB
        arrow_b = vedo.Arrow(cb, cb + arrow_len * nB, c="orange", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
        lin_actors.append(arrow_b)
        
        n_pair = normalize(rlp["normal"])
        arrow_rlp = vedo.Arrow(c_pair, c_pair + arrow_len * n_pair, c="orange", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
        sphere_rlp = vedo.Sphere(pos=c_pair, r=radius_sphere*1.2).c("orange")
        lin_actors.extend([arrow_rlp, sphere_rlp])
        
        print(f"[info] RLP #{idx_r} (parts {i} ↔ {j}) - Window 1: Linear Normal Match")
        print("  \033[38;2;255;127;14m[Normal Vector - Orange]\033[0m Linear Normal vectors for Part " + str(i) + ", Part " + str(j) + ", and Merged RLP Normal")

        plt1 = vedo.Plotter(title=f"RLP #{idx_r} (Linear Normal Match): parts {i} ↔ {j}", axes=0, size=(1800, 1800))
        plt1.show(lin_actors, interactive=True).close()
        
        # -------------------------------------------------------------
        # Window 2: Polyhedral Intersection RLP (Intersection vectors -> DodgerBlue)
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
                        
        if not lines_A and not lines_B:
            continue
            
        poly_actors = [*base_part_actors, conn_line]
        
        for idx_a, la in enumerate(lines_A):
            arrow_la = vedo.Arrow(ca, ca + arrow_len * la, c="dodgerblue", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
            poly_actors.append(arrow_la)
            
        for idx_b, lb in enumerate(lines_B):
            arrow_lb = vedo.Arrow(cb, cb + arrow_len * lb, c="dodgerblue", shaft_radius=shaft_rad, head_radius=head_rad, head_length=head_len)
            poly_actors.append(arrow_lb)
            
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
            
        for idx_a, la in enumerate(lines_A):
            if idx_a not in matched_A:
                merged_lines.append(la)
        for idx_b, lb in enumerate(lines_B):
            if idx_b not in matched_B:
                merged_lines.append(lb)
                
        for idx_m, line in enumerate(merged_lines):
            arrow_m = vedo.Arrow(c_pair, c_pair + arrow_len * line, c="dodgerblue", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
            poly_actors.append(arrow_m)
            
        print(f"[info] RLP #{idx_r} (parts {i} ↔ {j}) - Window 2: Polyhedral Intersection Match")
        print("  \033[38;2;30;144;255m[Intersection Vector - Blue]\033[0m Polyhedral Intersection Lines A, B & Merged Lines")

        plt2 = vedo.Plotter(title=f"RLP #{idx_r} (Polyhedral Intersection Match): parts {i} ↔ {j}", axes=0, size=(1800, 1800))
        plt2.show(poly_actors, interactive=True).close()
        
        # -------------------------------------------------------------
        # Window 3: 3rd Perpendicular Coordinate Frame
        # -------------------------------------------------------------
        frame_actors = [*base_part_actors]
        
        arrow_norm = vedo.Arrow(c_pair, c_pair + arrow_len * n_pair, c="orange", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
        frame_actors.append(arrow_norm)
        
        has_frame = False
        if len(merged_lines) > 0:
            line = merged_lines[0]
            arrow_m = vedo.Arrow(c_pair, c_pair + arrow_len * line, c="dodgerblue", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
            frame_actors.append(arrow_m)
            
            n_third = np.cross(n_pair, line)
            if np.linalg.norm(n_third) > 1e-6:
                n_third = normalize(n_third)
                arrow_t = vedo.Arrow(c_pair, c_pair + arrow_len * n_third, c="mediumseagreen", shaft_radius=shaft_rad*1.4, head_radius=head_rad*1.3, head_length=head_len*1.2)
                frame_actors.append(arrow_t)
                has_frame = True
                
        if has_frame:
            print(f"[info] RLP #{idx_r} (parts {i} ↔ {j}) - Window 3: 3rd Perpendicular Coordinate Frame")
            print("  1. \033[38;2;255;127;14m[Normal Vector - Orange]\033[0m Main Plane Normal")
            print("  2. \033[38;2;30;144;255m[Intersection Vector - Blue]\033[0m Intersection Line")
            print("  3. \033[38;2;46;139;87m[Cross Product Vector - Green]\033[0m Perpendicular Axis (Cross Product)")

            plt3 = vedo.Plotter(title=f"RLP #{idx_r} (3rd Perpendicular Coordinate Frame): parts {i} ↔ {j}", axes=0, size=(1800, 1800))
            plt3.show(frame_actors, interactive=True).close()


def visualize_all_boundary_loops(
    verts: np.ndarray,
    faces: np.ndarray,
    loops: list,
    loop_neighbors: list,
    parent_part: int,
    vert_label: np.ndarray = None,
    title: str = "All Boundary Loops",
    display: bool = True,
    show_none_loops: bool = False,
):
    """
    Visualize all boundary loops for a part with pastel colors and labels indicating neighbors.
    """
    if not display:
        return

    actors = []
    num_pastel = len(PASTEL_RGB_UINT8)
    parent_color_float = PASTEL_RGB_FLOAT[(parent_part - 1) % num_pastel]

    # 1) Base mesh coloring (Current part: sequence pastel color, Other parts: light gray)
    if vert_label is not None:
        face_labels = vert_label[faces[:, 0]]
        is_parent_face = (face_labels == parent_part)

        # Other parts mesh (light gray, high transparency)
        other_faces = faces[~is_parent_face]
        if len(other_faces) > 0:
            mesh_base = vedo.Mesh([verts, other_faces]).c("lightgray").alpha(0.15)
            actors.append(mesh_base)

        # Highlight parent part mesh with sequence pastel color (higher transparency: alpha=0.35)
        if np.any(is_parent_face):
            parent_faces = faces[is_parent_face]
            mesh_parent = vedo.Mesh([verts, parent_faces]).c(parent_color_float).alpha(0.35)
            actors.append(mesh_parent)
    else:
        mesh = vedo.Mesh([verts, faces]).c(parent_color_float).alpha(0.35)
        actors.append(mesh)

    # Calculate scene diagonal for scaling text sizes
    scene_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    sphere_r = scene_diag * 0.005 if np.isfinite(scene_diag) and scene_diag > 0 else 0.005

    for i, loop in enumerate(loops):
        neighbors = loop_neighbors[i]
        
        # Optionally skip showing loops with no neighbors (connecting to None/N)
        if not neighbors and not show_none_loops:
            continue

        # Target part for color: neighbor part if available, else parent_part
        target_part = neighbors[0] if (neighbors and isinstance(neighbors[0], int)) else parent_part
        color_rgb = tuple(PASTEL_RGB_UINT8[(target_part - 1) % num_pastel])
        color_vedo = '#%02x%02x%02x' % color_rgb
        loop_idx = np.asarray(loop, dtype=int)
        
        # Display loop vertices as dots using sequence pastel color
        loop_pts = vedo.Spheres(verts[loop_idx], r=sphere_r * 1.2, c=color_vedo, res=8).alpha(1.0)
        actors.append(loop_pts)
        
        # Print information to the terminal with a colored square block (ANSI TrueColor)
        neighbors_str = ",".join(map(str, neighbors)) if neighbors else "N"
        r, g, b = color_rgb
        color_block = f"\033[38;2;{r};{g};{b}m■\033[0m"
        print(f"[info] Cycle {i+1} ({color_block}): Part {parent_part} -> {neighbors_str}")

    # Show plot
    plt = vedo.Plotter(bg="white", title=title, size=(1800, 1800))
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

    # Category 0: background / unassigned (light gray)
    # Category 1..N: Pastel color sequence in fixed order
    bg_color = np.array([220, 220, 220], dtype=np.uint8)
    num_pastel = len(PASTEL_RGB_UINT8)

    rgb_colors = np.zeros((len(vert_label), 3), dtype=np.uint8)
    for i, l in enumerate(vert_label):
        l_int = int(l)
        if l_int <= 0:
            rgb_colors[i] = bg_color
        else:
            rgb_colors[i] = PASTEL_RGB_UINT8[(l_int - 1) % num_pastel]

    mesh = vedo.Mesh([verts, faces])
    mesh.pointcolors = rgb_colors
    mesh.alpha(0.4)  # High transparency as requested

    plt = vedo.Plotter(bg="white", title=title, size=(1800, 1800))
    plt.show([mesh], viewup="z").close()
