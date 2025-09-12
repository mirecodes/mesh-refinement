import os
import numpy as np
import plyfile
import pymeshlab
import trimesh
import vedo
from plyfile import PlyData, PlyElement
from json_handler import JsonHandler
from functools import partial

from functions import calculate_transforms

def apply_Rt_points(P: np.ndarray, Rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return (P @ Rmat.T) + tvec

def apply_R_normals(N: np.ndarray, Rmat: np.ndarray) -> np.ndarray:
    N2 = (N @ Rmat.T)
    N2 /= (np.linalg.norm(N2, axis=1, keepdims=True) + 1e-12)
    return N2

# -------- Small helper: robust shift for assembly (supports older vedo) -----
def _shift_assembly(assembly, dx=0.0, dy=0.0, dz=0.0):
    """Translate the whole assembly by (dx,dy,dz) in world frame."""
    try:
        assembly.shift(dx, dy, dz)  # modern vedo
        return
    except Exception:
        pass
    try:
        # Fallback: shift all parts
        for a in getattr(assembly, "actors", []):
            try:
                a.shift(dx, dy, dz)
            except Exception:
                pass
    except Exception:
        # Last resort: pos() additive update if available
        try:
            x, y, z = assembly.pos()
            assembly.pos(x + dx, y + dy, z + dz)
        except Exception:
            pass

# -------- Save mesh with applied transform --------
def save_mesh_with_transform(ms_mesh: pymeshlab.Mesh, out_path: str, T4: np.ndarray):
    V = ms_mesh.vertex_matrix().astype(np.float64)
    F = ms_mesh.face_matrix().astype(np.int64)

    Rmat = T4[:3, :3].astype(np.float64)
    tvec = T4[:3, 3].astype(np.float64)

    V2 = apply_Rt_points(V, Rmat, tvec)

    # Keep normals/colors if available

    # normals
    vnorm = ms_mesh.vertex_normal_matrix()
    if vnorm is not None and len(vnorm) == len(V):
        vnorm = apply_R_normals(vnorm.astype(np.float64), Rmat)
    else:
        vnorm = None

    # colors
    vcols = ms_mesh.vertex_color_matrix() if ms_mesh.has_vertex_color() else None

    # build trimesh
    tm = trimesh.Trimesh(vertices=V2, faces=F, process=False)

    if vnorm is not None:
        tm.vertex_normals = vnorm
    if vcols is not None:
        rgb = np.clip(vcols[:, :3] * 255.0, 0, 255).astype(np.uint8)
        alpha = 255 * np.ones((rgb.shape[0], 1), np.uint8)
        tm.visual.vertex_colors = np.hstack([rgb, alpha])

    tm.export(out_path)


# -------- Apply transform to Gaussian PLY --------
def save_gaussian_with_transform(ply: plyfile.PlyData, gaussian_out_path: str, T4: np.ndarray):
    # ply = PlyData.read(gaussian_in_path)
    v = ply['vertex']
    names = v.data.dtype.names

    Rmat = T4[:3, :3].astype(np.float64)
    tvec = T4[:3, 3].astype(np.float64)

    if all(k in names for k in ('x','y','z')):
        P = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
        P2 = apply_Rt_points(P, Rmat, tvec)
        v['x'] = P2[:, 0].astype(v['x'].dtype)
        v['y'] = P2[:, 1].astype(v['y'].dtype)
        v['z'] = P2[:, 2].astype(v['z'].dtype)

    if all(k in names for k in ('nx','ny','nz')):
        N = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float64)
        N2 = apply_R_normals(N, Rmat)
        v['nx'] = N2[:, 0].astype(v['nx'].dtype)
        v['ny'] = N2[:, 1].astype(v['ny'].dtype)
        v['nz'] = N2[:, 2].astype(v['nz'].dtype)

    PlyData([PlyElement.describe(v.data, 'vertex')], text=False).write(gaussian_out_path)


# -------- Key input callback (with translation) -----------------------------
def key_press_callback(e, *, assembly=None, plt=None, angle_step: float = 1.0, trans_step: float = 1.0):
    """
    Supported keys:
      Rotation
        - Z-axis: U(-), I(+)
        - Y-axis: H(-), J(+)
        - X-axis: B(-), N(+)
      Translation
        - X-axis: W(+), S(-)
        - Y-axis: D(+), A(-)
        - Z-axis: Space(+), Shift(-)
      Exit: q, esc
    Other keys are ignored.
    """
    key_raw = getattr(e, "keypress", "") or ""
    k = key_raw.lower()

    # exit
    if k in ("q", "esc", "escape"):
        try:
            plt.close()
        except Exception:
            vedo.close()
        return

    # --- rotation ---
    if   k == "u": assembly.rotate_z(-angle_step)
    elif k == "i": assembly.rotate_z(+angle_step)
    elif k == "h": assembly.rotate_y(-angle_step)
    elif k == "j": assembly.rotate_y(+angle_step)
    elif k == "b": assembly.rotate_x(-angle_step)
    elif k == "n": assembly.rotate_x(+angle_step)

    # --- translation ---
    elif k == "w": _shift_assembly(assembly, +trans_step, 0.0, 0.0)  # +X
    elif k == "s": _shift_assembly(assembly, -trans_step, 0.0, 0.0)  # -X
    elif k == "d": _shift_assembly(assembly, 0.0, +trans_step, 0.0)  # +Y
    elif k == "a": _shift_assembly(assembly, 0.0, -trans_step, 0.0)  # -Y
    elif k in (" ", "space"): _shift_assembly(assembly, 0.0, 0.0, +trans_step)  # +Z
    elif k == "z": _shift_assembly(assembly, 0.0, 0.0, -trans_step)  # -Z
    else:
        return  # ignore other keys

    try:
        plt.render()
    except Exception:
        pass


# -------- Main stage (only the binding part changed) ------------------------
def stage_transform(cfgs):
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(cfgs.mesh_in_dir)
        ms.load_new_mesh(cfgs.gaussian_in_dir)
        ms.set_current_mesh(0)
    except pymeshlab.PyMeshLabException:
        print(f"[error]: Failed to find mesh. dir={cfgs.in_dir}")
        raise FileNotFoundError()

    mesh = ms.mesh(0)
    initial_transform = calculate_transforms(ms)  # 4x4

    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    R0 = initial_transform[:3, :3].astype(np.float64)
    t0 = initial_transform[:3, 3].astype(np.float64)

    verts_vis = apply_Rt_points(verts.astype(np.float64), R0, t0)
    vmesh = vedo.Mesh([verts_vis, faces], alpha=0.5)

    if mesh.has_vertex_color():
        vmesh.pointcolors = mesh.vertex_color_matrix()[:, :3] * 255
    elif mesh.has_face_color():
        vmesh.cellcolors = mesh.face_color_matrix()[:, :3] * 255
    else:
        vmesh.color("lightblue")

    assembly = vedo.Assembly((vmesh,))
    bounds = vmesh.bounds()
    max_len = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    axis_len = max_len * 0.8

    axes = [
        vedo.Line([0,0,0], [axis_len,0,0], c='red', lw=3),
        vedo.Line([0,0,0], [0,axis_len,0], c='green', lw=3),
        vedo.Line([0,0,0], [0,0,axis_len], c='blue', lw=3),
        vedo.Grid(pos=(0,0,0), s=(max_len*1.2, max_len*1.2), alpha=0.15)
    ]
    plt = vedo.Plotter(title="Mesh Transform")

    # Disable default VTK keybindings
    try:
        iren = plt.interactor
        for _ev in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
            iren.RemoveObservers(_ev)
    except Exception:
        pass

    # Choose sensible default translation step from model size
    trans_step = max_len * 0.02 if np.isfinite(max_len) and max_len > 0 else 1.0

    # Bind external callback with partial
    handler = partial(
        key_press_callback,
        assembly=assembly,
        plt=plt,
        angle_step=1.0,
        trans_step=trans_step,
    )
    plt.add_callback('KeyPress', handler)

    plt.add(assembly, *axes)
    plt.show(interactive=True)

    # accumulate and save
    fine_transform = assembly.transform.matrix
    total_T = fine_transform @ initial_transform

    # save mesh
    mesh_out = os.path.join(cfgs.mesh_working_dir, f"transformed_mesh.{cfgs.extension}")
    save_mesh_with_transform(mesh, mesh_out, total_T)

    # save gaussian
    gaussian_in = cfgs.gaussian_in_dir
    ply = PlyData.read(gaussian_in)
    gaussian_out = os.path.join(cfgs.mesh_working_dir, f"transformed_gaussian.{cfgs.extension}")
    save_gaussian_with_transform(ply, gaussian_out, total_T)

    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    states.transform = {}
    states.transform.matrix = total_T.tolist()
    states.transform.dirs = {
        "mesh": mesh_out,
        "gaussian": gaussian_out,
    }

    try:
        ms.clear()
        vedo.close()
    except Exception:
        pass

    return