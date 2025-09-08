import os
import numpy as np
import pymeshlab
import vedo
import trimesh
from plyfile import PlyData, PlyElement
from json_handler import JsonHandler
from functools import partial

from functions import calculate_transforms


# -------- Utility: R@p + t / R@n --------
def _apply_Rt_points(P: np.ndarray, Rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return (P @ Rmat.T) + tvec

def _apply_R_normals(N: np.ndarray, Rmat: np.ndarray) -> np.ndarray:
    N2 = (N @ Rmat.T)
    N2 /= (np.linalg.norm(N2, axis=1, keepdims=True) + 1e-12)
    return N2


# -------- Save mesh with applied transform --------
def _save_mesh_with_matrix(ms_mesh: pymeshlab.Mesh, out_path: str, T4: np.ndarray):
    V = ms_mesh.vertex_matrix().astype(np.float64)
    F = ms_mesh.face_matrix().astype(np.int64)

    Rmat = T4[:3, :3].astype(np.float64)
    tvec = T4[:3, 3].astype(np.float64)

    V2 = _apply_Rt_points(V, Rmat, tvec)

    # Keep normals/colors if available
    try:
        vnorm = ms_mesh.vertex_normal_matrix()
        if vnorm is None or len(vnorm) != len(V):
            vnorm = None
    except Exception:
        vnorm = None
    if vnorm is not None:
        vnorm = _apply_R_normals(vnorm.astype(np.float64), Rmat)

    try:
        vcols = ms_mesh.vertex_color_matrix() if ms_mesh.has_vertex_color() else None
    except Exception:
        vcols = None
    tm = trimesh.Trimesh(vertices=V2, faces=F, process=False)

    if vnorm is not None:
        tm.vertex_normals = vnorm
    if vcols is not None:
        rgb = np.clip(vcols[:, :3] * 255.0, 0, 255).astype(np.uint8)
        tm.visual.vertex_colors = np.concatenate([rgb, 255*np.ones((rgb.shape[0],1), np.uint8)], axis=1)

    tm.export(out_path)


# -------- Apply transform to Gaussian PLY --------
def _apply_transform_to_gaussian(gaussian_in_path: str, gaussian_out_path: str, T4: np.ndarray):
    ply = PlyData.read(gaussian_in_path)
    v = ply['vertex']
    names = v.data.dtype.names

    Rmat = T4[:3, :3].astype(np.float64)
    tvec = T4[:3, 3].astype(np.float64)

    if all(k in names for k in ('x','y','z')):
        P = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
        P2 = _apply_Rt_points(P, Rmat, tvec)
        v['x'] = P2[:, 0].astype(v['x'].dtype)
        v['y'] = P2[:, 1].astype(v['y'].dtype)
        v['z'] = P2[:, 2].astype(v['z'].dtype)

    if all(k in names for k in ('nx','ny','nz')):
        N = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float64)
        N2 = _apply_R_normals(N, Rmat)
        v['nx'] = N2[:, 0].astype(v['nx'].dtype)
        v['ny'] = N2[:, 1].astype(v['ny'].dtype)
        v['nz'] = N2[:, 2].astype(v['nz'].dtype)

    PlyData([PlyElement.describe(v.data, 'vertex')], text=False).write(gaussian_out_path)


# -------- Key input callback --------
def key_press_callback(e, *, assembly=None, plt=None, angle_step: float = 1.0):
    """
    Supported keys:
      - Z-axis: E(-), R(+)
      - Y-axis: D(-), F(+)
      - X-axis: C(-), V(+)
      - Exit: q, esc
    Other keys ignored.
    """
    key_raw = getattr(e, "keypress", "") or ""
    k = key_raw.lower()

    if k in ("q", "esc", "escape"):
        try:
            plt.close()
        except Exception:
            vedo.close()
        return

    if k == "e":
        assembly.rotate_z(-angle_step)
    elif k == "r":
        assembly.rotate_z(+angle_step)
    elif k == "d":
        assembly.rotate_y(-angle_step)
    elif k == "f":
        assembly.rotate_y(+angle_step)
    elif k == "c":
        assembly.rotate_x(-angle_step)
    elif k == "v":
        assembly.rotate_x(+angle_step)
    else:
        return

    try:
        plt.render()
    except Exception:
        pass


# -------- Main stage --------
def stage_transform(cfgs) -> pymeshlab.MeshSet:
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
    verts_vis = _apply_Rt_points(verts.astype(np.float64), R0, t0)
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
        vedo.Text3D('X', pos=[axis_len,0,0], s=axis_len/10, c='red'),
        vedo.Text3D('Y', pos=[0,axis_len,0], s=axis_len/10, c='green'),
        vedo.Text3D('Z', pos=[0,0,axis_len], s=axis_len/10, c='blue'),
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

    # Bind external callback with partial
    handler = partial(key_press_callback, assembly=assembly, plt=plt, angle_step=1.0)
    plt.add_callback('KeyPress', handler)

    plt.add(assembly, *axes)
    plt.show(interactive=True)

    fine_transform = assembly.transform.matrix
    total_T = fine_transform @ initial_transform

    mesh_out = os.path.join(cfgs.mesh_working_dir, f"transformed_mesh.{cfgs.extension}")
    _save_mesh_with_matrix(mesh, mesh_out, total_T)

    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    gaussian_in = getattr(cfgs, 'gaussian_in_dir', None) \
                  or states.transform.dirs.get('gaussian_source', states.transform.dirs.get('gaussian'))
    gaussian_out = os.path.join(cfgs.mesh_working_dir, f"transformed_gaussian.{cfgs.extension}")
    _apply_transform_to_gaussian(gaussian_in, gaussian_out, total_T)

    states.transform = {}
    states.transform.matrix = total_T.tolist()
    states.transform.dirs = {
        "mesh": mesh_out,
        "gaussian": gaussian_out,
    }

    try:
        vedo.close()
    except Exception:
        pass
    return ms