import os
import numpy as np
import pymeshlab
import vedo
from plyfile import PlyData
from json_handler import JsonHandler
from functools import partial

from functions import calculate_transforms
from functions.util_transform import apply_Rt_points, save_mesh_with_transform, save_gaussian_with_transform


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

    # Bind external callback with partial
    handler = partial(key_press_callback, assembly=assembly, plt=plt, angle_step=1.0)
    plt.add_callback('KeyPress', handler)

    plt.add(assembly, *axes)
    plt.show(interactive=True)


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