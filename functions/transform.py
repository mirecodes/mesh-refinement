import os

import numpy as np
import pymeshlab
import vedo
from json_handler import JsonHandler

from functions import calculate_transforms, apply_transform_from_matrix


def stage_transform(cfgs, ms: pymeshlab.MeshSet) -> pymeshlab.MeshSet:

    # load mesh
    mesh = ms.mesh(0)

    # suggested transformation
    initial_transform = calculate_transforms(ms)
    apply_transform_from_matrix(ms, initial_transform)

    # prepare mesh visualization
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    vmesh = vedo.Mesh([verts, faces], alpha=0.5)

    vcolors = mesh.vertex_color_matrix()[:, :3] * 255 if mesh.has_vertex_color() else None
    fcolors = mesh.face_color_matrix()[:, :3] * 255 if mesh.has_face_color() else None

    if vcolors is not None:
        vmesh.pointcolors = vcolors
    elif fcolors is not None:
        vmesh.cellcolors = fcolors
    else:
        vmesh.color("lightblue")

    assembly = vedo.Assembly((vmesh,))

    bounds = vmesh.bounds()
    max_bound_length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    axis_length = max_bound_length * 0.8

    x_axis = vedo.Line([0, 0, 0], [axis_length, 0, 0], c='red', lw=3)
    y_axis = vedo.Line([0, 0, 0], [0, axis_length, 0], c='green', lw=3)
    z_axis = vedo.Line([0, 0, 0], [0, 0, axis_length], c='blue', lw=3)

    x_label = vedo.Text3D('X', pos=[axis_length, 0, 0], s=axis_length / 10, c='red')
    y_label = vedo.Text3D('Y', pos=[0, axis_length, 0], s=axis_length / 10, c='green')
    z_label = vedo.Text3D('Z', pos=[0, 0, axis_length], s=axis_length / 10, c='blue')

    plane = vedo.Grid(pos=(0, 0, 0), s=(max_bound_length * 1.2, max_bound_length * 1.2), alpha=0.15)

    plt = vedo.Plotter(title="Mesh Transform")

    def key_press_callback(event):
        if event.keypress == 'e':
            assembly.rotate_z(-1)
        elif event.keypress == 'r':
            assembly.rotate_z(1)
        elif event.keypress == '\t':
            pass

    plt.add_callback('KeyPress', key_press_callback)
    plt.add(assembly, x_axis, y_axis, z_axis, x_label, y_label, z_label, plane)
    plt.show(interactive=True)

    # apply the user transformation
    fine_transform = assembly.transform.matrix.T

    transform = fine_transform @ initial_transform
    apply_transform_from_matrix(ms, fine_transform)

    # save mesh
    mesh_dir = os.path.join(cfgs.mesh_working_dir, f"transformed_mesh.{cfgs.extension}")
    gaussian_dir = os.path.join(cfgs.mesh_working_dir, f"transformed_gaussian.{cfgs.extension}")

    ms.set_current_mesh(0); ms.save_current_mesh(mesh_dir)
    ms.set_current_mesh(1); ms.save_current_mesh(gaussian_dir)
    ms.set_current_mesh(0)

    # TODO: Remove current part later on
    # R = initial_transform[:3, :3] @ fine_transform[:3, :3]
    # T = initial_transform[:3, 3]
    # transform = np.eye(4)
    # transform[:3, :3] = R
    # transform[:3, 3] = T
    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh(cfgs.mesh_in_dir)
    # apply_transform_from_matrix(ms, transform)
    # ms.save_current_mesh(os.path.join(cfgs.mesh_working_dir, f"test_mesh.ply"))

    # Save intermediate state into the json file
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    states.transform = {}
    states.transform.matrix = transform.tolist()
    states.transform.dirs = {
        "mesh": mesh_dir,
        "gaussian": gaussian_dir,
    }

    return ms