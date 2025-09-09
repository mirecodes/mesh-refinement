import os

import coacd
import pymeshlab
import trimesh

from json_handler import JsonHandler

def stage_decompose(cfgs):
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)

    # Load mesh data
    ms = pymeshlab.MeshSet()

    mesh_in_dir = states.refine.dirs.mesh
    ms.load_new_mesh(mesh_in_dir)
    mesh = ms.mesh(0)

    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    mesh = coacd.Mesh(verts, faces)

    # Run convex decomposition
    parts = coacd.run_coacd(mesh)

    # Save intermediate state into the json file
    states.decompose = {}
    states.decompose.length = len(parts)
    states.decompose.dirs = {
        "mesh": list()
    }

    # Save each part with index suffix
    os.makedirs(os.path.dirname(cfgs.mesh_working_dir), exist_ok=True)

    for i, part in enumerate(parts, start=1):
        tri = trimesh.Trimesh(vertices=part[0], faces=part[1], process=False)
        fname = f"decomposed_{i}.{cfgs.extension}"
        decompose_dir = os.path.join(cfgs.mesh_working_dir, fname)
        tri.export(decompose_dir)
        states.decompose.dirs.mesh.append(decompose_dir)

    try:
        ms.clear()
    except Exception:
        pass

    return