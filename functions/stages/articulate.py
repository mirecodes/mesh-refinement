import trimesh
import vedo
from typing import Dict, List
import numpy as np

from json_handler import JsonHandler
from functions.lib.urdf import generate_urdf

# ================================
# Stage: articulate (minimal)
# ================================
def stage_articulate(cfgs):
    # Load states
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)

    # Load meshes
    dir_mesh = states.segment.dirs.mesh
    links: Dict[int, trimesh.Trimesh] = {}
    for k, path in dir_mesh.items():
        lid = int(k)
        try:
            mesh = trimesh.load(path, process=False)
        except Exception as e:
            print(f"[error]: Failed to load mesh for link {lid}: {e}")
            continue
        links[lid] = mesh

    if not links:
        print("[error]: No link meshes found in states.segment.dirs.mesh")
        return

    # Load Joints
    joints: List[dict] = states.segment.joints
    
    # Determine transform matrix for URDF generation
    transform_matrix = None
    if cfgs.urdf_use_original_coordinates:
        try:
            # Load the transform matrix saved in stage_transform
            # This matrix T transforms original -> aligned.
            # To go back to original, we need T_inv.
            # generate_urdf expects the matrix that transforms current meshes/joints -> target frame.
            # Since current meshes are aligned, we need T_inv to go back to original.

            print(states.transform)

            # Check if transform state exists
            if hasattr(states, "transform") and hasattr(states.transform, "matrix"):
                T_aligned = np.array(states.transform.matrix)
                # We need the inverse to revert the alignment
                transform_matrix = np.linalg.inv(T_aligned)
                print("[info]: Generating URDF in ORIGINAL coordinate system (reverting alignment).")
            else:
                print("[warn]: Transform matrix not found in states. Generating URDF in current (aligned) coordinates.")
        except Exception as e:
            print(f"[warn]: Failed to load/invert transform matrix: {e}. Generating URDF in current coordinates.")
    else:
        print("[info]: Generating URDF in ALIGNED coordinate system.")

    # Generate URDF File
    urdf_dir = generate_urdf(
        closed_parts=links,
        joints=joints,
        out_dir=cfgs.urdf_out_dir,
        urdf_name="object",
        density=1000.0,   # 필요시 변경
        mesh_fmt="stl",
        transform_matrix=transform_matrix
    )

    print(f"[info]: the urdf file has been generated in: {urdf_dir}")

    # Vedo viewer of the articulated object
    if cfgs.debug_mode:
        actors = [vedo.Mesh([m.vertices, m.faces]).c('lightblue').alpha(0.5) for _, m in sorted(links.items())]
        vedo.show(actors, "Loaded link meshes", axes=1).close()
