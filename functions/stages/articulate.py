import trimesh
import vedo
from typing import Dict, List

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

    # Generate URDF File
    urdf_dir = generate_urdf(
        closed_parts=links,
        joints=joints,
        out_dir=cfgs.urdf_out_dir,
        urdf_name="object",
        density=1000.0,   # 필요시 변경
        mesh_fmt="stl",
    )

    print(f"[info]: the urdf file has been generated in: {urdf_dir}")

    # Vedo viewer of the articulated object
    if cfgs.debug_mode:
        actors = [vedo.Mesh([m.vertices, m.faces]).c('lightblue').alpha(0.5) for _, m in sorted(links.items())]
        vedo.show(actors, "Loaded link meshes", axes=1).close()
