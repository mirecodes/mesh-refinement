import os
from dataclasses import dataclass, field

import coacd
import trimesh

from json_handler import JsonHandler

@dataclass
class SystemConfig:
    # --------------------------------------------------------------------------
    # 1. User Define Configurations
    # --------------------------------------------------------------------------
    # mesh data directory: input / output
    mesh_in_path: str = "work/spray/meshes/refined_mesh.ply"
    mesh_out_path: str = "work/spray/meshes/decomposed_mesh.ply"

    json_config_path: str = "work/spray/attributes/config.json"
    json_states_path: str = "work/spray/attributes/states.json"

    # --------------------------------------------------------------------------
    # 2. Automatically Generated Configurations
    # --------------------------------------------------------------------------
    root_dir: str = field(default_factory=os.getcwd, init=False)
    mesh_in_dir: str = field(init=False)
    mesh_out_dir: str = field(init=False)

    def __post_init__(self):
        self.mesh_in_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_in_path))
        self.mesh_out_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_out_path))
        self.json_config_path = os.path.abspath(os.path.join(self.root_dir, "..", self.json_config_path))
        self.json_states_path = os.path.abspath(os.path.join(self.root_dir, "..", self.json_states_path))

if __name__ == "__main__":
    cfg = SystemConfig()

    # Load mesh
    mesh = trimesh.load(cfg.mesh_in_dir, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    # Run convex decomposition
    parts = coacd.run_coacd(mesh)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(cfg.mesh_out_dir), exist_ok=True)

    # Save each part with index suffix
    base, ext = os.path.splitext(cfg.mesh_out_dir)
    for i, part in enumerate(parts, start=1):
        tri = trimesh.Trimesh(vertices=part[0], faces=part[1], process=False)
        out_path = f"{base}_{i}{ext}"
        tri.export(out_path)
        print(f"Saved: {out_path}")

    # Save intermediate state into the json file
    states = JsonHandler(
        filename=cfg.json_states_path,
        auto_save=False,
    )

    states.decompose = {}
    states.decompose.length = len(parts)
    states.decompose.file_names = [f"{base}_{i}{ext}" for i, part in enumerate(parts, start=1)]
    states.save()
