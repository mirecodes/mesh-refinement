import os
from dataclasses import dataclass, field

import numpy as np
import trimesh
import coacd
import vedo

@dataclass
class SystemConfig:
    # --------------------------------------------------------------------------
    # 1. User Define Configurations
    # --------------------------------------------------------------------------
    # mesh data directory: input / output
    mesh_in_path: str = "data/refined/refined_spray.ply"
    mesh_out_path: str = "out/ui/articulated_spray.ply"

    # --------------------------------------------------------------------------
    # 2. Automatically Generated Configurations
    # --------------------------------------------------------------------------
    root_dir: str = field(default_factory=os.getcwd, init=False)
    mesh_in_dir: str = field(init=False)
    mesh_out_dir: str = field(init=False)

    def __post_init__(self):
        self.mesh_in_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_in_path))
        self.mesh_out_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_out_path))

if __name__ == "__main__":
    cfg = SystemConfig()

    # Load mesh
    mesh = trimesh.load(cfg.mesh_in_dir, force="mesh")

    mesh = coacd.Mesh(mesh.vertices, mesh.faces)

    # Run convex decomposition
    parts = coacd.run_coacd(mesh)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(cfg.mesh_out_dir), exist_ok=True)

    # Save each part with index suffix, and collect vedo Meshes
    base, ext = os.path.splitext(cfg.mesh_out_dir)
    vedo_meshes = []
    for i, part in enumerate(parts, start=1):
        tri = trimesh.Trimesh(vertices=part[0], faces=part[1], process=False)
        out_path = f"{base}_{i}{ext}"
        tri.export(out_path)
        print(f"Saved: {out_path}")

        # Convert to vedo Mesh (with random color)
        vmesh = vedo.Mesh([tri.vertices, tri.faces])
        rand_color = np.random.rand(3)  # RGB (0~1)
        vmesh.c(rand_color).alpha(0.4)  # 색상 + 투명도(0.6)
        vedo_meshes.append(vmesh)
        vedo_meshes.append(vmesh)
    
    # Show all parts together
    vedo.show(vedo_meshes, __doc__, axes=1, viewup="z")