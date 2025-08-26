import os
from dataclasses import dataclass, field

import numpy as np
import trimesh
import coacd
import vedo

from json_handler import JsonHandler

@dataclass
class SystemConfig:
    # --------------------------------------------------------------------------
    # 1. User Define Configurations
    # --------------------------------------------------------------------------
    # mesh data directory: input / output
    mesh_in_path: str = "data/refined/refined_spray.ply"
    mesh_out_path: str = "out/ui/articulated_spray.ply"

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

    # TODO: Remove temporary attirubtes
    parts_num = 2
    segment_category = [[] for _ in range(parts_num)]

    default_group = 0
    current_group = 1

    states = JsonHandler(cfg.json_states_path)
    parts = []
    for i in range(states.decompose.length):
        tri = trimesh.load(states.decompose.file_names[i])
        parts.append(tri)


    # Ensure output directory exists
    os.makedirs(os.path.dirname(cfg.mesh_out_dir), exist_ok=True)

    # Save each part with index suffix, and collect vedo Meshes
    base, ext = os.path.splitext(cfg.mesh_out_dir)
    vedo_meshes = []
    for i, part in enumerate(parts, start=1):
        tri = part.copy()
        # Convert to vedo Mesh (with random color)
        vmesh = vedo.Mesh([tri.vertices, tri.faces])
        rand_color = np.random.rand(3)  # RGB (0~1)
        vmesh.c(rand_color).alpha(0.4)  # 색상 + 투명도(0.6)
        vedo_meshes.append(vmesh)

    mesh = trimesh.load(cfg.mesh_in_dir, force="mesh")
    vedo_meshes.append(mesh)


    # Show all parts together
    # vedo.show(vedo_meshes, __doc__, axes=1, viewup="z")


    def on_pick(event):
        picked = event.actor
        if picked is None:
            return
        if picked in vedo_meshes:
            segment_category[current_group].append(picked)
            picked.c("green")  # 클릭하면 색을 임시로 바꿔줌
            plt.render()
            print(f"[INFO] Added mesh to {current_group}")


    def switch_group():
        """버튼 눌러서 저장 그룹 전환"""
        global current_group
        current_group = "part2" if current_group == "part1" else "part1"
        print(f"[INFO] Now selecting for {current_group}")


    def finalize_selection(*args):
        """선택한 파트를 다른 색으로 칠해 최종 렌더링"""
        for parts in segment_category:
            for part in parts:
                rand_color = np.random.rand(3)  # RGB (0~1)
                part.c(rand_color).alpha(0.4)  # 색상 + 투명도(0.6)

        plt.render()
        print("[INFO] Finalized: part1=red, part2=blue")


    # GUI 생성
    plt = vedo.Plotter(title="Click to select parts")
    plt.add_callback("mouse_click", on_pick)

    # 그룹 전환 버튼
    plt.add_button(switch_group, pos=(0.7, 0.05), states=["Switch Group"], size=25, c="black", bc="orange")

    # 최종 색칠 버튼
    plt.add_button(finalize_selection, pos=(0.85, 0.05), states=["Finalize"], size=25, c="black", bc="lightgreen")

    plt.show(vedo_meshes, axes=1)
