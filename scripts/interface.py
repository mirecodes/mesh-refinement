import os
import sys
from pathlib import Path

# Add project root to python path to resolve local packages
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from dataclasses import dataclass, field

import pymeshlab
from json_handler import JsonHandler

from functions import stage_transform, stage_refine, stage_decompose, stage_segment, stage_articulate, stage_bind, \
    restore_transform


@dataclass
class SystemConfigs():
    # --------------------------------------------------------------------------
    # User Configurations
    # --------------------------------------------------------------------------

    # Configuration directories
    json_config_path = os.path.join(os.path.dirname(__file__), "configs.json")

    # Operational controller
    enable_stage_align: bool = True
    restore_align: bool = False
    enable_stage_refine: bool = True
    enable_stage_decompose: bool = True
    enable_stage_segment: bool = True
    enable_stage_articulate: bool = True
    enable_stage_bind: bool = True
    debug_mode: bool = False
    
    # URDF Generation Option
    # If True, the URDF will be generated in the original coordinate system (before transform).
    # If False, it will be in the transformed coordinate system.
    urdf_use_original_coordinates: bool = True

    # --------------------------------------------------------------------------
    # Automatically Generated Configurations
    # --------------------------------------------------------------------------

    json_cfgs = JsonHandler(json_config_path)

    # Parameters
    categories_num = json_cfgs.parts.num

    # Directories
    object_name = json_cfgs.object_name
    extension = json_cfgs.extension

    in_path = f"data/{object_name}/"
    out_path = f"out/{object_name}/"
    working_path = f"work/{object_name}/"

    fname_mesh = f"mesh.{extension}"
    fname_gaussian = f"gaussian.{extension}"

    root_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)), init=False)
    in_dir: str = field(init=False)
    out_dir: str = field(init=False)
    working_dir: str = field(init=False)

    def __post_init__(self):
        # Fundamental directories
        self.in_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.in_path))
        self.out_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.out_path))
        self.working_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.working_path))

        # Functional directories
        self.json_attr_dir = os.path.join(self.working_dir, "attributes")
        self.mesh_working_dir = os.path.join(self.working_dir, "meshes")
        self.json_configs_dir = os.path.join(self.json_attr_dir, "configs.json")
        self.json_states_dir = os.path.join(self.json_attr_dir, "states.json")

        self.urdf_working_dir = os.path.join(self.working_dir, "urdf")
        self.urdf_out_dir = os.path.join(self.out_dir, "urdf")
        self.gaussian_out_dir = os.path.join(self.out_dir, "gaussian")

        self.mesh_in_dir = os.path.join(self.in_dir, self.fname_mesh)
        self.gaussian_in_dir = os.path.join(self.in_dir, self.fname_gaussian)
        # self.mesh_out_dir = os.path.join(self.out_dir, self.fname_mesh)
        # self.gaussian_out_dir = os.path.join(self.out_dir, self.fname_gaussian)

        # Create directories if necessary
        self.create_directories()

    def create_directories(self):
        dirs_to_create = [
            self.out_dir,
            self.working_dir,
            self.json_attr_dir,
            self.mesh_working_dir,
            self.urdf_working_dir,
            self.urdf_out_dir,
            self.gaussian_out_dir,
        ]

        for path in dirs_to_create:
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                print(f"[error]: Failed to create directory {path}: {e}")


import time


def execute_pipeline(cfgs: SystemConfigs):
    print("[info]: Execute the object importing pipeline")
    
    import functions
    functions.total_gui_time = 0.0

    stages_time = {}
    pipeline_start = time.time()

    if cfgs.enable_stage_align:
        start_time = time.time()
        if cfgs.restore_align:
            restore_transform(cfgs)
            print("[info]: Restoring the alignment")
            stages_time["Alignment/Restore"] = time.time() - start_time
        else:
            stage_transform(cfgs)
            print("[info]: Running the alignment stage")
            stages_time["Alignment"] = time.time() - start_time

    if cfgs.enable_stage_refine:
        print("[info]: Running the refinement stage")
        start_time = time.time()
        stage_refine(cfgs, debug_mode=cfgs.debug_mode)
        stages_time["Refinement"] = time.time() - start_time

    if cfgs.enable_stage_decompose:
        print("[info]: Running the decomposition stage")
        start_time = time.time()
        stage_decompose(cfgs)
        stages_time["Decomposition"] = time.time() - start_time

    if cfgs.enable_stage_segment:
        print("[info]: Running the segmentation stage")
        start_time = time.time()
        stage_segment(cfgs)
        stages_time["Segmentation"] = time.time() - start_time

    if cfgs.enable_stage_articulate:
        print("[info]: Running the articulation stage")
        start_time = time.time()
        stage_articulate(cfgs)
        stages_time["Articulation"] = time.time() - start_time

    if cfgs.enable_stage_bind:
        print("[info]: Running the binding stage")
        start_time = time.time()
        stage_bind(cfgs)
        stages_time["Binding"] = time.time() - start_time

    total_time = time.time() - pipeline_start
    gui_time = functions.total_gui_time
    processing_time = max(0.0, total_time - gui_time)

    print("\n" + "=" * 40)
    print(f"{'Pipeline Execution Summary':^40}")
    print("=" * 40)
    for stage_name, duration in stages_time.items():
        print(f"- {stage_name} stage: {duration:.2f}s")
    print("-" * 40)
    print(f"- Time spent in User GUI: {gui_time:.2f}s")
    print(f"- Time spent in Processing (Waiting): {processing_time:.2f}s")
    print("-" * 40)
    print(f"Total elapsed time: {total_time:.2f}s")
    print("=" * 40 + "\n")



def load_from_saves(cfgs: SystemConfigs, stage: str):
    states = JsonHandler(cfgs.json_states_dir, auto_save=False)
    try:
        ms = pymeshlab.MeshSet()
        mesh_dir, gaussian_dir = cfgs.mesh_in_dir, cfgs.gaussian_in_dir

        if stage == "transform":
            mesh_dir = states.transform.dirs.mesh
            gaussian_dir = states.transform.dirs.gaussian
        elif stage == "refine":
            mesh_dir = states.refine.dirs.mesh
            gaussian_dir = states.transform.dirs.gaussian
        elif stage == "decompose":
            mesh_dir = states.refine.dirs.mesh
            gaussian_dir = states.transform.dirs.gaussian
        elif stage == "segment":
            mesh_dir = states.refine.dirs.mesh
            gaussian_dir = states.transform.dirs.gaussian

        ms.load_new_mesh(mesh_dir)
        ms.load_new_mesh(gaussian_dir)
        ms.set_current_mesh(0)
        return ms

    except pymeshlab.PyMeshLabException:
        print(f"[error]: Failed to find mesh in directory. dir={cfgs.in_dir}")


if __name__ == "__main__":
    execute_pipeline(cfgs=SystemConfigs())
