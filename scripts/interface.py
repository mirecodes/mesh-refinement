import os
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
    json_config_path = "configs.json"

    # Operational controller
    enable_stage_align: bool = False
    restore_align: bool = True
    enable_stage_refine: bool = False
    enable_stage_decompose: bool = False
    enable_stage_segment: bool = True
    enable_stage_articulate: bool = True
    enable_stage_bind: bool = True
    debug_mode: bool = True

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

    root_dir: str = field(default_factory=os.getcwd, init=False)
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


def execute_pipeline(cfgs: SystemConfigs):
    print("[info]: Execute the object importing pipeline")

    if cfgs.enable_stage_align:
        if cfgs.restore_align:
            restore_transform(cfgs)
            print("[info]: Restoring the alignment")
        else:
            stage_transform(cfgs)
            print("[info]: Running the alignment stage")

    if cfgs.enable_stage_refine:
        print("[info]: Running the refinement stage")
        stage_refine(cfgs, debug_mode=cfgs.debug_mode)

    if cfgs.enable_stage_decompose:
        print("[info]: Running the decomposition stage")
        stage_decompose(cfgs)

    if cfgs.enable_stage_segment:
        print("[info]: Running the segmentation stage")
        stage_segment(cfgs)

    if cfgs.enable_stage_articulate:
        print("[info]: Running the articulation stage")
        stage_articulate(cfgs)

    if cfgs.enable_stage_bind:
        print("[info]: Running the binding stage")
        stage_bind(cfgs)



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
