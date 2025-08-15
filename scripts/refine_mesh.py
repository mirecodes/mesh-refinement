from dataclasses import dataclass, field
import os
import pymeshlab
import trimesh
from pymeshlab import PercentageValue

from functions import *


@dataclass
class SystemConfig:
    # --------------------------------------------------------------------------
    # 1. User Define Configurations
    # --------------------------------------------------------------------------
    # mesh data directory: input / output
    mesh_in_path: str = "data/meshes/bonsai_pot.ply"
    mesh_out_path: str = "out/meshes/refined_bonsai_pot.ply"
    mesh_dev_path: str = "out/meshes/processed_bonsai_pot.ply"

    # function toggle flags
    automatic_alignment: bool = True  # True: 시작 시 PCA로 바닥면을 자동 정렬
    step_by_step_visualization: bool = True  # True: 각 단계별로 결과물을 시각화

    # --------------------------------------------------------------------------
    # 2. Automatically Generated Configurations
    # --------------------------------------------------------------------------
    root_dir: str = field(default_factory=os.getcwd, init=False)
    mesh_in_dir: str = field(init=False)
    mesh_out_dir: str = field(init=False)
    mesh_dev_dir: str = field(init=False)

    def __post_init__(self):
        self.mesh_in_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_in_path))
        self.mesh_out_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_out_path))
        self.mesh_dev_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_dev_path))

@dataclass
class ParametersConfig:
    # Detailed parameters information
    # https://pymeshlab.readthedocs.io/en/latest/filter_list.html

    screened_poisson_params = {
        # 'visiblelayer': False,
        'depth': 10,
        'fulldepth': 8,
        # 'cgdepth': 0,
        # 'scale': 1.1,
        # 'samplespernode': 1.5,
        # 'pointweight': 4,
        # 'iters': 8,
        # 'confidence': False,
        # 'preclean': True,
        # 'threads': 16,
    }
    cutoff_params = {
        'condselect': '(z < -1e-6)',
    }

    edge_collapse_params = {
        'targetfacenum': 20000,
        # 'targetperc': 0.0,
        # 'qualitythr': 0.3,
        # 'preserveboundary': False,
        # 'boundaryweight': 1.0,
        'preservenormal': False,
        # 'preservetopology': False,
        # 'optimalplacement': True,
        # 'planarquadric': True,
        # 'planarweight': 0.001,
        # 'qualityweight': False,
        # 'autoclean': True,
        # 'selected': False,
    }

    remshing_params = {
        # 'iterations': 10,
        # 'adaptive': False,
        # 'selectedonly': False,
        'targetlen': pymeshlab.PercentageValue(1.0), # percentage value
        # 'featuredeg': 30.0,
        # 'checksurfdist': True,
        # 'maxsurfdist': pymeshlab.PercentageValue(1.0), # percentage value
        # 'splitflag': True,
        # 'collapseflag': True,
        # 'swapflag': True,
        # 'smoothflag': True,
        # 'reprojectflag': True,
    }



def run_refinement_pipeline(ms, sys_cfg, params_cfg):
    """
    throughout mesh refinement pipeline
    :param ms: mesh_set
    :param sys_cfg: cfg
    :param forward_transform:
    :param inverse_transform:
    :return:
    """

    print("[info]: Start the mesh refinement pipeline.")

    # --------------------------------------------------------------------------
    # Initial Cleaning
    # --------------------------------------------------------------------------

    # step: clean mesh by removing isolated fragments
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(20.0))
    show_mesh(ms, title="remove isolated fragments", enabled=True, highlight_boundary_loops=True)
    # show_mesh_with_cutting_plane(ms, title="remove isolated fragmented", cut_z_value=0, enabled=True, highlight_boundary_loops=True)

    # --------------------------------------------------------------------------
    # Alignment
    # --------------------------------------------------------------------------

    # step: calculate transformation (align z-axis)
    if sys_cfg.automatic_alignment:
        forward_transform = calculate_transforms(ms)
    else:
        forward_transform = np.eye(4), np.eye(4)

    # step: apply forward transformation
    apply_transform_from_matrix(ms, forward_transform)
    show_mesh_with_cutting_plane(ms, title="transform z-up", cut_z_value=0, enabled=True, highlight_boundary_loops=True)

    # --------------------------------------------------------------------------
    # Surface reconstruction
    # --------------------------------------------------------------------------

    # step: Screened Poisson surface reconstruction
    ms.generate_surface_reconstruction_screened_poisson(**params_cfg.screened_poisson_params)
    show_mesh(ms, title="Poisson surface reconstruction", enabled=True, highlight_boundary_loops=True)

    # step: cut the vertices below the bottom plane
    ms.compute_selection_by_condition_per_vertex(**params_cfg.cutoff_params)
    if ms.current_mesh().selected_vertex_number() > 0:
        ms.meshing_remove_selected_vertices()
    show_mesh_with_cutting_plane(ms, title="transform z-up", cut_z_value=0, enabled=True, highlight_boundary_loops=True)

    # step: mesh simplification
    ms.meshing_decimation_quadric_edge_collapse(**params_cfg.edge_collapse_params)
    show_mesh(ms, title="Mesh simplication", enabled=True, highlight_boundary_loops=True)

    # step: remeshing
    ms.meshing_isotropic_explicit_remeshing(**params_cfg.remshing_params)

    # auxiliary step: intermediate mesh recovery
    repair_mesh(ms)

    # --------------------------------------------------------------------------
    # Infill the bottom occlusion
    # --------------------------------------------------------------------------

    # step: flatten bottom boundary loop
    ms = flatten_bottom_hole(ms, target_z=0.0)
    show_mesh(ms, title="Flatten boundary loop onto projected plane", enabled=True, highlight_boundary_loops=True)

    # step: delaunay triangulization
    ms = fill_bottom_hole(ms)
    show_mesh(ms, title="Bottom occlusion infill", enabled=True, highlight_boundary_loops=True)

    # step: intermediate mesh recovery
    repair_mesh(ms)

    # --------------------------------------------------------------------------
    # Infill the small sized holes
    # --------------------------------------------------------------------------

    # step: closing small holes
    ms.meshing_close_holes(maxholesize=5000)
    show_mesh(ms, title="Close other small holes", enabled=True)

    # step: final mesh recovery
    repair_mesh(ms)

    # --------------------------------------------------------------------------
    # Restore Transformation
    # --------------------------------------------------------------------------

    print("\n[단계 11] 최종 메쉬를 원래 좌표계로 복원합니다...")
    apply_inv_transform_from_matrix(ms, forward_transform)
    show_mesh(ms, title="After automated mesh refinement", enabled=sys_cfg.step_by_step_visualization)

    # Load new mesh
    print("[info]: Creating a copy of the original mesh to preserve vertex colors.")
    ms.load_new_mesh(sys_cfg.mesh_in_dir)

    print("\n[info]: Transferring vertex colors from the original mesh to the refined mesh.")
    ms.transfer_attributes_per_vertex(
        sourcemesh=1,
        targetmesh=0,
        vertexsampling=False,
        geomtransfer=False,
        normaltransfer=False,
        colortransfer=True,
        upperbound=PercentageValue(5.0),
    )
    show_mesh(ms, title="After transferring colors", enabled=sys_cfg.step_by_step_visualization)

    print("[info]: Setting the refined mesh (index 0) as the current mesh.")
    ms.set_current_mesh(0)

    #TODO: Differentiate the mesh layers

    # --------------------------------------------------------------------------
    # Save the Refined Mesh
    # --------------------------------------------------------------------------

    print(f"\n최종 메쉬를 '{sys_cfg.mesh_out_dir}'에 저장 중...")
    os.makedirs(os.path.dirname(sys_cfg.mesh_out_dir), exist_ok=True)
    ms.save_current_mesh(sys_cfg.mesh_out_dir)

    print("[info]: Mesh refinement complete.")


if __name__ == "__main__":
    sys_cfg = SystemConfig()
    params_cfg = ParametersConfig()
    ms = pymeshlab.MeshSet()

    try:
        ms.load_new_mesh(sys_cfg.mesh_in_dir)
    except pymeshlab.PyMeshLabException:
        print(f"[error]: Cannot find mesh '{sys_cfg.mesh_in_dir}'.")
        raise FileNotFoundError()

    if ms.mesh_number() > 0:
        run_refinement_pipeline(ms, sys_cfg, params_cfg)
    else:
        print("[error]: Cannot load the proper mesh.")