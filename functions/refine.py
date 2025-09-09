from dataclasses import dataclass, field
import os
import pymeshlab
import trimesh
from pymeshlab import PercentageValue
from json_handler import JsonHandler

from functions import repair_mesh, flatten_bottom_hole, fill_bottom_hole


@dataclass
class RefinementConfig:
    """
    Centralizes all ms.* call parameters.
    """
    params: dict = field(default_factory=lambda: {
        # 1) cleaning / components
        "meshing_remove_connected_component_by_diameter": {
            "mincomponentdiag": PercentageValue(20.0),
        },
        # 2) reconstruction
        "generate_surface_reconstruction_screened_poisson": {
            "depth": 10,
            "fulldepth": 8,
            # 'cgdepth': 0,
            # 'scale': 1.1,
            # 'samplespernode': 1.5,
            # 'pointweight': 4,
            # 'iters': 8,
            # 'confidence': False,
            # 'preclean': True,
            # 'threads': 16,
        },
        # 3) vertex selection (cut)
        "compute_selection_by_condition_per_vertex": {
            "condselect": "(z < -1e-6)",
        },
        # 4) decimation
        "meshing_decimation_quadric_edge_collapse": {
            "targetfacenum": 20000,
            "preservenormal": False,
        },
        # 5) remeshing
        "meshing_isotropic_explicit_remeshing": {
            "targetlen": PercentageValue(1.0),
            # 'iterations': 10,
            # 'adaptive': False,
            # 'selectedonly': False,
            # 'featuredeg': 30.0,
            # 'checksurfdist': True,
            # 'maxsurfdist': PercentageValue(1.0),
            # 'splitflag': True,
            # 'collapseflag': True,
            # 'swapflag': True,
            # 'smoothflag': True,
            # 'reprojectflag': True,
        },
        # 6) close holes
        "meshing_close_holes": {
            "maxholesize": 10000,
        },
        # 7) attribute transfer
        "transfer_attributes_per_vertex": {
            "sourcemesh": 1,
            "targetmesh": 0,
            "vertexsampling": False,
            "geomtransfer": False,
            "normaltransfer": True,
            "colortransfer": True,
            "upperbound": PercentageValue(5.0),
        },
    })


def stage_refine(cfgs):
    # Load configurations
    refCfgs = RefinementConfig()
    states = JsonHandler(cfgs.json_states_dir, auto_save=False)

    # Load meshes
    ms = pymeshlab.MeshSet()
    mesh_in_dir = states.transform.dirs.mesh
    ms.load_new_mesh(mesh_in_dir)

    # --------------------------------------------------------------------------
    # Mesh Refinement Process
    # --------------------------------------------------------------------------

    # 1) Cleaning: remove small connected components
    ms.meshing_remove_connected_component_by_diameter(
        **refCfgs.params["meshing_remove_connected_component_by_diameter"]
    )

    # 2) Reconstruction: Screened Poisson
    ms.generate_surface_reconstruction_screened_poisson(
        **refCfgs.params["generate_surface_reconstruction_screened_poisson"]
    )

    # 3) Vertex selection (cut-off)
    ms.compute_selection_by_condition_per_vertex(
        **refCfgs.params["compute_selection_by_condition_per_vertex"]
    )
    if ms.current_mesh().selected_vertex_number() > 0:
        ms.meshing_remove_selected_vertices()

    # 4) Decimation (edge collapse)
    ms.meshing_decimation_quadric_edge_collapse(
        **refCfgs.params["meshing_decimation_quadric_edge_collapse"]
    )

    # 5) Remeshing (isotropic explicit)
    ms.meshing_isotropic_explicit_remeshing(
        **refCfgs.params["meshing_isotropic_explicit_remeshing"]
    )

    # AUX: recovery
    repair_mesh(ms)

    # Infill bottom occlusion
    ms = flatten_bottom_hole(ms, target_z=0.0)
    ms = fill_bottom_hole(ms)

    # AUX: recovery
    repair_mesh(ms)

    # 6) Close holes
    ms.meshing_close_holes(
        **refCfgs.params["meshing_close_holes"]
    )

    # AUX: recovery
    repair_mesh(ms)

    # 7) Restore vertex colors (attribute transfer)
    ms.load_new_mesh(mesh_in_dir)
    ms.set_current_mesh(0)

    ms.transfer_attributes_per_vertex(
        **refCfgs.params["transfer_attributes_per_vertex"]
    )

    ms.set_current_mesh(1)
    ms.delete_current_mesh()
    ms.set_current_mesh(0)

    # --------------------------------------------------------------------------
    # Save the Refined Mesh
    # --------------------------------------------------------------------------
    mesh_out_dir = os.path.join(cfgs.mesh_working_dir, f"refined_mesh.{cfgs.extension}")
    ms.save_current_mesh(mesh_out_dir)

    # Save intermediate state into JSON
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    states.refine = {}
    states.refine.dirs = {"mesh": mesh_out_dir}

    try:
        ms.clear()
    except Exception:
        pass

    return