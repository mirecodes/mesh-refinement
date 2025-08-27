from dataclasses import dataclass, field
import os
import pymeshlab
import trimesh
from json_handler import JsonHandler
from pymeshlab import PercentageValue

from functions import repair_mesh, flatten_bottom_hole, fill_bottom_hole


@dataclass
class RefinementConfig:
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

def stage_refine(cfgs, ms: pymeshlab.MeshSet) -> pymeshlab.MeshSet:

    refCfgs = RefinementConfig()

    # --------------------------------------------------------------------------
    # Mesh Refinement Process
    # --------------------------------------------------------------------------

    # Initial cleaning
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(20.0))

    # Reconstruction: Screened Poisson Surface Reconstruction
    ms.generate_surface_reconstruction_screened_poisson(**refCfgs.screened_poisson_params)

    # Cut vertices
    ms.compute_selection_by_condition_per_vertex(**refCfgs.cutoff_params)
    if ms.current_mesh().selected_vertex_number() > 0:
        ms.meshing_remove_selected_vertices()

    # Mesh simplification
    ms.meshing_decimation_quadric_edge_collapse(**refCfgs.edge_collapse_params)
    ms.meshing_isotropic_explicit_remeshing(**refCfgs.remshing_params)

    # AUX: mesh recovery
    repair_mesh(ms)

    # Infill bottom occlusion
    ms = flatten_bottom_hole(ms, target_z=0.0)
    ms = fill_bottom_hole(ms)

    # AUX: mesh recovery
    repair_mesh(ms)

    # Infill small holes
    ms.meshing_close_holes(maxholesize=5000)

    # AUX: mesh recovery
    repair_mesh(ms)

    # Restore the vertex colors
    try:
        ms.load_new_mesh(cfgs.mesh_in_dir)
        ms.set_current_mesh(0)
    except pymeshlab.PyMeshLabException:
        print(f"[error]: Failed to find mesh in directory. dir={cfgs.in_dir}")
        raise FileNotFoundError()

    ms.transfer_attributes_per_vertex(
        sourcemesh=2,
        targetmesh=0,
        vertexsampling=False,
        geomtransfer=False,
        normaltransfer=False,
        colortransfer=True,
        upperbound=PercentageValue(5.0),
    )

    ms.set_current_mesh(2)
    ms.delete_current_mesh()
    ms.set_current_mesh(0)

    # --------------------------------------------------------------------------
    # Save the Refined Mesh
    # --------------------------------------------------------------------------

    mesh_dir = os.path.join(cfgs.mesh_working_dir, f"refined_mesh.{cfgs.extension}")

    ms.set_current_mesh(0)
    ms.save_current_mesh(mesh_dir)

    # Save intermediate state into the json file
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)
    states.refine = {}
    states.refine.dirs = {
        "mesh": mesh_dir,
    }

    return ms