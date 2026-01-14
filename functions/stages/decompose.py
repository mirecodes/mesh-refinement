import os
from dataclasses import dataclass, asdict

import coacd
import pymeshlab
import trimesh

from json_handler import JsonHandler

@dataclass
class CoACDParams:
    # Concavity threshold (0.01~1). Lower = more details/parts. Default 0.05.
    threshold: float = 0.05
    # Max # convex hulls. -1 = unlimited. Default -1.
    max_convex_hull: int = -1
    # Manifold preprocessing mode ('auto', 'on', 'off'). Default 'auto'.
    preprocess_mode: str = 'auto'
    # Resolution for manifold preprocess (20~100). Default 50.
    prep_resolution: int = 50
    # Sampling resolution for Hausdorff distance (1e3~1e4). Default 2000.
    resolution: int = 2000
    # Max child nodes in MCTS (10~40). Default 20.
    mcts_nodes: int = 20
    # MCTS search iterations (60~2000). Default 150.
    mcts_iterations: int = 150
    # Max search depth in MCTS (2~7). Default 3.
    mcts_max_depth: int = 3
    # Enable PCA pre-processing. Default False.
    pca: bool = False
    # Enable merge postprocessing. Default True (CLI --no-merge is False).
    merge: bool = True
    # Enable max vertex constraint per convex hull. Default False.
    decimate: bool = False
    # Max vertex count per hull if decimate is True. Default 256.
    max_ch_vertex: int = 256
    # Extrude neighboring convex hulls along overlapping faces. Default False.
    extrude: bool = False
    # Extrude margin if extrude is True. Default 0.01.
    extrude_margin: float = 0.01
    # Approximation shape ("ch" for convex hulls, "box" for cubes). Default "ch".
    approximate_mode: str = 'ch'
    # Random seed. Default 0 (or random).
    seed: int = 0


def stage_decompose(cfgs):
    states = JsonHandler(cfgs.json_states_dir, auto_save=True)

    # Load mesh data
    ms = pymeshlab.MeshSet()

    mesh_in_dir = states.refine.dirs.mesh
    ms.load_new_mesh(mesh_in_dir)
    mesh = ms.mesh(0)

    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    mesh = coacd.Mesh(verts, faces)

    # Initialize parameters
    # You can modify the default values in the CoACDParams class definition above
    # or override them via cfgs if needed.
    params = CoACDParams()
    
    # Optional: Override from cfgs if keys exist (e.g. cfgs.coacd_threshold)
    if hasattr(cfgs, "coacd_threshold"): params.threshold = cfgs.coacd_threshold
    if hasattr(cfgs, "coacd_resolution"): params.resolution = cfgs.coacd_resolution
    if hasattr(cfgs, "coacd_max_convex_hull"): params.max_convex_hull = cfgs.coacd_max_convex_hull

    print(f"[info] Running CoACD with parameters: {params}")

    # Run convex decomposition
    # Note: We pass parameters as kwargs. 
    # Ensure the installed coacd version supports these arguments.
    parts = coacd.run_coacd(
        mesh,
        threshold=params.threshold,
        max_convex_hull=params.max_convex_hull,
        preprocess_mode=params.preprocess_mode,
        preprocess_resolution=params.prep_resolution,
        resolution=params.resolution,
        mcts_nodes=params.mcts_nodes,
        mcts_iterations=params.mcts_iterations,
        mcts_max_depth=params.mcts_max_depth,
        pca=params.pca,
        merge=params.merge,
        decimate=params.decimate,
        max_ch_vertex=params.max_ch_vertex,
        extrude=params.extrude,
        extrude_margin=params.extrude_margin,
        apx_mode=params.approximate_mode,
        seed=params.seed
    )

    # Save intermediate state into the json file
    states.decompose = {}
    states.decompose.length = len(parts)
    states.decompose.dirs = {
        "mesh": list()
    }

    # Save each part with index suffix
    os.makedirs(os.path.dirname(cfgs.mesh_working_dir), exist_ok=True)

    for i, part in enumerate(parts, start=1):
        tri = trimesh.Trimesh(vertices=part[0], faces=part[1], process=False)
        fname = f"decomposed_{i}.{cfgs.extension}"
        decompose_dir = os.path.join(cfgs.mesh_working_dir, fname)
        tri.export(decompose_dir)
        states.decompose.dirs.mesh.append(decompose_dir)

    try:
        ms.clear()
    except Exception:
        pass

    return