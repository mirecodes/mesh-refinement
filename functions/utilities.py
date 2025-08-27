import functools
from collections import defaultdict

import numpy as np
import pymeshlab
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon


def post_mesh_repair(func):
    '''
    Post-processing of mesh refinement
    Restore non-manifold edges / close vertices
    :return: a decorator function
    '''

    @functools.wraps(func)
    def wrapper(ms, *args, **kwargs):
        result_ms = func(ms, *args, **kwargs)
        result_ms.meshing_repair_non_manifold_edges()
        result_ms.meshing_merge_close_vertices()
        return result_ms
    return wrapper


def repair_mesh(ms: pymeshlab.MeshSet):
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_merge_close_vertices()
    return ms


def replace_mesh_0(ms: pymeshlab.MeshSet, new_mesh: pymeshlab.Mesh) -> pymeshlab.MeshSet:
    out = pymeshlab.MeshSet()
    out.add_mesh(mesh=new_mesh)

    for i in range(1, len(ms.mesh_number())):
        out.add_mesh(ms.mesh(i))
    return out

def replace_mesh_0_points(ms: pymeshlab.MeshSet, verts_matrix, faces_matrix) -> pymeshlab.MeshSet:
    out = pymeshlab.MeshSet()
    out.add_mesh(pymeshlab.Mesh(vertex_matrix=verts_matrix, face_matrix=faces_matrix))

    for i in range(1, ms.mesh_number()):
        out.add_mesh(ms.mesh(i))
    return out


def find_boundary_loops(t_mesh, verbose=False) -> list:
    '''
    Find all the boundary loops of the mesh
    :param t_mesh: trimesh
    :param verbose: bool
    :return: list of loops
    '''
    if verbose: print("[info] Function 'find_boundary_loops' called")

    edges = t_mesh.edges_sorted
    unique_edges, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary_edges = unique_edges[counts == 1] # edges where face counts = 1 -> boundary edges

    if len(boundary_edges) == 0: return []
    neighbours = defaultdict(list)
    for v1, v2 in boundary_edges: neighbours[v1].append(v2); neighbours[v2].append(v1)

    loops = []
    remaining_edges = set(map(tuple, boundary_edges))
    while remaining_edges:
        v_start, v_next = remaining_edges.pop()
        current_loop = [v_start, v_next]
        while v_next != v_start:
            v_prev = current_loop[-2]
            options = neighbours[v_next]
            if len(options) != 2: break
            if options[0] == v_prev:
                v_next = options[1]
            else:
                v_next = options[0]
            if v_next == v_start: break
            current_loop.append(v_next)
        for i in range(len(current_loop) - 1):
            remaining_edges.discard(tuple(sorted((current_loop[i], current_loop[i + 1]))))
        remaining_edges.discard(tuple(sorted((current_loop[-1], current_loop[0]))))
        loops.append(current_loop)

    if verbose: print(f"[info] {len(loops)} loops founds.")
    return loops


def calculate_transforms(ms, verbose=False) -> (np.array, np.array):
    '''
    Calculate the transformation matrices given a mesh
    The mesh is automatically aligned along z-up
    :param ms: mesh_set
    :param verbose: bool
    :return: forward_transform, inverse_transform
    '''
    if verbose: print("[info] Function 'calculate_transforms' called")
    current_mesh = ms.current_mesh()
    t_mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())
    loops_indices = find_boundary_loops(t_mesh)
    if not loops_indices:
        print("[warning] Boundary loop does not exist. Transform matrix cannot be calculated.")
        return None, None

    # get the largest loop (used as a bottom occlusion)
    largest_loop_indices = max(loops_indices, key=len)
    loop_verts = t_mesh.vertices[largest_loop_indices]

    # estimate the planar information of the bottom occlusion
    pca = PCA(n_components=3)
    pca.fit(loop_verts)
    plane_normal = pca.components_[2] # the direction of with the least deviation
    if np.dot(plane_normal, loop_verts.mean(axis=0) - t_mesh.centroid) > 0:
        plane_normal *= -1

    # use Rodriguez' Rotation Formula for rotational matrix
    target_normal = np.array([0., 0., 1.])
    v = np.cross(plane_normal, target_normal)
    s = np.linalg.norm(v)
    c = np.dot(plane_normal, target_normal)

    if np.isclose(s, 0):
        rotation_matrix = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s ** 2))

    # get the translation along z-axis
    rotated_loop_verts = loop_verts @ rotation_matrix.T
    z_mean = np.zeros(3)
    z_mean[0] = rotated_loop_verts[:, 0].mean()
    z_mean[1] = rotated_loop_verts[:, 1].mean()
    z_mean[2] = rotated_loop_verts[:, 2].mean()

    # get the forward se(3) transformation
    translation_matrix = np.eye(4)
    translation_matrix[0:3, 3] = -z_mean

    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation_matrix.T

    # calculate the forward / inverse se(3) transformation
    forward_transform = translation_matrix @ rotation_4x4

    if verbose: print(f"[info] The transformation matrix have found.")
    return forward_transform


def apply_transform_from_matrix(ms: pymeshlab.MeshSet, matrix_4x4, verbose=False):
    '''
    Apply the transformation matrix ms to the mesh
    :param ms: mesh_set
    :param matrix_4x4: transformation matrix
    :param verbose: bool
    :return: None
    '''
    if verbose: print("[info] Function 'apply_transform_from_matrix' called")
    rotation_matrix = matrix_4x4[:3, :3]
    translation_vector = matrix_4x4[:3, 3]

    rot = R.from_matrix(rotation_matrix)
    euler_angles = rot.as_euler('xyz', degrees=True)

    for idx in range(ms.mesh_number()):
        ms.set_current_mesh(idx)
        ms.compute_matrix_from_translation_rotation_scale( # rotation -> translation
            rotationx=euler_angles[0],
            rotationy=euler_angles[1],
            rotationz=euler_angles[2],
            translationx=translation_vector[0],
            translationy=translation_vector[1],
            translationz=translation_vector[2],
            compose=True,
            freeze=True
        )

    if verbose: print("[info] The mesh transformation has applied")

    ms.set_current_mesh(0)

def flatten_bottom_hole(ms: pymeshlab.MeshSet, target_z: float = 0.0, verbose=False) -> pymeshlab.MeshSet:
    '''
    flatten the boundary loop at the bottom (biggest boundary loop) onto a plane.
    :param ms: mesh_set
    :param target_z: float
    :return: pymeshlab.MeshSet
    '''

    if verbose: print("[info] function 'flatten_bottom_hole' called")

    # pymeshlab mesh -> trimesh mesh
    current_mesh = ms.current_mesh()
    verts = current_mesh.vertex_matrix()
    faces = current_mesh.face_matrix()
    t_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # find the largest loop
    loops_indices = find_boundary_loops(t_mesh)
    if not loops_indices:
        print("[warning] Boundary loop does not exist. Transform matrix cannot be calculated.")
        return ms

    largest_loop_indices = max(loops_indices, key=len)
    if verbose: print(f"[info] Found the largest loop with {len(largest_loop_indices)} vertices.")

    # translate all loop vertices onto target_z plane
    t_mesh.vertices[largest_loop_indices, 2] = target_z
    if verbose: print(f"[info] Moved the boundary loop vertices onto z={target_z}.")

    # put the mesh format back to the meshlab mesh
    out = replace_mesh_0_points(ms, t_mesh.vertices, t_mesh.faces)
    return out


def fill_bottom_hole(ms: pymeshlab.MeshSet, verbose=False) -> pymeshlab.MeshSet:
    '''
    apply constrained delaunay infill technique for the occlusion at the bottom
    :param ms: mesh_set
    :param verbose: bool
    :return: pymeshlab.MeshSet
    '''

    if verbose: print("[info] function 'fill_bottom_hole' called")

    # pymeshlab mesh -> trimesh mesh
    current_mesh = ms.current_mesh()
    t_mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(),
                             faces=current_mesh.face_matrix())

    # find the largest loop
    loops = find_boundary_loops(t_mesh)
    if not loops:
        print("[warning] Boundary loop does not exist. Transform matrix cannot be calculated.")
        return ms

    largest_loop_indices = max(loops, key=len)
    if verbose: print(f"[info] Found the largest loop with {len(largest_loop_indices)} vertices.")

    # find the 3D-coordinates of the loop vertices
    # assumed that the vertices are already on the const-z plane
    loop_verts_3d = t_mesh.vertices[largest_loop_indices]

    # transform 3D-space vertices onto 2D-plane vertices (ignore z-value)
    loop_verts_2d = loop_verts_3d[:, :2]  # use only (x, y) coordinates
    polygon_2d = Polygon(loop_verts_2d)

    # triangulation
    new_vertices_2d, new_faces = trimesh.creation.triangulate_polygon(polygon_2d)

    # restore the 2D-plane vertices into the original space
    z_coords = np.zeros((new_vertices_2d.shape[0], 1)) # include z=0
    new_vertices_3d = np.hstack((new_vertices_2d, z_coords))

    # merge the generated plane with the original mesh
    face_offset = len(t_mesh.vertices)
    new_faces_offset = new_faces + face_offset

    t_mesh.vertices = np.vstack([t_mesh.vertices, new_vertices_3d])
    t_mesh.faces = np.vstack([t_mesh.faces, new_faces_offset])

    # clean the duplicated vertices
    t_mesh.merge_vertices()


    if verbose: print(f"[info] Fill the bottom holes by adding {len(new_faces)} faces.")

    # put the mesh format back to the meshlab mesh
    out = replace_mesh_0_points(ms, t_mesh.vertices, t_mesh.faces)
    return out