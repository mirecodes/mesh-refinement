# -------- Utility: R@p + t / R@n --------
import numpy as np
import plyfile
from plyfile import PlyData, PlyElement
import pymeshlab
import trimesh


def apply_Rt_points(P: np.ndarray, Rmat: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return (P @ Rmat.T) + tvec

def apply_R_normals(N: np.ndarray, Rmat: np.ndarray) -> np.ndarray:
    N2 = (N @ Rmat.T)
    N2 /= (np.linalg.norm(N2, axis=1, keepdims=True) + 1e-12)
    return N2


# -------- Save mesh with applied transform --------
def save_mesh_with_transform(ms_mesh: pymeshlab.Mesh, out_path: str, T4: np.ndarray):
    V = ms_mesh.vertex_matrix().astype(np.float64)
    F = ms_mesh.face_matrix().astype(np.int64)

    Rmat = T4[:3, :3].astype(np.float64)
    tvec = T4[:3, 3].astype(np.float64)

    V2 = apply_Rt_points(V, Rmat, tvec)

    # Keep normals/colors if available

    # normals
    vnorm = ms_mesh.vertex_normal_matrix()
    if vnorm is not None and len(vnorm) == len(V):
        vnorm = apply_R_normals(vnorm.astype(np.float64), Rmat)
    else:
        vnorm = None

    # colors
    vcols = ms_mesh.vertex_color_matrix() if ms_mesh.has_vertex_color() else None

    # build trimesh
    tm = trimesh.Trimesh(vertices=V2, faces=F, process=False)

    if vnorm is not None:
        tm.vertex_normals = vnorm
    if vcols is not None:
        rgb = np.clip(vcols[:, :3] * 255.0, 0, 255).astype(np.uint8)
        alpha = 255 * np.ones((rgb.shape[0], 1), np.uint8)
        tm.visual.vertex_colors = np.hstack([rgb, alpha])

    tm.export(out_path)


# -------- Apply transform to Gaussian PLY --------
def save_gaussian_with_transform(ply: plyfile.PlyData, gaussian_out_path: str, T4: np.ndarray):
    # ply = PlyData.read(gaussian_in_path)
    v = ply['vertex']
    names = v.data.dtype.names

    Rmat = T4[:3, :3].astype(np.float64)
    tvec = T4[:3, 3].astype(np.float64)

    if all(k in names for k in ('x','y','z')):
        P = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
        P2 = apply_Rt_points(P, Rmat, tvec)
        v['x'] = P2[:, 0].astype(v['x'].dtype)
        v['y'] = P2[:, 1].astype(v['y'].dtype)
        v['z'] = P2[:, 2].astype(v['z'].dtype)

    if all(k in names for k in ('nx','ny','nz')):
        N = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float64)
        N2 = apply_R_normals(N, Rmat)
        v['nx'] = N2[:, 0].astype(v['nx'].dtype)
        v['ny'] = N2[:, 1].astype(v['ny'].dtype)
        v['nz'] = N2[:, 2].astype(v['nz'].dtype)

    PlyData([PlyElement.describe(v.data, 'vertex')], text=False).write(gaussian_out_path)
