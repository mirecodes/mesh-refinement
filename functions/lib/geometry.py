import numpy as np
from typing import Tuple, List

# =============================================================================
# Geometry & linear algebra helpers
# =============================================================================
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    n = np.linalg.norm(v)
    return v * 0.0 if n < 1e-12 else v / n

def orthonormal_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normalize(n)
    axes = np.eye(3)
    k = int(np.argmin(np.abs(axes @ n)))
    u = normalize(np.cross(n, axes[k]))
    v = normalize(np.cross(n, u))
    return u, v, n

def pca_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    c = pts.mean(axis=0)
    X = pts - c
    C = X.T @ X
    _, V = np.linalg.eigh(C)
    n = normalize(V[:, 0])
    return c, n

def rotmat(axis: np.ndarray, theta: float) -> np.ndarray:
    """World-frame axis-angle rotation matrix (right-handed)."""
    a = normalize(np.asarray(axis, dtype=float))
    x, y, z = a
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=float)

def closest_points_between_lines(p1: np.ndarray, d1: np.ndarray,
                                  p2: np.ndarray, d2: np.ndarray):
    """Closest points on two (p+t d) lines; returns q1,q2,dist."""
    p1 = np.asarray(p1, dtype=float); d1 = normalize(d1)
    p2 = np.asarray(p2, dtype=float); d2 = normalize(d2)
    r = p1 - p2
    a = 1.0
    b = float(np.dot(d1, d2))
    c = 1.0
    d = float(np.dot(d1, r))
    e = float(np.dot(d2, r))
    den = a*c - b*b
    if abs(den) < 1e-12:
        s = e / (c if c > 1e-30 else 1.0)
        q1 = p1
        q2 = p2 + s * d2
    else:
        t = (b*e - c*d) / den
        s = (a*e - b*d) / den
        q1 = p1 + t * d1
        q2 = p2 + s * d2
    dist = float(np.linalg.norm(q1 - q2))
    return q1, q2, dist

# =============================================================================
# 2D polygon tests & projections
# =============================================================================
def points_in_polygon_2d(poly_xy: np.ndarray, grid_xy: np.ndarray) -> np.ndarray:
    """Ray casting in 2D."""
    x = grid_xy[:, 0]; y = grid_xy[:, 1]
    xp = poly_xy[:, 0]; yp = poly_xy[:, 1]
    n = len(poly_xy)
    inside = np.zeros(len(grid_xy), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = xp[i], yp[i]
        xj, yj = xp[j], yp[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside

def nearest_segment_projection(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Project 2D points P onto segment A→B; return (proj, t, d2)."""
    AB = B - A
    denom = (AB @ AB) + 1e-12
    t = ((P - A) @ AB) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = A + t[:, None] * AB
    d2 = np.sum((P - proj) ** 2, axis=1)
    return proj, t, d2

def boundary_distance_and_height(grid_xy: np.ndarray, poly_uv: np.ndarray, h_loop: np.ndarray):
    """Distance to polyline and interpolated boundary height at closest point."""
    if poly_uv.shape[0] >= 2 and not np.allclose(poly_uv[0], poly_uv[-1]):
        poly_uv = np.vstack([poly_uv, poly_uv[0]])
        h_loop = np.concatenate([h_loop, h_loop[0:1]])
    M = grid_xy.shape[0]
    dist2 = np.full(M, np.inf, dtype=float)
    hb = np.zeros(M, dtype=float)
    for i in range(len(poly_uv) - 1):
        A = poly_uv[i]; B = poly_uv[i + 1]
        _, t, d2 = nearest_segment_projection(grid_xy, A, B)
        upd = d2 < dist2
        if np.any(upd):
            dist2[upd] = d2[upd]
            hb[upd] = (1.0 - t[upd]) * h_loop[i] + t[upd] * h_loop[i + 1]
    return np.sqrt(dist2), hb

# =============================================================================
# Loop densification & planar triangulation
# =============================================================================
def densify_loop_uv(poly_uv: np.ndarray, step: float) -> np.ndarray:
    """Insert samples along edges every ~step (closed polyline out)."""
    poly_uv = np.asarray(poly_uv, dtype=float)
    if poly_uv.shape[0] >= 2 and not np.allclose(poly_uv[0], poly_uv[-1]):
        poly_uv = np.vstack([poly_uv, poly_uv[0]])
    pts = [poly_uv[0]]
    for i in range(len(poly_uv) - 1):
        a = poly_uv[i]; b = poly_uv[i + 1]
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L <= 1e-12:
            continue
        nadd = max(0, int(np.ceil(L / max(step, 1e-12))) - 1)
        for t in range(1, nadd + 1):
            pts.append(a + (t / (nadd + 1)) * seg)
        pts.append(b)
    return np.asarray(pts, dtype=float)

def triangulate_points_2d(points_uv: np.ndarray):
    """Delaunay via matplotlib.tri. Returns (triangulation_object, used_flag)."""
    try:
        import matplotlib.tri as mtri
    except Exception as e:
        raise ImportError("matplotlib.tri is required for triangulation") from e
    pts = np.asarray(points_uv, dtype=float)
    if pts.shape[0] < 3:
        return None, False
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
    if tri.triangles.size == 0:
        return None, False
    return tri, True


# =============================================================================
# Externalized local updaters
# =============================================================================
def update_from_neighbors_dense(idx: int,
                                pts_uv: np.ndarray,
                                H: np.ndarray,
                                known: np.ndarray,
                                neighbors: List[np.ndarray] | None,
                                sigma: float,
                                k_min: int,
                                radius: float) -> bool:
    """Gaussian-weighted update using neighbor graph or radius search."""
    if neighbors is not None:
        cand = neighbors[idx]
    else:
        dif = pts_uv - pts_uv[idx]
        d = np.sqrt(np.sum(dif * dif, axis=1))
        cand = np.nonzero((d > 0) & (d <= radius))[0]
    if cand.size == 0:
        return False
    kk = cand[known[cand]]
    if kk.size < max(1, k_min):
        return False
    d2 = np.sum((pts_uv[kk] - pts_uv[idx]) ** 2, axis=1)
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    ws = float(np.sum(w))
    if ws <= 1e-12:
        return False
    H[idx] = float(np.sum(w * H[kk]) / ws)
    known[idx] = True
    return True


def upd_radius(idx: int,
               pts_uv: np.ndarray,
               H: np.ndarray,
               known: np.ndarray,
               sigma: float,
               k_min: int,
               radius: float) -> bool:
    """Gaussian update with pure radius search (no triangulation)."""
    d = np.sqrt(np.sum((pts_uv - pts_uv[idx])**2, axis=1))
    cand = np.nonzero((d > 0) & (d <= radius))[0]
    kk = cand[known[cand]]
    if kk.size < k_min:
        return False
    w = np.exp(-(d[kk]**2) / (2.0 * sigma * sigma))
    ws = float(np.sum(w))
    if ws <= 1e-12:
        return False
    H[idx] = float(np.sum(w * H[kk]) / ws)
    return True
