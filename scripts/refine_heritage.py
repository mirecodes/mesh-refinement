from dataclasses import dataclass, field
import pymeshlab
import os
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from collections import defaultdict
import open3d as o3d
from scipy.spatial.transform import Rotation as R  # 회전 변환을 위해 scipy 임포트
from scipy.spatial import Delaunay


@dataclass
class MeshRefinementConfig:
    # --------------------------------------------------------------------------
    # 1. 기본 설정 (사용자가 직접 수정)
    # --------------------------------------------------------------------------
    # 프로젝트 루트 폴더 기준 입력/출력 경로
    mesh_in_path: str = "data/meshes/bonsai_pot.ply"
    mesh_out_path: str = "out/meshes/refined_bonsai_pot.obj"
    mesh_dev_path: str = "out/meshes/processed_bonsai_pot.obj"

    # 기능 활성화 플래그
    automatic_alignment: bool = True  # True: 시작 시 PCA로 바닥면을 자동 정렬
    step_by_step_visualization: bool = True  # True: 각 단계별로 결과물을 시각화

    # --- [추가] 초기 클리닝 파라미터 ---
    # 메인 메쉬의 대각선 길이 대비 % 크기로, 이보다 작은 조각들을 제거합니다.
    # 예: 1.0은 1%보다 작은 모든 고립된 조각을 제거합니다.
    initial_cleanup_threshold: float = 1.0

    # --------------------------------------------------------------------------
    # 2. 자동 생성 경로 (수정 불필요)
    # --------------------------------------------------------------------------
    root_dir: str = field(default_factory=os.getcwd, init=False)
    mesh_in_dir: str = field(init=False)
    mesh_out_dir: str = field(init=False)
    mesh_dev_dir: str = field(init=False)

    def __post_init__(self):
        self.mesh_in_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_in_path))
        self.mesh_out_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_out_path))
        self.mesh_dev_dir = os.path.abspath(os.path.join(self.root_dir, "..", self.mesh_dev_path))


# ============================================================================
# 루프 찾기 및 자동 정렬 함수 (최종 검증 버전)
# ============================================================================

def find_boundary_loops(t_mesh):
    # (이 함수는 제공해주신 코드와 동일하게 유지됩니다)
    edges = t_mesh.edges_sorted
    unique_edges, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) == 0:
        return []
    neighbours = defaultdict(list)
    for v1, v2 in boundary_edges:
        neighbours[v1].append(v2)
        neighbours[v2].append(v1)
    loops = []
    remaining_edges = set(map(tuple, boundary_edges))
    while remaining_edges:
        v_start, v_next = remaining_edges.pop()
        current_loop = [v_start, v_next]
        while v_next != v_start:
            v_prev = current_loop[-2]
            options = neighbours[v_next]
            if len(options) != 2:
                break
            if options[0] == v_prev:
                v_next = options[1]
            else:
                v_next = options[0]
            if v_next == v_start:
                break
            current_loop.append(v_next)
        for i in range(len(current_loop) - 1):
            remaining_edges.discard(tuple(sorted((current_loop[i], current_loop[i + 1]))))
        remaining_edges.discard(tuple(sorted((current_loop[-1], current_loop[0]))))
        loops.append(current_loop)
    return loops


def calculate_transforms(ms):
    # (이 함수는 제공해주신 코드와 동일하게 유지됩니다)
    print("원본 메쉬에서 정렬을 위한 변환 행렬을 계산합니다...")
    current_mesh = ms.current_mesh()
    t_mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())
    loops_indices = find_boundary_loops(t_mesh)
    if not loops_indices:
        print("경고: 경계 루프가 없어 변환 행렬을 계산할 수 없습니다.")
        return None, None

    print(f"# loops: {len(loops_indices)}")
    largest_loop_indices = max(loops_indices, key=len)
    loop_verts = t_mesh.vertices[largest_loop_indices]

    pca = PCA(n_components=3)
    pca.fit(loop_verts)
    plane_normal = pca.components_[2]
    if np.dot(plane_normal, loop_verts.mean(axis=0) - t_mesh.centroid) < 0:
        plane_normal *= -1

    # --- 여기가 사용자께서 제공해주신, 작동이 확인된 코드로 교체된 부분 ---
    target_normal = np.array([0., 0., 1.])
    v = np.cross(plane_normal, target_normal)
    s = np.linalg.norm(v)
    c = np.dot(plane_normal, target_normal)

    if np.isclose(s, 0):
        rotation_matrix = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s ** 2))
    # --- 교체 끝 ---

    # 가상으로 회전시켜 Z 이동값 계산
    rotated_loop_verts = loop_verts @ rotation_matrix.T
    z_mean = rotated_loop_verts[:, 2].mean()

    # Forward 변환 행렬 (회전 + 이동) 생성
    translation_matrix = np.eye(4)
    translation_matrix[2, 3] = z_mean

    rotation_4x4 = np.eye(4)
    rotation_4x4[:3, :3] = rotation_matrix

    forward_transform = translation_matrix @ rotation_4x4

    inverse_transform = np.linalg.inv(forward_transform)

    print("Forward 및 Inverse 변환 행렬 계산 완료.")
    return forward_transform, inverse_transform


def apply_transform_from_matrix(ms, matrix_4x4):
    # (이 함수는 제공해주신 코드와 동일하게 유지됩니다)
    rotation_matrix = matrix_4x4[:3, :3]
    translation_vector = matrix_4x4[:3, 3]

    rot = R.from_matrix(rotation_matrix)
    euler_angles = rot.as_euler('xyz', degrees=True)

    print(f"euler_angles {euler_angles}")

    ms.compute_matrix_from_translation_rotation_scale(
        rotationx=euler_angles[0] / 180 * np.pi,
        rotationy=euler_angles[1] / 180 * np.pi,
        rotationz=euler_angles[2] / 180 * np.pi,
        translationx=translation_vector[0],
        translationy=translation_vector[1],
        translationz=translation_vector[2],
        freeze=True
    )


def show_mesh(mesh_set, title="Intermediate Step", enabled=True):
    # (이 함수는 제공해주신 코드와 동일하게 유지됩니다)
    if not enabled or mesh_set.mesh_number() == 0:
        if not enabled:
            print(f"[{title}] 시각화 비활성화됨. 건너뜁니다.")
        return
    print(f"[{title}] 결과를 시각화합니다...")
    current_mesh = ms.current_mesh()
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(current_mesh.vertex_matrix()),
        o3d.utility.Vector3iVector(current_mesh.face_matrix())
    )
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([o3d_mesh], window_name=title)


def show_mesh_with_cutting_plane(ms, title, cut_z_value, enabled=True):
    """메쉬, 좌표축, 그리고 지정된 Z 위치의 절단 평면을 함께 시각화합니다."""
    if not enabled or ms.mesh_number() == 0:
        if not enabled:
            print(f"[{title}] 시각화 비활성화됨. 건너뜁니다.")
        return

    print(f"[{title}] 메쉬, 좌표축, 절단 평면(z={cut_z_value})을 시각화합니다...")

    # 1. 메쉬 준비
    current_mesh = ms.current_mesh()
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(current_mesh.vertex_matrix()),
        o3d.utility.Vector3iVector(current_mesh.face_matrix())
    )
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    # 2. 좌표축 준비
    mesh_bounds = o3d_mesh.get_axis_aligned_bounding_box()
    axis_size = max(mesh_bounds.get_extent()) * 0.5
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])

    # 3. 절단 평면 준비
    bounds_extent = mesh_bounds.get_extent()
    plane_width = bounds_extent[0] * 1.2
    plane_height = bounds_extent[1] * 1.2
    plane_depth = max(bounds_extent) * 0.005

    plane = o3d.geometry.TriangleMesh.create_box(width=plane_width, height=plane_height, depth=plane_depth)

    plane_center = mesh_bounds.get_center()
    plane.translate(
        (plane_center[0],
         plane_center[1],
         cut_z_value - plane_depth / 2),
        relative=False
    )
    plane.paint_uniform_color([1, 0, 0])

    # 4. 함께 시각화
    o3d.visualization.draw_geometries(
        [o3d_mesh, coord_frame, plane],
        window_name=title
    )


def flatten_bottom_hole_with_trimesh(ms: pymeshlab.MeshSet, target_z: float = 0.0) -> pymeshlab.MeshSet:
    """
    Pymeshlab MeshSet을 받아, 가장 큰 경계 루프의 정점들을 지정된 Z 평면으로 이동시킨다.
    """
    print("Trimesh를 사용하여 경계 루프 평면화 시작...")
    current_mesh = ms.current_mesh()
    t_mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

    loops_indices = find_boundary_loops(t_mesh)
    if not loops_indices:
        print("경고: 경계 루프를 찾지 못해 평면화 작업을 건너뜁니다.")
        return ms

    largest_loop_indices = max(loops_indices, key=len)
    print(f"가장 큰 경계 루프({len(largest_loop_indices)}개의 정점)를 찾았습니다.")

    t_mesh.vertices[largest_loop_indices, 2] = target_z
    print(f"경계 루프의 모든 정점을 Z={target_z} 평면으로 투영(이동)했습니다.")

    new_ms = pymeshlab.MeshSet()
    new_ms.add_mesh(pymeshlab.Mesh(vertex_matrix=t_mesh.vertices, face_matrix=t_mesh.faces))
    return new_ms


def fill_bottom_hole_constrained(ms: pymeshlab.MeshSet) -> pymeshlab.MeshSet:
    """
    Pymeshlab MeshSet을 받아, 가장 큰 경계 루프를 찾아 고품질의 평면으로 해당 구멍을 메운다.
    """
    print("Shapely와 Trimesh를 사용하여 '경계 제약 삼각분할'로 바닥 구멍 메우기를 시작합니다...")
    try:
        from shapely.geometry import Polygon
    except ImportError:
        print("\n오류: 'shapely' 라이브러리가 필요합니다. 터미널에서 'pip install shapely'를 실행해주세요.")
        raise

    current_mesh = ms.current_mesh()
    t_mesh = trimesh.Trimesh(vertices=current_mesh.vertex_matrix(), faces=current_mesh.face_matrix())

    loops = find_boundary_loops(t_mesh)
    if not loops:
        print("경고: 메울 구멍(경계 루프)이 없습니다.")
        return ms
    largest_loop_indices = max(loops, key=len)
    print(f"가장 큰 경계 루프({len(largest_loop_indices)}개의 정점)를 찾았습니다.")

    loop_verts_3d = t_mesh.vertices[largest_loop_indices]
    loop_verts_2d = loop_verts_3d[:, :2]
    polygon_2d = Polygon(loop_verts_2d)

    new_vertices_2d, new_faces = trimesh.creation.triangulate_polygon(polygon_2d)

    z_coords = np.zeros((new_vertices_2d.shape[0], 1))
    new_vertices_3d = np.hstack((new_vertices_2d, z_coords))

    face_offset = len(t_mesh.vertices)
    new_faces_offset = new_faces + face_offset
    t_mesh.vertices = np.vstack([t_mesh.vertices, new_vertices_3d])
    t_mesh.faces = np.vstack([t_mesh.faces, new_faces_offset])

    t_mesh.merge_vertices()
    print(f"{len(new_faces)}개의 새로운 면을 추가하여 구멍을 메웠습니다.")

    new_ms = pymeshlab.MeshSet()
    new_ms.add_mesh(pymeshlab.Mesh(vertex_matrix=t_mesh.vertices, face_matrix=t_mesh.faces))
    return new_ms


def run_refinement_pipeline(ms, cfg, forward_transform=None, inverse_transform=None):
    """
    전체 후처리 파이프라인. (고품질 바닥 생성 + 나머지 구멍 메우기)
    """
    print("\n후처리 파이프라인을 시작합니다...")
    show_mesh_with_cutting_plane(ms, "Step 0: 외모 check Plane", 0, enabled=cfg.step_by_step_visualization)

    if forward_transform is not None:
        print("\n[단계 1] 계산된 변환을 적용하여 메쉬를 정렬합니다...")
        apply_transform_from_matrix(ms, forward_transform)
        show_mesh_with_cutting_plane(ms, "Step 1: After Alignment", 0, enabled=cfg.step_by_step_visualization)

    print("\n[단계 2] Screened Poisson 표면 재구성 실행 중...")
    ms.generate_surface_reconstruction_screened_poisson(depth=8, fulldepth=8, preclean=True)
    show_mesh(ms, title="Step 2: After Poisson Reconstruction", enabled=cfg.step_by_step_visualization)

    if forward_transform is not None:
        cutting_plane_z = -1e-6
        print(f"\n[단계 3] 바닥면을 잘라냅니다...")
        show_mesh_with_cutting_plane(ms, "Step 3: Preview Cutting Plane", cutting_plane_z,
                                     enabled=cfg.step_by_step_visualization)

        ms.compute_selection_by_condition_per_vertex(condselect=f'z < {cutting_plane_z}')
        if ms.current_mesh().selected_vertex_number() > 0:
            ms.meshing_remove_selected_vertices()
        show_mesh(ms, title="Step 3: After Cutting Bottom", enabled=cfg.step_by_step_visualization)

    print("\n[단계 4] 메쉬 단순화 ...")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=20000, preservenormal=True)
    show_mesh(ms, title="Step 4: After Simplification", enabled=cfg.step_by_step_visualization)

    print("\n[단계 5] 메쉬 리메싱 ...")
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(1))
    show_mesh(ms, title="Step 5: After Remeshing", enabled=cfg.step_by_step_visualization)

    print("\n[단계 6] 1차 비다양체 복구 ...")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_merge_close_vertices()
    show_mesh(ms, title="Step 6: After 1st Repair", enabled=cfg.step_by_step_visualization)

    if forward_transform is not None:
        print("\n[단계 7] Trimesh를 사용하여 바닥 경계 루프를 z=0 평면으로 이동...")
        ms = flatten_bottom_hole_with_trimesh(ms, target_z=0.0)
        show_mesh(ms, title="Step 7: After Flattening Boundary Loop", enabled=cfg.step_by_step_visualization)

        print("\n[단계 8] '경계 제약 삼각분할'로 바닥면 구멍을 메웁니다...")
        ms = fill_bottom_hole_constrained(ms)
        show_mesh(ms, title="Step 8: After Constrained Fill", enabled=cfg.step_by_step_visualization)

        print("\n[단계 8.5] 중간 비다양체(Non-Manifold) 복구/정리 중...")
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_merge_close_vertices()
        show_mesh(ms, title="Step 8.5: After Mid-Repair", enabled=cfg.step_by_step_visualization)

    print("\n[단계 9] 남아있는 다른 구멍들을 메웁니다...")
    ms.meshing_close_holes(maxholesize=5000)
    show_mesh(ms, title="Step 9: After Filling Other Holes", enabled=cfg.step_by_step_visualization)

    print("\n[단계 10] 최종 비다양체(Non-Manifold) 복구/정리 중...")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_merge_close_vertices()
    show_mesh(ms, title="Step 10: After Final Repair", enabled=cfg.step_by_step_visualization)

    if inverse_transform is not None:
        print("\n[단계 11] 최종 메쉬를 원래 좌표계로 복원합니다...")
        apply_transform_from_matrix(ms, inverse_transform)
        show_mesh(ms, title="Step 11: After Reverting Transform", enabled=cfg.step_by_step_visualization)

    print(f"\n최종 메쉬를 '{cfg.mesh_out_dir}'에 저장 중...")
    os.makedirs(os.path.dirname(cfg.mesh_out_dir), exist_ok=True)
    ms.save_current_mesh(cfg.mesh_out_dir)


if __name__ == "__main__":
    cfg = MeshRefinementConfig()
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(cfg.mesh_in_dir)
    except pymeshlab.PyMeshLabException:
        print(f"경고: '{cfg.mesh_in_path}'를 찾을 수 없어 테스트용 메쉬를 생성합니다.")
        trimesh.creation.bowl().export(cfg.mesh_in_dir)
        ms.load_new_mesh(cfg.mesh_in_dir)

    if ms.mesh_number() > 0:
        if cfg.step_by_step_visualization:
            show_mesh(ms, title="Step 0: Original Mesh")

        # --- [추가된 부분] 초기 클리닝 단계 ---
        # 본격적인 처리를 시작하기 전에, 작고 연결되지 않은 조각들(떠다니는 노이즈)을 제거합니다.
        # 이 단계는 파이프라인의 안정성을 크게 향상시킵니다.
        print("\n[단계 0.5] 작은 고립된 조각들을 제거합니다...")
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(20.0)
        )
        show_mesh(ms, title="Step 0.5: After Initial Cleaning", enabled=cfg.step_by_step_visualization)

        # --- [수정 끝] ---

        forward_transform, inverse_transform = None, None
        if cfg.automatic_alignment:
            forward_transform, inverse_transform = calculate_transforms(ms)

        if forward_transform is not None:
            run_refinement_pipeline(ms, cfg, forward_transform=forward_transform, inverse_transform=inverse_transform)
        else:
            run_refinement_pipeline(ms, cfg)

        print("\n모든 과정이 완료되었습니다.")
    else:
        print("\n메쉬를 로드하지 못해 처리를 시작할 수 없습니다.")