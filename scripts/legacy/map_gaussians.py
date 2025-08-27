import os
import glob
from dataclasses import dataclass, field
from typing import Optional, Tuple

import trimesh
import numpy as np
from plyfile import PlyElement, PlyData
from trimesh.proximity import ProximityQuery


@dataclass
class SystemConfig:
    # --------------------------------------------------------------------------
    # 1. 사용자 정의 설정
    # --------------------------------------------------------------------------
    # 분해된 메시 파일들이 있는 경로의 기본 이름
    # 예: 'out/convex_decomp/refined_bonsai_pot'는
    # 'refined_bonsai_pot_1.ply', 'refined_bonsai_pot_2.ply', ... 와 매칭됩니다.
    decomposed_mesh_base_path: str = "out/convex_decomp/refined_spray"

    # 가우시안 포인트 클라우드 파일 경로
    gaussian_in_path: str = "data/gaussians/spray_gaussian.ply"

    part_groups: Optional[Tuple[Tuple[int, ...], ...]] = ((1, 2, 3, 4, 5, 7, 8, ), (6, 9, ))

    # 그룹화된 가우시안 포인트를 저장할 출력 디렉토리
    gaussian_out_dir: str = "../../out/gaussian_parts"

    # --------------------------------------------------------------------------
    # 2. 자동 생성 설정 (필요 시 사용)
    # --------------------------------------------------------------------------
    root_dir: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                          init=False)

    def __post_init__(self):
        # 출력 디렉토리가 존재하지 않으면 생성
        os.makedirs(self.gaussian_out_dir, exist_ok=True)
        self.decomposed_mesh_base_dir = os.path.join(self.root_dir, self.decomposed_mesh_base_path)
        self.gaussian_in_dir: str = os.path.join(self.root_dir, self.gaussian_in_path)


def assign_points_to_parts(cfg: SystemConfig):
    """
    가우시안 포인트들을 가장 가까운 메시 조각에 할당하고, 원본 데이터 형식을 유지하여 파일로 저장합니다.
    """
    # 1. 가우시안 포인트 클라우드 로드
    # 1a. Trimesh를 사용해 기하학적 위치(vertices) 정보만 로드 (거리 계산용)
    print(f"Loading Gaussian point cloud from: {cfg.gaussian_in_dir}")
    try:
        gaussian_cloud_geom = trimesh.load(cfg.gaussian_in_dir)
        gaussian_points = gaussian_cloud_geom.vertices
    except Exception as e:
        print(f"Error: Failed to load Gaussian geometry with Trimesh. {e}")
        return

    # 1b. plyfile을 사용해 모든 속성을 포함한 전체 데이터 로드 (데이터 보존용)
    try:
        plydata = PlyData.read(cfg.gaussian_in_dir)
        vertex_data = plydata['vertex'].data
        print(f"-> Loaded {len(vertex_data)} points with all attributes.")
    except Exception as e:
        print(f"Error: Failed to load full Gaussian data with plyfile. {e}")
        return

    # 2. 분해된 메시 조각들 로드 (기존과 동일)
    part_paths = sorted(glob.glob(f"{cfg.decomposed_mesh_base_dir}_*.ply"))
    if not part_paths:
        print(f"Error: No decomposed parts found with base path: {cfg.decomposed_mesh_base_dir}")
        return
    print(f"Found {len(part_paths)} decomposed mesh parts. Loading...")
    parts = [trimesh.load(p, force="mesh") for p in part_paths]
    print("-> All parts loaded.")

    # 3. 거리 계산 및 가장 가까운 그룹 찾기 (기존과 동일)
    print("Calculating distances from points to each mesh part...")
    all_distances = []
    for i, part_mesh in enumerate(parts):
        prox_query = ProximityQuery(part_mesh)
        distances = np.abs(prox_query.signed_distance(gaussian_points))
        all_distances.append(distances)
        print(f"-> Calculated distances for part {i + 1}/{len(parts)}")

    distances_matrix = np.array(all_distances)

    if cfg.part_groups:
        print("Processing part groups...")
        group_distances_list = []
        for group in cfg.part_groups:
            group_indices = [p - 1 for p in group]
            min_dist_to_group = np.min(distances_matrix[group_indices, :], axis=0)
            group_distances_list.append(min_dist_to_group)
        final_distances = np.array(group_distances_list)
        print("-> Group distance calculation complete.")
    else:
        print("No groups defined, assigning to individual parts.")
        final_distances = distances_matrix

    print("Assigning each point to the closest part/group...")
    closest_indices = np.argmin(final_distances, axis=0)
    print("-> Assignment complete.")

    # 5. 결과를 그룹화하여 "원본 형식 그대로" 파일로 저장 (수정된 부분)
    print("Saving grouped Gaussian points while preserving original format...")
    output_base_name = os.path.splitext(os.path.basename(cfg.gaussian_in_path))[0]

    if cfg.part_groups:
        # 그룹별로 저장
        for i, group in enumerate(cfg.part_groups):
            mask = (closest_indices == i)
            # 원본의 모든 속성을 포함한 데이터에서 마스크를 사용해 필터링
            points_for_this_group_data = vertex_data[mask]

            if points_for_this_group_data.shape[0] > 0:
                # 필터링된 데이터로 새로운 PlyElement 생성
                # 원본의 데이터 타입과 속성 이름(header)이 그대로 유지됨
                new_element = PlyElement.describe(points_for_this_group_data, 'vertex')

                # 새로운 PlyData 객체 생성 (binary 형식으로 저장 권장)
                new_plydata = PlyData([new_element], text=False)

                output_part_number = min(group)
                out_path = os.path.join(cfg.gaussian_out_dir, f"{output_base_name}_{output_part_number}.ply")

                # plyfile을 사용해 파일 쓰기
                new_plydata.write(out_path)
                print(f"Saved {len(points_for_this_group_data)} points for group {group} to: {out_path}")
            else:
                print(f"Group {group} has no closest points. Skipping.")
    else:
        # 개별 파트별로 저장 (그룹 설정이 없을 경우)
        for i in range(len(parts)):
            mask = (closest_indices == i)
            points_for_this_part_data = vertex_data[mask]

            if points_for_this_part_data.shape[0] > 0:
                new_element = PlyElement.describe(points_for_this_part_data, 'vertex')
                new_plydata = PlyData([new_element], text=False)
                out_path = os.path.join(cfg.gaussian_out_dir, f"{output_base_name}_{i + 1}.ply")
                new_plydata.write(out_path)
                print(f"Saved {len(points_for_this_part_data)} points to: {out_path}")
            else:
                print(f"Part {i + 1} has no closest points. Skipping.")

    print("\nProcessing finished successfully.")


if __name__ == "__main__":
    config = SystemConfig()
    assign_points_to_parts(config)