import os
import glob
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict

import trimesh


@dataclass
class SystemConfig:
    # --------------------------------------------------------------------------
    # 1. 사용자 정의 설정
    # --------------------------------------------------------------------------
    # 병합할 원본 메시 파일들이 있는 경로의 기본 이름
    # 예: 'out/convex_decomp/refined_bonsai_pot'는
    # 'refined_bonsai_pot_1.ply', 'refined_bonsai_pot_2.ply', ... 와 매칭됩니다.
    input_mesh_base_path: str = os.path.abspath("../out/convex_decomp/refined_spray")

    # 병합된 메시를 저장할 출력 디렉토리
    output_dir: str = os.path.abspath("out/merged_meshes")

    # 병합할 메시 그룹을 지정 (1부터 시작하는 번호).
    # 예: ((1, 2), (3, 4, 6), (5, 7)) -> 1,2번 메시를 합치고, 3,4,6번을 합치고...
    # None으로 설정하면 아무 작업도 수행하지 않습니다.
    mesh_groups: Optional[Tuple[Tuple[int, ...], ...]] = ((1, 2, 3, 4, 5, 7, 8), (6, 9))

    # --------------------------------------------------------------------------
    # 2. 자동 생성 설정
    # --------------------------------------------------------------------------
    root_dir: str = field(default_factory=lambda: os.path.abspath(os.path.join(os.path.dirname(__file__))),
                          init=False)

    def __post_init__(self):
        # 출력 디렉토리가 존재하지 않으면 생성
        os.makedirs(self.output_dir, exist_ok=True)
        # 절대 경로로 변환
        self.input_mesh_base_path = os.path.join(self.root_dir, self.input_mesh_base_path)
        self.output_dir = os.path.join(self.root_dir, self.output_dir)


def merge_meshes_by_group(cfg: SystemConfig):
    """
    설정된 그룹에 따라 메시 파일들을 병합하고 저장합니다.
    """
    if not cfg.mesh_groups:
        print("Warning: No mesh groups are defined in the configuration. Exiting.")
        return

    # 1. 분해된 모든 메시 파일 경로 찾기
    # glob을 사용하여 '_*.ply' 패턴에 맞는 모든 파일을 찾습니다.
    all_part_paths = sorted(glob.glob(f"{cfg.input_mesh_base_path}_*.ply"))

    if not all_part_paths:
        print(f"Error: No decomposed parts found with base path: {cfg.input_mesh_base_path}")
        return

    # 2. 파일 경로를 파트 번호와 매핑하는 딕셔너리 생성
    part_path_map: Dict[int, str] = {}
    for path in all_part_paths:
        try:
            # 파일 이름에서 숫자 부분 추출 (예: refined_bonsai_pot_12.ply -> 12)
            filename = os.path.basename(path)
            num_str = filename.split('_')[-1].split('.')[0]
            part_num = int(num_str)
            part_path_map[part_num] = path
        except (ValueError, IndexError):
            print(f"Warning: Could not parse part number from filename: {path}")

    print(f"Found {len(part_path_map)} individual mesh parts.")

    # 3. 정의된 그룹별로 메시 병합 및 저장
    output_base_name = os.path.basename(cfg.input_mesh_base_path)

    for group in cfg.mesh_groups:
        print(f"\nProcessing group: {group}")
        meshes_to_merge = []

        # 현재 그룹에 속한 메시들을 로드
        for part_num in group:
            if part_num in part_path_map:
                mesh_path = part_path_map[part_num]
                print(f"-> Loading part {part_num} from: {mesh_path}")
                mesh = trimesh.load(mesh_path, force="mesh")
                meshes_to_merge.append(mesh)
            else:
                print(f"Warning: Part {part_num} not found. Skipping.")

        # 병합할 메시가 2개 이상 있을 경우에만 진행
        if len(meshes_to_merge) >= 1:
            print(f"-> Merging {len(meshes_to_merge)} meshes...")
            # trimesh의 concatenate 함수를 사용하여 모든 메시를 하나로 합침
            combined_mesh = trimesh.util.concatenate(meshes_to_merge)

            # 파일 이름은 그룹에서 가장 작은 번호를 사용
            output_part_number = min(group)
            out_filename = f"{output_base_name}_merged_{output_part_number}.ply"
            out_path = os.path.join(cfg.output_dir, out_filename)

            # 병합된 메시를 파일로 내보내기
            combined_mesh.export(out_path)
            print(f"Saved merged mesh for group {group} to: {out_path}")
        else:
            print(f"-> No valid meshes found for group {group}. Nothing to merge.")

    print("\nProcessing finished successfully.")


if __name__ == "__main__":
    config = SystemConfig()
    merge_meshes_by_group(config)