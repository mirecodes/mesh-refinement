import numpy as np
import trimesh
from yourdfpy import URDF
import os
import yaml
from dataclasses import dataclass


@dataclass
class VoxelConfig:
    # Name of the object to process (must match a key in the YAML config)
    object_name: str = "vault"
    
    # Path to the YAML configuration file
    config_file: str = "merge_config.yml"
    
    # Voxel pitch (resolution) in meters
    voxel_pitch: float = 0.005


def merge_urdf_voxel_ply(urdf_path, joint_cfg, output_path="fused_voxel.ply", voxel_pitch=0.005):
    print(f"[Info] URDF 로드 중: {urdf_path}")
    robot = URDF.load(urdf_path)

    # 1. Configuration 적용
    valid_cfg = {k: v for k, v in joint_cfg.items() if k in robot.actuated_joint_names}
    robot.update_cfg(configuration=valid_cfg)

    # 2. Scene 객체 가져오기
    scene = robot.scene
    all_meshes = []

    for name, node in scene.geometry.items():
        transform = scene.graph.get(name)[0]
        mesh = node.copy()
        mesh.apply_transform(transform)
        all_meshes.append(mesh)

    if not all_meshes:
        print("[Error] 병합할 메쉬가 없습니다.")
        return None

    print("[Info] 메쉬들을 하나의 덩어리로 뭉칩니다 (Concatenate)...")
    combined_raw = trimesh.util.concatenate(all_meshes)
    
    # 원본 메쉬의 중심점과 바운딩 박스 저장 (나중에 복원 확인용)
    raw_bounds = combined_raw.bounds
    raw_center = combined_raw.centroid
    print(f"[Debug] Raw Mesh Bounds: {raw_bounds}")
    print(f"[Debug] Raw Mesh Center: {raw_center}")

    # 3. Voxelization & Remeshing
    print(f"[Info] 복셀화 및 재구성 진행 중 (Pitch: {voxel_pitch}m)...")

    try:
        # Voxelization with filling
        voxel_grid = combined_raw.voxelized(pitch=voxel_pitch)
        voxel_grid.fill() 
        
        # Marching Cubes로 메쉬 재구성
        final_mesh = voxel_grid.marching_cubes

        # [Fix] 스케일 및 위치 보정
        # trimesh 버전에 따라 marching_cubes가 복셀 인덱스 좌표(정수)를 반환할 수도 있고,
        # 이미 변환된 좌표를 반환할 수도 있음.
        # 사용자가 "200배 작아졌다"고 한다면, 현재 final_mesh가 너무 작은 상태.
        # 즉, 이미 pitch가 적용되었거나, 혹은 단위가 m인데 cm로 해석되어야 하는 상황일 수 있음.
        
        # 확인을 위해 현재 final_mesh의 스케일을 체크
        final_bounds = final_mesh.bounds
        final_size = final_bounds[1] - final_bounds[0]
        raw_size = raw_bounds[1] - raw_bounds[0]
        
        # 스케일 비율 계산 (Voxel / Raw)
        scale_ratio = np.mean(final_size / (raw_size + 1e-9))
        print(f"[Debug] Initial Voxel Mesh Size: {final_size}")
        print(f"[Debug] Scale Ratio (Voxel/Raw): {scale_ratio:.4f}")
        
        # 만약 스케일이 너무 작다면 (예: 0.005배), pitch의 역수를 곱해야 할 수도 있음.
        # 하지만 보통 marching_cubes 결과가 정수라면 pitch를 곱해야 함.
        # 사용자가 "200배 작아졌다"고 했으므로, 현재 상태가 1/200 = 0.005 배인 상태일 수 있음.
        # 이는 pitch(0.005)가 두 번 곱해졌거나, 혹은 단위 문제일 수 있음.
        
        # 안전한 복원 방법: 비율 기반 보정
        if scale_ratio < 0.1: # 너무 작으면
            print(f"[Info] 스케일이 너무 작습니다. 비율({1/scale_ratio:.2f})만큼 확대합니다.")
            final_mesh.apply_scale(1.0 / scale_ratio)
            
            # 위치 보정 (중심점 맞추기)
            # 스케일만 키우면 원점 기준으로 커지므로 중심이 틀어질 수 있음.
            # 원본 중심점으로 이동
            current_center = final_mesh.centroid
            translation = raw_center - current_center
            print(f"[Info] 위치 보정을 위해 이동합니다: {translation}")
            final_mesh.apply_translation(translation)
            
        elif scale_ratio > 10.0: # 너무 크면 (이전 문제)
            print(f"[Info] 스케일이 너무 큽니다. 비율({1/scale_ratio:.2f})만큼 축소합니다.")
            final_mesh.apply_scale(1.0 * voxel_pitch)
            
            # 위치 보정
            # current_center = final_mesh.centroid
            # translation = raw_center - current_center
            # print(f"[Info] 위치 보정을 위해 이동합니다: {translation}")
            # final_mesh.apply_translation(translation)

        # 보정 후 다시 확인
        final_bounds_fixed = final_mesh.bounds
        final_center_fixed = final_mesh.centroid
        print(f"[Debug] Fixed Mesh Bounds: {final_bounds_fixed}")
        print(f"[Debug] Fixed Mesh Center: {final_center_fixed}")
        
        # 중심점 차이 확인
        # center_diff = np.linalg.norm(final_center_fixed - raw_center)
        # print(f"[Debug] Center Difference: {center_diff:.6f}")

        # 표면 부드럽게 만들기 (Laplacian Smoothing)
        trimesh.smoothing.filter_laplacian(final_mesh, iterations=2)

        # [중요] PLY 출력을 위한 법선 계산
        final_mesh.fix_normals()

        if final_mesh.is_watertight:
            print("[Success] Watertight Mesh 생성 성공.")
        else:
            print(f"[Warning] 결과 메쉬가 Watertight하지 않습니다. (V={len(final_mesh.vertices)}, F={len(final_mesh.faces)})")

        # PLY로 내보내기
        # PLY 헤더에 단위를 명시하는 표준 방법은 없지만, 주석(comment)으로 남길 수는 있음.
        # 하지만 대부분의 뷰어는 이를 무시함.
        # 대신 obj_info 등을 활용할 수 있으나 trimesh export에서 지원하는지 확인 필요.
        # 여기서는 단순히 바이너리로 저장.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_mesh.export(output_path, file_type='ply', encoding='binary')
        print(f"[Info] 저장 완료: {output_path}")
        return final_mesh

    except Exception as e:
        print(f"[Error] 복셀화 과정 실패: {e}")
        return None


if __name__ == "__main__":
    # --- Configuration ---
    cfg = VoxelConfig(
        object_name="lamp",  # Change this to "fridge" or other keys in yml
        config_file="merge_config.yml",
        voxel_pitch=0.005 # 해상도를 높이면(값을 줄이면) 더 정밀해지지만 구멍이 생길 수도 있음. 적절한 값 필요.
    )
    # ---------------------

    config_path = cfg.config_file
    if not os.path.exists(config_path):
        # Try looking in the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, cfg.config_file)
    
    if not os.path.exists(config_path):
        print(f"[Error] Config file not found: {cfg.config_file}")
        exit(1)

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    object_name = cfg.object_name
    if object_name not in full_config:
        print(f"[Error] Object '{object_name}' not found in configuration file.")
        print(f"Available objects: {list(full_config.keys())}")
        exit(1)

    obj_config = full_config[object_name]
    
    urdf_path = obj_config.get("urdf_path")
    joint_cfg = obj_config.get("joint_config", {})
    output_dir = obj_config.get("output_dir", "../merged")

    # Resolve paths
    base_path = os.getcwd()
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.join(base_path, urdf_path)
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_path, output_dir)
        
    output_filename = f"{object_name}_voxel.ply"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(urdf_path):
        merge_urdf_voxel_ply(urdf_path, joint_cfg, output_path, voxel_pitch=cfg.voxel_pitch)
    else:
        print(f"[Error] URDF file not found: {urdf_path}")
