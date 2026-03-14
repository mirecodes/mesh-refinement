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
        # voxelized()는 메쉬의 Bounding Box를 기준으로 로컬 그리드를 생성합니다.
        # 따라서 그리드의 원점은 (0,0,0)이 아니라 mesh.bounds[0] 근처가 됩니다.
        voxel_grid = combined_raw.voxelized(pitch=voxel_pitch)
        voxel_grid.fill() 
        
        # Marching Cubes로 메쉬 재구성
        final_mesh = voxel_grid.marching_cubes

        # [Fix] 좌표계 복원 (Origin & Scale)
        # voxel_grid.transform은 (Index 좌표 -> World 좌표) 변환 행렬입니다.
        # 여기에는 Scale(pitch)과 Translation(origin)이 모두 포함되어 있습니다.
        matrix = voxel_grid.transform
        grid_origin = matrix[:3, 3]
        print(f"[Debug] Voxel Grid Origin (Translation): {grid_origin}")
        
        # 현재 메쉬의 스케일 상태 확인
        current_size = final_mesh.extents
        raw_size = raw_bounds[1] - raw_bounds[0]
        ratio = np.mean(current_size / (raw_size + 1e-9))
        print(f"[Debug] Mesh/Raw Size Ratio: {ratio:.2f}")

        if ratio > 10.0:
            # marching_cubes 결과가 Index 좌표계(정수)인 경우
            # Transform 행렬을 전체 적용하여 Scale과 Origin을 모두 복원합니다.
            print("[Info] Index 좌표계로 감지됨 -> Transform 행렬 전체 적용")
            final_mesh.apply_transform(matrix)
        else:
            # marching_cubes 결과가 이미 Scale은 적용되었으나(미터 단위),
            # 위치가 Local(0,0,0 기준)인 경우입니다.
            # 이 경우 Grid의 Origin 만큼 이동시켜주면 됩니다.
            
            # 현재 위치가 원본과 얼마나 차이나는지 확인
            dist = np.linalg.norm(final_mesh.centroid - raw_center)
            if dist > voxel_pitch:
                print(f"[Info] 위치 보정을 위해 Grid Origin({grid_origin}) 만큼 이동합니다.")
                final_mesh.apply_translation(grid_origin)
            else:
                print("[Info] 위치가 이미 원본과 일치합니다.")

        # 보정 후 최종 확인
        final_center_fixed = final_mesh.centroid
        print(f"[Debug] Fixed Mesh Center: {final_center_fixed}")
        
        # 표면 부드럽게 만들기 (Laplacian Smoothing)
        trimesh.smoothing.filter_laplacian(final_mesh, iterations=2)

        # [중요] PLY 출력을 위한 법선 계산
        final_mesh.fix_normals()

        if final_mesh.is_watertight:
            print("[Success] Watertight Mesh 생성 성공.")
        else:
            print(f"[Warning] 결과 메쉬가 Watertight하지 않습니다. (V={len(final_mesh.vertices)}, F={len(final_mesh.faces)})")

        # PLY로 내보내기
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
        object_name="usb",  # Change this to "fridge" or other keys in yml
        config_file="merge_config.yml",
        voxel_pitch=0.01 # 해상도를 높이면(값을 줄이면) 더 정밀해지지만 구멍이 생길 수도 있음. 적절한 값 필요.
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
