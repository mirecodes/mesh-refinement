import numpy as np
import trimesh
from yourdfpy import URDF
import os


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
        return None

    print("[Info] 메쉬들을 하나의 덩어리로 뭉칩니다 (Concatenate)...")
    combined_raw = trimesh.util.concatenate(all_meshes)

    # 3. Voxelization & Remeshing
    print(f"[Info] 복셀화 및 재구성 진행 중 (Pitch: {voxel_pitch}m)...")

    try:
        voxel_grid = combined_raw.voxelized(pitch=voxel_pitch)
        voxel_grid.fill()
        final_mesh = voxel_grid.marching_cubes

        # 표면 부드럽게 만들기 (Laplacian Smoothing)
        trimesh.smoothing.filter_laplacian(final_mesh, iterations=2)

        # [중요] PLY 출력을 위한 법선 계산
        final_mesh.fix_normals()

        if final_mesh.is_watertight:
            print("[Success] Watertight Mesh 생성 성공.")

        # PLY로 내보내기
        final_mesh.export(output_path, file_type='ply', encoding='binary')
        print(f"[Info] 저장 완료: {output_path}")
        return final_mesh

    except Exception as e:
        print(f"[Error] 복셀화 과정 실패: {e}")
        return None


# --- 실행 예시 ---
if __name__ == "__main__":
    my_urdf = "faucet/mobility.urdf"
    my_config = {
        'joint_0': 0.0,
        'joint_1': 0,
        'joint_2': 0,
    }

    import os

    if os.path.exists(my_urdf):
        # 5mm 해상도로 복셀화
        merge_urdf_voxel_ply(my_urdf, my_config, voxel_pitch=0.005)