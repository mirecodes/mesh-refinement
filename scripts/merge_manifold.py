import numpy as np
import trimesh
import manifold3d as m3d
from yourdfpy import URDF
import os
import yaml
from dataclasses import dataclass


@dataclass
class MergeConfig:
    # Name of the object to process (must match a key in the YAML config)
    object_name: str = "folding_chair"
    
    # Path to the YAML configuration file
    config_file: str = "merge_config.yml"


def merge_urdf_manifold_ply(urdf_path, joint_cfg, output_path="fused_mesh.ply"):
    print(f"[Info] URDF 로드 중: {urdf_path}")
    robot = URDF.load(urdf_path)

    # 1. Joint Configuration 적용
    valid_cfg = {k: v for k, v in joint_cfg.items() if k in robot.actuated_joint_names}
    robot.update_cfg(configuration=valid_cfg)

    # 2. Scene 객체 가져오기 (수정된 로직)
    scene = robot.scene

    manifolds = []

    for name, node in scene.geometry.items():
        transform = scene.graph.get(name)[0]
        mesh = node.copy()
        mesh.apply_transform(transform)

        try:
            # Watertight 체크 및 수리
            if not mesh.is_watertight:
                # print(f"[Warning] '{name}' 메쉬가 닫혀있지 않아 구멍을 메웁니다.")
                mesh.fill_holes()
                if not mesh.is_watertight:
                    print(f"  -> '{name}' 수리 실패. Convex Hull로 대체합니다.")
                    mesh = mesh.convex_hull

            # 유효성 검사
            if mesh.is_empty:
                print(f"  -> [Skip] '{name}' 메쉬가 비어있습니다 (is_empty).")
                continue
            
            if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
                print(f"  -> [Skip] '{name}' 메쉬 데이터 부족 (V={len(mesh.vertices)}, F={len(mesh.faces)}).")
                continue

            # --- Trimesh -> Manifold 변환 (수정된 로직) ---
            verts = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)
            
            m_mesh = m3d.Mesh(vert_properties=verts, tri_verts=faces)
            m_obj = m3d.Manifold(m_mesh)
            
            if m_obj.num_vert() == 0:
                print(f"  -> [Skip] '{name}' Manifold 변환 결과가 비어있습니다.")
                continue

            manifolds.append(m_obj)
            # ------------------------------------------------

        except Exception as e:
            print(f"[Error] '{name}' 변환 실패: {e}")

    if not manifolds:
        print("[Error] 병합할 유효한 메쉬가 없습니다.")
        return None

    print(f"[Info] {len(manifolds)}개의 파츠를 Boolean Union으로 결합합니다...")

    # 3. Boolean Union 수행
    combined_manifold = manifolds[0]
    for i in range(1, len(manifolds)):
        try:
            combined_manifold += manifolds[i]
        except Exception as e:
            print(f"[Warning] 병합 중 에러 발생 (index {i}): {e}")

    if combined_manifold.num_vert() == 0:
        print("[Error] 최종 병합 결과가 비어있습니다 (Vertices=0).")
        return None

    # 4. 결과물 변환 및 PLY 저장 설정
    out_mesh = combined_manifold.to_mesh()
    final_trimesh = trimesh.Trimesh(
        vertices=np.array(out_mesh.vert_properties),
        faces=np.array(out_mesh.tri_verts)
    )

    # [중요] PLY 출력을 위해 법선(Normal) 재계산 (부드러운 쉐이딩을 위해)
    final_trimesh.fix_normals()

    if final_trimesh.is_watertight:
        print("[Success] 완벽한 Watertight Mesh가 생성되었습니다.")
    else:
        print(f"[Warning] 결과 메쉬가 Watertight하지 않습니다. (V={len(final_trimesh.vertices)}, F={len(final_trimesh.faces)})")

    if final_trimesh.is_empty:
         print("[Error] 최종 Trimesh가 비어있습니다.")
         return None

    # PLY로 내보내기 (binary=True는 파일 용량을 줄임, 텍스트로 보려면 False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_trimesh.export(output_path, file_type='ply', encoding='binary')
    print(f"[Info] 저장 완료: {output_path}")
    return final_trimesh


if __name__ == "__main__":
    # --- Configuration ---
    cfg = MergeConfig(
        object_name="scissors",  # Change this to "fridge" or other keys in yml
        config_file="merge_config.yml"
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
        
    output_filename = f"{object_name}_fused.ply"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(urdf_path):
        merge_urdf_manifold_ply(urdf_path, joint_cfg, output_path)
    else:
        print(f"[Error] URDF file not found: {urdf_path}")
