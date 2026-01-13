import numpy as np
import trimesh
import manifold3d as m3d
from yourdfpy import URDF
import os


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

            # --- Trimesh -> Manifold 변환 (수정된 로직) ---
            verts = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)
            m_mesh = m3d.Mesh(vert_properties=verts, tri_verts=faces)
            m_obj = m3d.Manifold(m_mesh)
            manifolds.append(m_obj)
            # ------------------------------------------------

        except Exception as e:
            print(f"[Error] '{name}' 변환 실패: {e}")

    if not manifolds:
        print("[Error] 병합할 메쉬가 없습니다.")
        return None

    print(f"[Info] {len(manifolds)}개의 파츠를 Boolean Union으로 결합합니다...")

    # 3. Boolean Union 수행
    combined_manifold = manifolds[0]
    for i in range(1, len(manifolds)):
        combined_manifold += manifolds[i]

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

    # PLY로 내보내기 (binary=True는 파일 용량을 줄임, 텍스트로 보려면 False)
    final_trimesh.export(output_path, file_type='ply', encoding='binary')
    print(f"[Info] 저장 완료: {output_path}")
    return final_trimesh


if __name__ == "__main__":
    my_urdf = "urdf/toilet/mobility.urdf"

    # 테스트용 설정 (필요시 값 조절)
    my_config = {
        'joint_0': 0.0,
        'joint_1': np.pi / 3,
        'joint_2': np.pi / 6,
        'joint_3': 0.0,
    }

    if os.path.exists(my_urdf):
        merge_urdf_manifold_ply(my_urdf, my_config)
    else:
        print(f"파일을 찾을 수 없습니다: {my_urdf}")