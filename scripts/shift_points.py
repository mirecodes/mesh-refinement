import os

import numpy as np
from plyfile import PlyData, PlyElement

def translate_ply(input_path: str, output_path: str, offset: tuple):
    """
    .ply 파일의 모든 정점(vertex) 좌표를 주어진 오프셋만큼 이동시키고,
    이동 전/후의 평균 좌표를 출력합니다.

    Args:
        input_path (str): 원본 .ply 파일 경로
        output_path (str): 좌표 이동 후 저장할 .ply 파일 경로
        offset (tuple): (x_offset, y_offset, z_offset) 형태의 튜플
    """
    try:
        # 1. PlyData.read() 함수로 .ply 파일을 읽습니다.
        print(f"'{input_path}' 파일을 읽는 중...")
        plydata = PlyData.read(input_path)

        # 2. 'vertex' 요소가 있는지 확인합니다.
        if 'vertex' not in plydata:
            print("오류: 파일에 'vertex' 요소가 없습니다.")
            return

        vertices = plydata['vertex']

        # 3. (추가) 좌표 이동 전 평균 위치를 계산하고 출력합니다.
        #    np.mean() 함수로 각 좌표 배열의 평균을 계산합니다.
        avg_x_before = np.mean(vertices['x'])
        avg_y_before = np.mean(vertices['y'])
        avg_z_before = np.mean(vertices['z'])
        print("-" * 30)
        print(f"좌표 이동 전 평균 위치:")
        print(f"  Avg X: {avg_x_before:.6f}")
        print(f"  Avg Y: {avg_y_before:.6f}")
        print(f"  Avg Z: {avg_z_before:.6f}")
        print("-" * 30)

        # 4. 오프셋 값을 가져옵니다.
        x_offset, y_offset, z_offset = offset
        print(f"적용할 오프셋: (x: {x_offset}, y: {y_offset}, z: {z_offset})")

        # 5. 'vertex' 데이터에 접근하여 좌표를 수정합니다.
        vertices['x'] -= x_offset
        vertices['y'] -= y_offset
        vertices['z'] -= z_offset
        # vertices['x'] -= avg_x_before
        # vertices['y'] -= avg_y_before
        # vertices['z'] -= avg_z_before
        print("모든 정점의 좌표 이동을 완료했습니다.")

        # 6. (추가) 좌표 이동 후 평균 위치를 계산하고 출력합니다.
        avg_x_after = np.mean(vertices['x'])
        avg_y_after = np.mean(vertices['y'])
        avg_z_after = np.mean(vertices['z'])
        print("-" * 30)
        print(f"좌표 이동 후 평균 위치:")
        print(f"  Avg X: {avg_x_after:.6f}")
        print(f"  Avg Y: {avg_y_after:.6f}")
        print(f"  Avg Z: {avg_z_after:.6f}")
        print("-" * 30)

        # 7. 수정된 데이터로 새 .ply 파일을 작성합니다.
        print(f"수정된 데이터를 '{output_path}' 파일에 저장하는 중...")
        plydata.write(output_path)

        print("작업이 성공적으로 완료되었습니다!")

    except FileNotFoundError:
        print(f"오류: '{input_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


# --- 메인 실행 부분 ---
if __name__ == "__main__":

    base_dir = os.getcwd()

    # --- 여기를 수정하세요 ---

    # 1. 원본 .ply 파일 경로
    input_file = os.path.abspath(os.path.join(base_dir, "..", "out", "transfer", "spray_gaussian_6.ply"))

    # 2. 결과물을 저장할 .ply 파일 경로
    output_file = os.path.abspath(os.path.join(base_dir, "..", "out", "shifted_meshes", "shifted_spray_gaussian_6.ply"))

    # 3. 이동시킬 기준 좌표 (오프셋 벡터)
    offset_vector = (0.639987, 1.448243, 1.526706)

    # -------------------------

    # 함수 호출
    translate_ply(input_file, output_file, offset_vector)