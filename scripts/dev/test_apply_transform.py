# test_apply_transform_equivalence.py
import os
import numpy as np
import pymeshlab
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

# ====== 설정 ======
INPUT_PLY = "cuce.ply"        # 테스트용 메시 파일 경로 (삼각메시)
OUT_DIR   = "out_test"        # 출력 폴더
INVERT_EULER_SIGN = True     # mesh가 시계로 보일 때 gaussian도 맞추려면 True로 바꿔 테스트

os.makedirs(OUT_DIR, exist_ok=True)

# ====== pymeshlab 적용 함수(원본과 동일 경로) ======
def apply_transform_from_matrix(ms: pymeshlab.MeshSet, matrix_4x4, verbose=True):
    if verbose:
        print("[info] apply_transform_from_matrix called")
    Rmat = matrix_4x4[:3, :3]
    tvec = matrix_4x4[:3, 3]

    rot = R.from_matrix(Rmat)
    euler_angles = rot.as_euler('xyz', degrees=True)  # Mesh는 이 경로를 탑니다.

    for idx in range(ms.mesh_number()):
        ms.set_current_mesh(idx)
        ms.compute_matrix_from_translation_rotation_scale(
            rotationx=euler_angles[0],
            rotationy=euler_angles[1],
            rotationz=euler_angles[2],
            translationx=tvec[0],
            translationy=tvec[1],
            translationz=tvec[2],
            compose=True,
            freeze=True
        )
    if verbose:
        print(f"[mesh] Euler xyz (deg): {euler_angles}")
        print(f"[mesh] Translation    : {tvec}")
    ms.set_current_mesh(0)


# ====== gaussian(PLY) 쪽: mesh와 동일한 Euler 경로로 좌표/노멀만 변환 ======
def apply_like_mesh_to_ply_vertices(input_path: str, output_path: str, matrix_4x4: np.ndarray,
                                    invert_euler_sign: bool = False):
    ply = PlyData.read(input_path)
    v = ply['vertex']
    names = v.data.dtype.names

    Rmat = matrix_4x4[:3, :3]
    tvec = matrix_4x4[:3, 3].astype(np.float64)

    # Mesh와 동일: R -> Euler('xyz', deg) -> 재구성
    euler = R.from_matrix(Rmat).as_euler('xyz', degrees=True)
    if invert_euler_sign:
        euler = -euler
    Rxyz = R.from_euler('xyz', euler, degrees=True).as_matrix()

    if all(k in names for k in ('x', 'y', 'z')):
        P = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
        P_new = (P @ Rxyz.T) + tvec
        v['x'] = P_new[:, 0].astype(v['x'].dtype)
        v['y'] = P_new[:, 1].astype(v['y'].dtype)
        v['z'] = P_new[:, 2].astype(v['z'].dtype)

    if all(k in names for k in ('nx', 'ny', 'nz')):
        N = np.stack([v['nx'], v['ny'], v['nz']], axis=1).astype(np.float64)
        N_new = (N @ Rxyz.T)
        N_new /= (np.linalg.norm(N_new, axis=1, keepdims=True) + 1e-12)
        v['nx'] = N_new[:, 0].astype(v['nx'].dtype)
        v['ny'] = N_new[:, 1].astype(v['ny'].dtype)
        v['nz'] = N_new[:, 2].astype(v['nz'].dtype)

    PlyData([PlyElement.describe(v.data, 'vertex')], text=False).write(output_path)


# ====== 비교 유틸 ======
def rms_max_err(A: np.ndarray, B: np.ndarray):
    """
    A,B: (N,3). 순서가 같다고 가정. 다르면 최근접 매칭 사용.
    """
    if A.shape != B.shape:
        # 최근접 매칭으로 정렬
        tree = cKDTree(B)
        d, idx = tree.query(A, k=1)
        Bm = B[idx]
    else:
        d = np.linalg.norm(A - B, axis=1)
    rms = np.sqrt(np.mean(d**2))
    return float(rms), float(np.max(d))


# ====== 테스트 SE(3) 몇 가지 ======
def make_tests():
    tests = []
    # 1) X90
    Rmat = R.from_euler('x', 90, degrees=True).as_matrix(); T = np.array([0,0,0])
    M = np.eye(4); M[:3,:3] = Rmat; M[:3,3] = T; tests.append(("Rx90", M))
    # 2) Y45 + Tz
    Rmat = R.from_euler('y', 45, degrees=True).as_matrix(); T = np.array([0,0,1])
    M = np.eye(4); M[:3,:3] = Rmat; M[:3,3] = T; tests.append(("Ry45_Tz1", M))
    # 3) Z-30 + T(1,2,3)
    Rmat = R.from_euler('z', -30, degrees=True).as_matrix(); T = np.array([1,2,3])
    M = np.eye(4); M[:3,:3] = Rmat; M[:3,3] = T; tests.append(("Rz-30_T123", M))
    # 4) XYZ 혼합
    Rmat = R.from_euler('xyz', [90, 0, 90], degrees=True).as_matrix(); T = np.array([1,1,1])
    M = np.eye(4); M[:3,:3] = Rmat; M[:3,3] = T; tests.append(("Rxyz_90_0_90_T111", M))
    return tests


if __name__ == "__main__":
    # --- Mesh 경로 로드 ---
    ms = pymeshlab.MeshSet()
    if not os.path.exists(INPUT_PLY):
        raise FileNotFoundError(f"INPUT_PLY not found: {INPUT_PLY}")
    ms.load_new_mesh(INPUT_PLY)

    # 원본 정점 (비교용)
    V0 = ms.current_mesh().vertex_matrix().copy()

    for name, M in make_tests():
        print("="*60)
        print(f"[TEST] {name}")
        print(M)

        # --- Mesh 경로 ---
        # 매번 원본 다시 로드해서 동일 초기 상태에서 시작
        ms.clear()
        ms.load_new_mesh(INPUT_PLY)
        apply_transform_from_matrix(ms, M, verbose=True)
        V_mesh = ms.current_mesh().vertex_matrix().copy()
        mesh_out = os.path.join(OUT_DIR, f"mesh_{name}.ply")
        ms.save_current_mesh(mesh_out)
        print(f"[mesh] saved -> {mesh_out}")

        # --- Gaussian 경로(PLY 직접): 동일 파일 읽어 동일 Euler 경로로 적용 ---
        ply_in  = INPUT_PLY
        ply_out = os.path.join(OUT_DIR, f"gauss_{name}.ply")
        apply_like_mesh_to_ply_vertices(ply_in, ply_out, M, invert_euler_sign=INVERT_EULER_SIGN)

        # 결과 좌표 읽어서 비교(PLY의 vertex 순서가 같다는 가정; 다르면 최근접 매칭)
        Vg = PlyData.read(ply_out)['vertex']
        V_gauss = np.stack([Vg['x'], Vg['y'], Vg['z']], axis=1).astype(np.float64)

        rms, mxe = rms_max_err(V_mesh, V_gauss)
        print(f"[compare] RMS = {rms:.6e}, Max = {mxe:.6e}  (invert_euler_sign={INVERT_EULER_SIGN})")

    # 메모리 정리(IDE에서 반복 실행시 유용)
    try:
        ms.clear()
    except Exception:
        pass
    del ms
    import gc, vedo
    vedo.close()
    gc.collect()