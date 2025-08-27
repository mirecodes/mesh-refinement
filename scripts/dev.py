import numpy as np
import pymeshlab as ml
from scipy.spatial.transform import Rotation as R

# ---- 유틸: 현재 MeshSet의 (R_eff, t_eff) 뽑기 ----
# 원점 O=(0,0,0)과 기준축 e1,e2,e3를 버텍스로 가진 아주 작은 메쉬에서,
# 변환 후 점들 O', X', Y', Z'를 읽어 R_eff와 t_eff를 재구성합니다.
def make_basis_meshset():
    ms = ml.MeshSet()
    V = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)  # O, X, Y, Z
    F = np.array([[0,1,2]], dtype=int)
    ms.add_mesh(ml.Mesh(vertex_matrix=V, face_matrix=F))
    return ms

def get_vertices(ms):
    m = ms.current_mesh()
    return m.vertex_matrix().copy()  # (4,3): O', X', Y', Z'

def RT_from_vertices(Vp):
    O = Vp[0]
    Xp = Vp[1]; Yp = Vp[2]; Zp = Vp[3]
    # R의 각 열 = (축점 - 원점)
    R_eff = np.column_stack([Xp - O, Yp - O, Zp - O])
    t_eff = O
    return R_eff, t_eff

# ---- PyMeshLab 변환 적용: rotationx/y/z, translationx/y/z ----
def apply_pymeshlab_trs(ms, rx, ry, rz, tx=0.0, ty=0.0, tz=0.0):
    ms.set_current_mesh(0)
    ms.compute_matrix_from_translation_rotation_scale(
        rotationx=rx, rotationy=ry, rotationz=rz,
        translationx=tx, translationy=ty, translationz=tz,
        freeze=True
    )

# --- 수정: 단일 회전 규약 검증을 임의 시퀀스로 ---
def test_single_rotation(rx, ry, rz, seq='xyz'):
    ms = make_basis_meshset()
    apply_pymeshlab_trs(ms, rx, ry, rz, 0, 0, 0)
    R_eff, t_eff = RT_from_vertices(get_vertices(ms))

    R_intr = R.from_euler(seq.lower(), [rx, ry, rz], degrees=True).as_matrix()
    R_extr = R.from_euler(seq.upper(), [rx, ry, rz], degrees=True).as_matrix()

    print(f"[Single:{seq}] t_eff:", t_eff)
    print(f"match intrinsic {seq.lower()}:", np.allclose(R_eff, R_intr, atol=1e-9))
    print(f"match extrinsic {seq.upper()}:", np.allclose(R_eff, R_extr, atol=1e-9))
    return R_eff, t_eff

# --- 수정: 합성 규약 검증도 시퀀스 사용 ---
def test_composition(r1, t1, r2, t2, seq='xyz'):
    # A: 순차 적용 (T1 후 T2)
    msA = make_basis_meshset()
    apply_pymeshlab_trs(msA, *r1, *t1)
    apply_pymeshlab_trs(msA, *r2, *t2)
    RA, tA = RT_from_vertices(get_vertices(msA))

    R1 = R.from_euler(seq.lower(), r1, degrees=True).as_matrix()
    R2 = R.from_euler(seq.lower(), r2, degrees=True).as_matrix()
    t1v = np.array(t1); t2v = np.array(t2)

    # 좌곱(column): T = T2 @ T1 → R = R2 R1, t = R2 t1 + t2
    R_col = R2 @ R1
    t_col = R2 @ t1v + t2v

    # 우곱(row): T = T1 @ T2 → R = R1 R2, t = t1 R2 + t2
    R_row = R1 @ R2
    t_row = (t1v @ R2) + t2v

    # 한 번에 적용 (PyMeshLab 경로와 동일하게 각/병진 인자로)
    msB_col = make_basis_meshset()
    rx, ry, rz = R.from_matrix(R_col).as_euler(seq.lower(), degrees=True)
    apply_pymeshlab_trs(msB_col, rx, ry, rz, *t_col)
    RBc, tBc = RT_from_vertices(get_vertices(msB_col))

    msB_row = make_basis_meshset()
    rx, ry, rz = R.from_matrix(R_row).as_euler(seq.lower(), degrees=True)
    apply_pymeshlab_trs(msB_row, rx, ry, rz, *t_row)
    RBr, tBr = RT_from_vertices(get_vertices(msB_row))

    print(f"[Compose:{seq}] match column (T = T2@T1):",
          np.allclose(RA, RBc, atol=1e-9) and np.allclose(tA, tBc, atol=1e-9))
    print(f"[Compose:{seq}] match row    (T = T1@T2):",
          np.allclose(RA, RBr, atol=1e-9) and np.allclose(tA, tBr, atol=1e-9))
    return (RA, tA), (RBc, tBc), (RBr, tBr)

if __name__ == "__main__":
    if __name__ == "__main__":
        # 단일 회전: zyx / ZYX 둘 다 확인
        test_single_rotation(30, 20, 10, seq='zyx')  # intrinsic zyx
        test_single_rotation(30, 20, 10, seq='ZYX')  # extrinsic ZYX

        # 합성 규약: zyx 기준으로 비교 (t2=0 케이스 포함)
        test_composition(
            r1=(15, 0, 25), t1=(0.2, -0.1, 0.0),
            r2=(0, 35, 0), t2=(0.0, 0.0, 0.0),
            seq='zyx'
        )