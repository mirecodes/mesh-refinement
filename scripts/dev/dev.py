import vedo
from vedo import Mesh, Plotter

# decomposition 결과 예시: mesh_parts (Mesh 객체 리스트)
mesh_parts = [
    Mesh(vedo.shapes.Cube().scale(0.5).pos([0,0,0])).c("lightblue", 0.5),
    Mesh(vedo.shapes.Sphere().scale(0.4).pos([1,0,0])).c("lightblue", 0.5),
    Mesh(vedo.shapes.Cylinder().scale(0.3).pos([-1,0,0])).c("lightblue", 0.5),
]

# 선택된 파트를 저장할 배열
selected_parts = { "part1": [], "part2": [] }
current_group = "part1"   # 기본적으로 part1에 저장

def on_pick(event):
    """사용자가 mesh를 클릭했을 때 호출"""
    print("Hello")
    picked = event.actor
    if picked is None:
        return
    if picked in mesh_parts:
        selected_parts[current_group].append(picked)
        picked.c("green")  # 클릭하면 색을 임시로 바꿔줌
        plt.render()
        print(f"[INFO] Added mesh to {current_group}")

def switch_group():
    """버튼 눌러서 저장 그룹 전환"""
    global current_group
    current_group = "part2" if current_group == "part1" else "part1"
    print(f"[INFO] Now selecting for {current_group}")

def finalize_selection():
    """선택한 파트를 다른 색으로 칠해 최종 렌더링"""
    for m in selected_parts["part1"]:
        m.c("red", 0.7)
    for m in selected_parts["part2"]:
        m.c("blue", 0.7)
    plt.render()
    print("[INFO] Finalized: part1=red, part2=blue")

# GUI 생성
plt = Plotter(title="Click to select parts")
plt.add_callback("mouse_click", on_pick)

# 그룹 전환 버튼
plt.add_button(switch_group, pos=(0.7,0.05), states=["Switch Group"], size=25, c="black", bc="orange")

# 최종 색칠 버튼
plt.add_button(finalize_selection, pos=(0.85,0.05), states=["Finalize"], size=25, c="black", bc="lightgreen")

plt.show(mesh_parts, axes=1)