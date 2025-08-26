# JsonHandler 라이브러리에서 필요한 클래스를 가져옵니다.
from json_handler import JsonHandler

# --- 예제 1: 파일 접근 최소화 방식 (auto_save=False, 권장) ---
print("--- 예제 1: 수동 저장으로 파일 접근 최소화 ---")

# auto_save=False로 설정하여 변경 시 자동 저장을 비활성화합니다.
# 파일이 없으면 초기 데이터(default_data)를 사용해 생성됩니다.
config = JsonHandler(
    filename='app_data/settings.json',
    default_data={'version': '1.0', 'user': None},
    auto_save=False
)

print(f"초기 설정: {config}")

# 여러 데이터를 순차적으로 변경합니다.
# 이 시점에서는 메모리에서만 데이터가 변경되고 파일 I/O는 발생하지 않습니다.
config.version = '1.1.0'
config.user = 'mireflare'
config.window = {  # 객체처럼 새로운 중첩 딕셔너리 추가
    'width': 1920,
    'height': 1080
}
config.recent_files = [] # 리스트 추가
config.recent_files.append('file1.txt') # 리스트 수정

print(f"수정된 설정: {config}")
print("... 여러 항목을 변경했지만 아직 파일에 저장되지 않았습니다 ...")

# 모든 변경이 끝난 후, 원하는 시점에 .save()를 호출하여 파일에 한 번만 씁니다.
config.save()
print(f"'{config.filename}'에 모든 변경사항을 한 번에 저장했습니다.")


# --- 예제 2: 모든 변경마다 자동 저장 (auto_save=True) ---
print("\n--- 예제 2: 자동 저장으로 매번 파일에 기록 ---")

# auto_save=True로 설정하면 모든 변경이 즉시 파일에 반영됩니다.
stats = JsonHandler(
    filename='app_data/stats.json',
    default_data={'login_count': 0},
    auto_save=True
)

print(f"초기 통계: {stats}")

# 아래 각 라인이 실행될 때마다 파일 쓰기가 발생합니다.
print("로그인 횟수 증가 (파일 쓰기 1회 발생)")
stats.login_count += 1

print("마지막 접속 시간 기록 (파일 쓰기 2회 발생)")
stats.last_login = '2025-08-26 13:18' # 현재 시간으로 가정

print(f"최종 통계: {stats}")
print(f"'{stats.filename}' 파일은 총 2번 업데이트 되었습니다.")