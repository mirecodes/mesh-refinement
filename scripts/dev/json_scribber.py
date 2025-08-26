import json
import os

# 데이터를 저장할 JSON 파일 경로를 지정합니다.
file_path = 'process_data.json'

# --- 1단계: 초기 데이터 생성 및 JSON 파일로 저장 ---
# 프로그램이 시작될 때 필요한 초기 데이터를 딕셔너리 형태로 정의합니다.
initial_data = {
    "process_id": 12345,
    "status": "initialized",
    "item_count": 0,
    "records": []
}

print("--- 1단계: 초기 데이터 파일에 저장 시작 ---")
try:
    # 'w'(쓰기) 모드로 파일을 엽니다. 파일이 없으면 새로 생성됩니다.
    # encoding='utf-8' : 한글이 깨지지 않도록 UTF-8 인코딩을 사용합니다.
    # ensure_ascii=False : 한글을 유니코드(\uXXXX)가 아닌 그대로 저장합니다.
    # indent=4 : JSON 파일을 사람이 읽기 쉽게 4칸 들여쓰기로 포맷팅합니다.
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, ensure_ascii=False, indent=4)

    print(f"'{file_path}'에 초기 데이터를 성공적으로 저장했습니다.")
    print("저장된 내용:")
    print(json.dumps(initial_data, ensure_ascii=False, indent=4))
    print("-" * 30)

except IOError as e:
    print(f"파일 쓰기 중 오류 발생: {e}")

# --- 2단계: 파일에서 데이터 읽어와서 수정 ---
# 프로세스 중간에 파일의 데이터를 수정해야 하는 상황을 가정합니다.
print("\n--- 2단계: 데이터 읽기 및 수정 시작 ---")

# 파일이 존재하는지 먼저 확인하는 것이 안전합니다.
if os.path.exists(file_path):
    try:
        # 'r'(읽기) 모드로 파일을 엽니다.
        with open(file_path, 'r', encoding='utf-8') as f:
            # json.load() 함수를 사용해 파일에서 데이터를 파이썬 딕셔너리로 불러옵니다.
            data_from_file = json.load(f)

        print(f"'{file_path}'에서 데이터를 성공적으로 읽었습니다.")
        print("수정 전 데이터:")
        print(json.dumps(data_from_file, ensure_ascii=False, indent=4))

        # 데이터 수정
        data_from_file['status'] = 'in_progress'
        data_from_file['item_count'] += 1
        new_record = {
            "record_id": "A-001",
            "timestamp": "2025-08-26T13:30:00",
            "message": "첫 번째 기록 추가"
        }
        data_from_file['records'].append(new_record)

        print("\n메모리에서 데이터가 다음과 같이 수정되었습니다:")
        print(json.dumps(data_from_file, ensure_ascii=False, indent=4))
        print("-" * 30)

        # --- 3단계: 수정된 데이터를 다시 파일에 덮어쓰기 ---
        print("\n--- 3단계: 수정된 데이터 파일에 덮어쓰기 시작 ---")
        try:
            # 다시 'w'(쓰기) 모드로 파일을 열어 수정된 데이터로 전체 파일을 덮어씁니다.
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_from_file, f, ensure_ascii=False, indent=4)

            print(f"'{file_path}'에 수정된 데이터를 성공적으로 덮어썼습니다.")

        except IOError as e:
            print(f"파일 덮어쓰기 중 오류 발생: {e}")

    except (IOError, json.JSONDecodeError) as e:
        print(f"파일 읽기 또는 JSON 파싱 중 오류 발생: {e}")
else:
    print(f"오류: '{file_path}' 파일이 존재하지 않습니다.")