import base64
import os
from typing import Optional


def image_to_base64(image_path: str) -> Optional[str]:
    """
    로컬 이미지 파일을 읽어 Base64 문자열로 인코딩하여 반환합니다.
    파일을 찾을 수 없거나 읽기 오류가 발생하면 None을 반환합니다.

    :param image_path: 인코딩할 이미지 파일의 절대 또는 상대 경로
    :return: Base64로 인코딩된 문자열 (str) 또는 오류 시 None
    """
    print("이미지 전처리 시작")
    if not os.path.exists(image_path):
        print(f"오류: 파일 '{image_path}'를 찾을 수 없습니다.")
        return None

    try:
        with open(image_path, "rb") as image_file:
            # 파일을 바이너리(rb) 모드로 읽어 Base64로 인코딩
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except IOError as e:
        print(f"오류: 파일 '{image_path}'를 읽는 중 오류가 발생했습니다. {e}")
        return None

'''
# --- 사용 예시 ---
if __name__ == "__main__":
    # 인코딩할 이미지 파일의 경로를 여기에 지정하세요.
    # 예: 'C:/Users/ljcho/Downloads/multimodal/test_image.jpg'
    # 이 코드를 실행하기 전에 해당 경로에 실제 이미지가 있는지 확인해 주세요.
    file_path = "C:\\Users\\ljcho\\Downloads\\multimodal\\loader\\temp_img.py"

    base64_image_data = image_to_base64(file_path)

    if base64_image_data:
        print("이미지를 Base64로 변환하는 데 성공했습니다.")
        # 변환된 Base64 문자열의 일부만 출력하여 확인
        print(f"Base64 데이터 (일부): {base64_image_data[:50]}...")

        # 이 Base64 문자열을 API 호출 등의 input으로 사용할 수 있습니다.
        # 예: {"image_base64": base64_image_data, "user_input": "이거 뭐야?"}
    else:
        print("이미지 변환에 실패했습니다. 경로를 확인해 주세요.")
'''