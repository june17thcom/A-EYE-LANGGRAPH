# tools.py
import os
import logging
from itertools import count

import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
import os
import logging
from typing import Dict, Any
from langchain_core.tools import tool
from temp_img import image_to_base64  # 로컬 테스트용
load_dotenv()
log = logging.getLogger("wearable.tools")

IMAGE_API_BASE = os.getenv("IMAGE_API_BASE", "http://127.0.0.1:8000")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 이미지 전처리(서버단 오류 잡기용)
MAX_PAYLOAD_SIZE = 120_000  # command + base64 총합 제한 (byte 기준)
MIN_DIM = 64  # 최소 리사이징 크기

def safe_base64_for_llm(b64_str: str, command: str) -> str:
    """
    1. image_lookup 툴에서 서버 요청 후 받은 결과(res)를 확인
    2. base64 + 사용자 쿼리 길이가 API 처리 제한 토큰 수(124k) 이상이면
    3. 그대로 LLM에 전달하지 말고, 이미지 크기를 줄여
    (디코딩 후 압축resizing, 다시 한 번 base64 인코딩: 평균 1.33배 가량 용량이 늘어나므로 기준은 img = resized_base64 * 1.33 < 124k - command)
    안전하게 토큰 제한 내로 맞춤
    """
    from PIL import Image
    import io, base64

    # 1) 예상 크기 확인
    total_len = len(command.encode("utf-8")) + len(b64_str.encode("utf-8"))
    if total_len <= MAX_PAYLOAD_SIZE:
        return b64_str

    # 2) base64 → 이미지 디코딩
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data))

    # 3) 반복 리사이즈: payload < MAX_PAYLOAD_SIZE
    while total_len > MAX_PAYLOAD_SIZE:
        w, h = img.size
        if w <= MIN_DIM or h <= MIN_DIM:
            break  # 최소 크기 이하이면 강제 종료
        img = img.resize((w // 2, h // 2))  # 1/2 리사이즈
        # 재인코딩
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        total_len = len(command.encode("utf-8")) + len(b64_str.encode("utf-8"))

    return b64_str


# 임시 - 삭제 예정
from temp_img import image_to_base64

@tool("image_lookup")
def image_lookup(query: str) -> Dict[str, Any]:
    """
    [목적]
    - 내부 FastAPI의 /image/search 엔드포인트를 호출해 로컬/캐시된 이미지 목록을 받아와
      '음성만으로' 요약/설명할 때 사용한다. (분석/설명용 우선순위 도구)
    - 사용자의 요청이 '이미지'와 연관되면 이 도구를 우선 고려한다.

    [트리거 예시]
    - "이미지/사진/그림/프레임/캡처/스크린샷 보여줘·찾아줘·분석해"
    - "OOO 사진으로 설명해줘", "최근 프레임에 뭐가 보여?" 등
    - YOLO/시선추적/깊이맵 결과와 결합해 '장면 설명'이 필요할 때

    [입력]
    - query: 키워드(짧게). 예) "crosswalk", "traffic light", "pedestrian"
      * 결과가 너무 많으면 더 구체적인 키워드로 재시도한다.

    [동작]
    - GET {IMAGE_API_BASE}/image/search?q={query}
    - 응답 스키마:
      {
        "items": [
          {"filename": str, "data_uri": str},  # data:...;base64, 대용량
          ...
        ]
      }

    [응답 사용 가이드(중요)]
    - 사용자는 화면을 보지 못하므로:
      1) 파일명/경로/data URI/원시 JSON을 '읽지 말 것'.
      2) 이미지에서 유추 가능한 핵심을 1문장 요약 + 최대 3포인트로 간결히 말할 것.
      3) 여러 장이면 공통점/차이점/추세만 말로 묶어 설명(최대 3포인트).
      4) 민감 정보(얼굴 등)는 일반화하여 설명하고, 불확실하면 단정 금지.

    [외부 사실 보강]
    - 이미지 자체 설명에 '외부 사실'이 필요하면 web_search 도구로 2~3개 출처를 조회해
      '출처 이름만' 말로 덧붙인다(링크 낭독 금지).

    [빈 결과]
    - items가 비면 "관련 이미지를 찾지 못했다"고 짧게 알리고,
      더 구체적 키워드 제안(예: '신호등' → '빨간 신호등') 후 재시도 유도.
      (추후 /image/push, /image/recent 도입 시 그 경로로 업링크/최근 프레임을 안내)

    [반환값]
    - dict (그대로 LLM에게 전달): {"items": [{"filename":..., "data_uri":...}, ...]}
      * 로깅 시 data_uri는 길 수 있으므로 길이만 요약/마스킹 권장.
    """
    '''
    log.debug(f"[image_lookup] q='{query}' -> GET {IMAGE_API_BASE}/image/search")
    r = requests.get(f"{IMAGE_API_BASE}/image/search", params={"q": query}, timeout=15)
    r.raise_for_status()
    data = r.json()
    count = len(data.get("items", []))
    log.info(f"[image_lookup] q='{query}' | items={count}")
    return data
    '''
    print("이미지 연산 시작: 로컬 base64 테스트 모드")
    """
            멀티모달 LLM와 연계 가능한 이미지 조회/연산용 툴.

            동작:
            1) 서버 IMAGE_API_BASE에 GET 요청하여 BASE64 이미지 획득
            2) 실패 시 로컬 이미지 사용
            3) 반환값은 LLM에서 바로 처리 가능하도록 dict 구조
            """
    try:
        # --- 서버 호출 버전 ---
        # log.debug(f"[image_lookup] q='{query}' -> GET {IMAGE_API_BASE}/image/search")
        # r = requests.get(f"{IMAGE_API_BASE}/image/search", params={"q": query}, timeout=15)
        # r.raise_for_status()
        # data = r.json()
        # return data
        '''
        try:
        # --- 서버 호출 버전 ---
        log.debug(f"[image_lookup] q='{query}' -> GET {IMAGE_API_BASE}/image/search")
        r = requests.get(f"{IMAGE_API_BASE}/image/search", params={"q": query}, timeout=15)
        r.raise_for_status()
        data = r.json()

        # items 안에 data_uri가 들어온다고 가정
        items = data.get("items", [])
        for item in items:
            if "data_uri" in item and item["data_uri"].startswith("data:image"):
                # 헤더 분리
                header, b64_str = item["data_uri"].split(",", 1)
                safe_b64 = safe_base64_for_llm(b64_str, query)
                item["data_uri"] = f"{header},{safe_b64}"

        log.info(f"[image_lookup] q='{query}' | items={len(items)} (서버)")
        return {"query": query, "items": items}
        '''

        # --- 로컬 테스트 버전 ---
        log.info("[image_lookup] 서버 연결 실패, 로컬 이미지 테스트 모드")
        b64_str = image_to_base64(
            "C:\\Users\\ljcho\\Downloads\\A-EYE-LANGGRAPH\\images\\지하철일상.jpg"
        )
        command_text = f"사용자 질문: {query}"
        print(command_text)

        b64_str_safe = safe_base64_for_llm(b64_str, command_text)
        print(b64_str_safe[:40])
        data = {
            "items": [
                {
                    "filename": "지하철일상.jpg",
                    "data_uri": f"data:image/jpeg;base64,{b64_str_safe}"
                }
            ]
        }
        return {"query": query, "items": data["items"]}

    except Exception as e:
        log.exception(f"[image_lookup] 이미지 처리 실패: {e}")
        return {"items": []}


@tool("web_search")
def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    최신/정확한 사실 확인, 정의, 수치, 절차가 필요할 때 사용.
    입력: query(한/영), max_results(기본 5)
    답변에는 2~3개 출처 '이름만' 언급(링크 낭독 금지).
    """
    results: List[Dict[str, str]] = []
    if not TAVILY_API_KEY:
        log.warning("[web_search] no TAVILY_API_KEY; return empty results")
        return results
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        log.debug(f"[web_search] query='{query}' | max_results={max_results}")
        data = client.search(query=query, max_results=max_results)
        for item in data.get("results", []):
            results.append({
                "source": item.get("source", "") or item.get("url", ""),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": (item.get("content", "") or "")[:300],
            })
        log.info(f"[web_search] hits={len(results)} for query='{query}'")
        return results
    except Exception as e:
        log.exception(f"[web_search] exception: {e}")
        return results

@tool("weather_now")
def weather_now(lat: float, lon: float) -> dict:
    """
    사용자가 '날씨/기온/비/바람' 등을 물으면 호출. 좌표(lat, lon) 필수.
    Open-Meteo API로 현재 상태와 시간별 강수·기온 요약을 가져온다.
    응답을 말로 전달할 때는 수치를 반올림해 섭씨/미터초 기준으로 간결히 안내.
    """
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "hourly": "temperature_2m,precipitation_probability",
            "timezone": "auto",
        },
        timeout=12,
    )
    r.raise_for_status()
    return r.json()

def get_tools():
    tools = [image_lookup]
    if TAVILY_API_KEY:
        tools.insert(0, web_search)
    tools.append(weather_now)
    return tools
