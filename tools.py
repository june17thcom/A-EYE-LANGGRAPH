# tools.py
import os
import logging
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
log = logging.getLogger("wearable.tools")

IMAGE_API_BASE = os.getenv("IMAGE_API_BASE", "http://127.0.0.1:8000")
# YOLO/GAZE 업링크 엔드포인트도 동일 FastAPI 안에 있으므로 기본은 IMAGE_API_BASE와 동일
PERCEPTION_API_BASE = os.getenv("PERCEPTION_API_BASE", IMAGE_API_BASE)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


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
      (추후 /image/push, /image/recent 경로도 함께 안내 가능)

    [반환값]
    - dict: {"items": [{"filename":..., "data_uri":...}, ...]}
    """
    log.debug(f"[image_lookup] q='{query}' -> GET {IMAGE_API_BASE}/image/search")
    r = requests.get(f"{IMAGE_API_BASE}/image/search", params={"q": query}, timeout=15)
    r.raise_for_status()
    data = r.json()
    count = len(data.get("items", []))
    log.info(f"[image_lookup] q='{query}' | items={count}")
    return data


@tool("image_recent")
def image_recent(session_id: str, limit: int = 1) -> Dict[str, Any]:
    """
    [목적] 디바이스가 /image/push로 업로드한 '최근 프레임'을 가져옴.
    [동작] GET {IMAGE_API_BASE}/image/recent?session_id=...&limit=...
    [응답] {"items":[{"filename","data_uri"}, ...]}
    [사용] 파일명/URI 낭독 금지. 장면만 말로 요약(1문장+최대3포인트).
    """
    log.debug(f"[image_recent] session_id='{session_id}', limit={limit}")
    r = requests.get(
        f"{IMAGE_API_BASE}/image/recent",
        params={"session_id": session_id, "limit": limit},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


@tool("yolo_scene")
def yolo_scene(session_id: str, last: int = 1) -> Dict[str, Any]:
    """
    [목적] 최근 YOLO 검출 결과(JSON)를 가져와, LLM이 '차량 n대, 보행자 m명, 신호등 빨간불 → 대기' 식으로 요약.
    [동작] GET {PERCEPTION_API_BASE}/perception/yolo/recent?session_id=...&last=...
    [응답] {"items":[{ width,height, ts, detections:[{label,bbox,conf,...}], image_filename?, frame_id? }]}
    """
    log.debug(f"[yolo_scene] session_id='{session_id}', last={last}")
    r = requests.get(
        f"{PERCEPTION_API_BASE}/perception/yolo/recent",
        params={"session_id": session_id, "last": last},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


@tool("gaze_depth_status")
def gaze_depth_status(session_id: str, last: int = 1) -> Dict[str, Any]:
    """
    [목적] 최근 시선/깊이/위험 인지 데이터를 가져와, LLM이 경고/행동 지침을 말하게 함.
    [동작] GET {PERCEPTION_API_BASE}/perception/gaze/recent?session_id=...&last=...
    [응답] {"items":[{gaze_xy_norm:[x,y], focus_dist_m, sudden_entry, hazards:[{kind,distance_m,rel_bearing_deg,risk}], ...}]}
    """
    log.debug(f"[gaze_depth_status] session_id='{session_id}', last={last}")
    r = requests.get(
        f"{PERCEPTION_API_BASE}/perception/gaze/recent",
        params={"session_id": session_id, "last": last},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


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
    log.debug(f"[weather_now] lat={lat}, lon={lon}")
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
    """
    사용 가능한 도구 목록을 반환.
    - 기본: image_lookup, image_recent, yolo_scene, gaze_depth_status, weather_now
    - Tavily 키가 있으면 web_search를 최우선으로 추가
    """
    tools = [image_lookup, image_recent, yolo_scene, gaze_depth_status, weather_now]
    if TAVILY_API_KEY:
        tools.insert(0, web_search)
    return tools
