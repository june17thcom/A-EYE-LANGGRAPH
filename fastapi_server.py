import os
import mimetypes
import logging
import uuid
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conlist

from langchain_core.messages import SystemMessage, HumanMessage
from graph import build_graph

# ── Logging ─────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("wearable.server")

# ── FastAPI & CORS ─────────────────────────────────────────────
app = FastAPI(title="Wearable LangGraph Service", version="3.0.0")
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost,http://127.0.0.1").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 이미지 검색(도구가 내부적으로 호출) ───────────────────────────
IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

def to_data_uri(fp: Path) -> str:
    mime, _ = mimetypes.guess_type(str(fp))
    mime = mime or "application/octet-stream"
    import base64 as _b64
    b64 = _b64.b64encode(fp.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

@app.get("/image/search")
def image_search(q: str = Query(..., description="파일명 키워드")) -> JSONResponse:
    if not q.strip():
        return JSONResponse(
            {"error":"bad_request","detail":"q must not be empty."},
            media_type="application/json; charset=utf-8", status_code=400
        )
    items: List[Dict] = []
    for fp in IMAGES_DIR.glob("*"):
        if fp.is_file() and q.lower() in fp.name.lower():
            items.append({"filename": fp.name, "data_uri": to_data_uri(fp)})
    log.debug(f"[image_search] q='{q}' -> {len(items)} items")
    return JSONResponse({"items": items}, media_type="application/json; charset=utf-8")

# ── Perception 저장 위치 ─────────────────────────────────────────
PERCEPTION_DIR = Path(__file__).parent / "perception"
GAZE_DIR = PERCEPTION_DIR / "gaze"
GAZE_DIR.mkdir(parents=True, exist_ok=True)

# ── LangGraph: 지연 초기화 + 세션(thread_id) ─────────────────────
GRAPH = None
GRAPH_ERROR: Optional[str] = None

def ensure_graph():
    global GRAPH, GRAPH_ERROR
    if GRAPH or GRAPH_ERROR:
        return
    try:
        GRAPH = build_graph()  # 체크포인터 포함 app
        log.info("LangGraph compiled and ready.")
    except Exception as e:
        GRAPH_ERROR = str(e)
        log.exception("Failed to build graph: %s", GRAPH_ERROR)

class InvokeIn(BaseModel):
    text: str = Field(description="사용자 발화 텍스트")
    session_id: Optional[str] = Field(default="default", description="세션 ID(동일 ID면 대화 기억 유지)")
    lat: Optional[float] = Field(default=None, description="사용자 위도")
    lon: Optional[float] = Field(default=None, description="사용자 경도")

BANNED_PHRASES = [
    "무엇을 도와드릴지", "무엇을 도와드릴까요", "주제를 알려", "입력이 모호",
    "잘 안 들려요", "말씀해 주시면", "알려주시면", "원하시는 형식",
]
def scrub_answer(s: str) -> str:
    if not isinstance(s, str):
        return s
    for p in BANNED_PHRASES:
        s = s.replace(p, "")
    return " ".join(s.split())

@app.post("/invoke")
def invoke(payload: InvokeIn, request: Request):
    ensure_graph()
    req_id = str(uuid.uuid4())[:8]
    session_id = payload.session_id or "default"

    if GRAPH_ERROR:
        log.error(f"[{req_id}] /invoke init error | session={session_id} | {GRAPH_ERROR}")
        return JSONResponse(
            {"error":"graph_init_failed","detail":GRAPH_ERROR},
            media_type="application/json; charset=utf-8",
            status_code=500
        )
    try:
        log.info(f"[{req_id}] /invoke start | session={session_id} | text='{payload.text}' | lat={payload.lat} lon={payload.lon}")
        config = {"configurable": {"thread_id": session_id}}

        msgs = []
        if payload.lat is not None and payload.lon is not None:
            msgs.append(SystemMessage(content=f"사용자 좌표(lat,lon): {payload.lat},{payload.lon}"))
        msgs.append(HumanMessage(content=payload.text))
        state_in = {"messages": msgs}

        out = GRAPH.invoke(state_in, config=config)
        out_keys = list(out.keys())
        log.debug(f"[{req_id}] graph.invoke OK | out_keys={out_keys}")

        final = out.get("final_response")
        if final is None:
            msgs = out.get("messages", [])
            ai_text = ""
            for m in reversed(msgs):
                role = getattr(m, "type", None) or getattr(m, "role", None)
                content = getattr(m, "content", None)
                if role == "ai" and content:
                    ai_text = content if isinstance(content, str) else str(content)
                    break
            ans = scrub_answer(ai_text)
            log.info(f"[{req_id}] /invoke fallback | session={session_id} | answer='{ans[:120]}...'")
            return JSONResponse(
                {"final_output": {"intent":"qa","answer": ans, "citations":[]}},
                media_type="application/json; charset=utf-8",
                status_code=200
            )

        if hasattr(final, "model_dump"):
            data = final.model_dump()
        elif isinstance(final, dict):
            data = final
        else:
            try:
                from pydantic import BaseModel as _BaseModel
                data = final.dict() if isinstance(final, _BaseModel) else {
                    "final_output":{"intent":"qa","answer": str(final), "citations":[]}
                }
            except Exception:
                data = {"final_output":{"intent":"qa","answer": str(final), "citations":[]}}

        if isinstance(data, dict) and "final_output" in data and "answer" in data["final_output"]:
            data["final_output"]["answer"] = scrub_answer(data["final_output"]["answer"])
            log.info(f"[{req_id}] /invoke done | session={session_id} | answer='{data['final_output']['answer'][:120]}...'")

        return JSONResponse(data, media_type="application/json; charset=utf-8", status_code=200)

    except Exception as e:
        log.exception(f"[{req_id}] /invoke exception | session={session_id}")
        return JSONResponse(
            {"error":"exception","detail":str(e)},
            media_type="application/json; charset=utf-8",
            status_code=500
        )

# ── Gaze Push 전용(422 방지: 리스트 길이2 허용) ────────────────────
class GazePushIn(BaseModel):
    session_id: str = Field(..., description="대화/사용자 세션 ID")
    ts: Optional[float] = Field(default=None, description="타임스탬프(초, epoch)")
    gaze_xy_norm: conlist(float, min_length=2, max_length=2) = Field(
        ..., description="정규화 좌표 [x,y], 0~1 범위"
    )
    focus_dist_m: Optional[float] = Field(default=None, description="주시 거리(미터)")
    sudden_entry: Optional[bool] = Field(default=False, description="갑작스런 진입 감지 여부")
    hazards: List[Dict[str, Any]] = Field(default_factory=list, description="추가 위험/메타 정보")

@app.post("/perception/gaze/push")
def perception_gaze_push(payload: GazePushIn):
    rec = payload.model_dump()
    GAZE_DIR.mkdir(parents=True, exist_ok=True)
    # 최근 1건 저장
    (GAZE_DIR / "recent.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 간단 OK
    return JSONResponse({"ok": True}, media_type="application/json; charset=utf-8")

@app.get("/health")
def health():
    ensure_graph()
    status = "ok" if not GRAPH_ERROR else "degraded"
    return {"status": status, "graph_error": GRAPH_ERROR}
