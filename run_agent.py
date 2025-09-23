import sys
import json
import argparse
from langchain_core.messages import HumanMessage
from graph import build_graph

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

def main():
    ap = argparse.ArgumentParser(description="LangGraph CLI runner")
    ap.add_argument("text", help="사용자 입력 문장 (따옴표로 감싸세요)")
    ap.add_argument("--session", default="cli", help="세션 ID(thread_id)")
    ap.add_argument("--lat", type=float, default=None, help="위도")
    ap.add_argument("--lon", type=float, default=None, help="경도")
    args = ap.parse_args()

    graph = build_graph()

    # /invoke와 동일: 좌표가 있으면 시스템 힌트로 먼저 전달
    msgs = []
    if args.lat is not None and args.lon is not None:
        msgs.append({"role": "system", "content": f"사용자 좌표(lat,lon): {args.lat},{args.lon}"})
    msgs.append({"role": "user", "content": args.text})

    config = {"configurable": {"thread_id": args.session}}
    out = graph.invoke({"messages": msgs}, config=config)

    final = out.get("final_response")
    if final is None:
        # fallback: 마지막 AI 텍스트라도 출력
        msgs = out.get("messages", [])
        ai_text = ""
        for m in reversed(msgs):
            role = getattr(m, "type", None) or getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role == "ai" and content:
                ai_text = content if isinstance(content, str) else str(content)
                break
        data = {"final_output": {"intent": "qa", "answer": scrub_answer(ai_text), "citations": []}}
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # Pydantic 모델/딕셔너리 모두 대응
    if hasattr(final, "model_dump"):
        data = final.model_dump()
    elif isinstance(final, dict):
        data = final
    else:
        try:
            from pydantic import BaseModel as _BaseModel
            data = final.dict() if isinstance(final, _BaseModel) else {
                "final_output": {"intent": "qa", "answer": str(final), "citations": []}
            }
        except Exception:
            data = {"final_output": {"intent": "qa", "answer": str(final), "citations": []}}

    # 불필요한 멘트 정리
    if isinstance(data, dict) and "final_output" in data and "answer" in data["final_output"]:
        data["final_output"]["answer"] = scrub_answer(data["final_output"]["answer"])

    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
