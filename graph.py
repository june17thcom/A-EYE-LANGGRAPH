import os
import logging
from typing import Annotated, Literal, TypedDict, Optional, Any, Dict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool

from schema import FinalResponse
from tools import get_tools  # ← weather_now, image_lookup, web_search 포함

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("wearable.graph")
log.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

class State(TypedDict):
    messages: Annotated[list, add_messages]
    final_response: Optional[dict]

SYSTEM_PROMPT = """\
너는 한국어 일상 대화형 챗봇이야. 사용자는 화면을 보지 못하고, 답변은 항상 음성으로만 들을 거야.
먼저 한 문장으로 핵심을 말하고, 이어서 최대 셋만 '첫째, 둘째, 셋째'로 짧게 보충해.
시각 의존 표현·링크·데이터 URI·코드 낭독 금지. 필요할 때만 끝에 한 줄 확인.
'주제를 알려달라/모호하다/무엇을 도와드릴까요/잘 안 들려요' 같은 말은 절대 하지 마.

날씨가 물어보이면:
- 이전 메시지 중 '사용자 좌표(lat,lon): a,b'가 있으면 그 좌표로 weather_now(lat=a, lon=b) 도구를 호출해 현재 상태를 요약해.
- 좌표가 없으면 일반 조언 대신, 도시/위치 질문을 한 줄로만 붙여. (그러나 먼저 가능한 가정으로 간단 요약은 제공)
이미지 관련이면 image_lookup를 우선 고려하고, 외부 사실 보강이 필요하면 web_search로 2~3개 출처 이름만 첨언해.
"""

def _ellipsize(s: str, n: int = 200) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."

def _get_role_and_content(msg: Any):
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    content = getattr(msg, "content", None)
    if role is None or content is None:
        if isinstance(msg, dict):
            role = role or msg.get("type") or msg.get("role")
            content = content or msg.get("content")
    if isinstance(content, list):
        try:
            text_parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type")=="text" and p.get("text")]
            content = "\n".join(text_parts) if text_parts else ""
        except Exception:
            content = ""
    return (str(role) if role else None), (str(content) if content else "")

class LoggedToolNode:
    def __init__(self, tools: list[BaseTool]):
        self.tools_by_name: Dict[str, BaseTool] = {t.name: t for t in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("LoggedToolNode: no messages in input")

        last = messages[-1]
        outputs = []
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            log.debug("[tools] no tool_calls")
            return {"messages": outputs}

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            call_id = call.get("id", "<no-id>")
            tool = self.tools_by_name.get(name)
            if not tool:
                log.warning(f"[tools] unknown tool '{name}'")
                continue
            try:
                log.info(f"[tools] call -> {name} | args={args}")
                result = tool.invoke(args)
                # data_uri 같은 대용량은 로깅 생략
                short = str(result)[:300] + ("..." if len(str(result)) > 300 else "")
                log.info(f"[tools] ok <- {name} | preview={_ellipsize(short, 180)}")
                outputs.append(
                    ToolMessage(
                        content=str(result) if not isinstance(result, (dict, list)) else str(result),
                        name=name,
                        tool_call_id=call_id,
                    )
                )
            except Exception as e:
                log.exception(f"[tools] exception in '{name}': {e}")
                outputs.append(
                    ToolMessage(
                        content=f'{{"error":"tool_failed","name":"{name}","detail":"{str(e)}"}}',
                        name=name,
                        tool_call_id=call_id,
                    )
                )
        return {"messages": outputs}

def build_graph():
    llm = ChatOpenAI(model="gpt-5-nano", temperature=1)
    tools = get_tools()  # image_lookup, web_search(옵션), weather_now 포함
    llm_with_tools = llm.bind_tools(tools)
    model_struct = llm.with_structured_output(FinalResponse)

    def agent(state: State):
        # 시스템 프롬프트를 항상 맨 앞에
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state.get("messages", [])
        # 마지막 메시지 로그
        if msgs:
            r, c = _get_role_and_content(msgs[-1])
            log.info(f"[agent] in | last_role={r} | last='{_ellipsize(c)}'")
        ai = llm_with_tools.invoke(msgs)
        has_tools = bool(getattr(ai, "tool_calls", None))
        ai_text = getattr(ai, "content", "")
        log.info(f"[agent] out | tool_calls={has_tools} | text='{_ellipsize(ai_text)}'")
        return {"messages": [ai]}

    def route(state: State) -> Literal["tools", "respond"]:
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        decision = "tools" if (getattr(last, "tool_calls", None)) else "respond"
        log.debug(f"[route] decision={decision}")
        return decision

    def respond(state: State):
        """
        마지막 AIMessage(툴 호출 이후 agent가 생성한 실제 답변)를 최우선으로 사용해
        최종 응답을 포장한다. 없을 때만 최소 fallback을 쓴다.
        """
        # 1) 마지막 AI 메시지 찾기
        last_ai_text = ""
        for m in reversed(state.get("messages", [])):
            role, content = _get_role_and_content(m)
            if role in ("ai", "assistant") and content:
                last_ai_text = content
                break

        # 2) AI 응답이 있으면 그걸 그대로 사용 (툴 결과 반영됨)
        if last_ai_text:
            final_dict = {
                "final_output": {
                    "intent": "qa",
                    "answer": last_ai_text,
                    "citations": []
                }
            }
            log.info(f"[respond] use_last_ai answer='{_ellipsize(last_ai_text)}'")
            return {"final_response": final_dict, "messages": []}

        # 3) 없으면 최신 user 문장으로 최소 생성 (드물게 발생하는 비정상 케이스)
        last_user = ""
        for m in reversed(state["messages"]):
            role, content = _get_role_and_content(m)
            if role in ("human", "user") and content:
                last_user = content
                break
        if not last_user:
            last_user = "간단히 정의해줘: 시선추적이란 무엇인가?"

        msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=last_user)]
        final_model = model_struct.invoke(msgs)  # Pydantic -> dict
        final_dict = final_model.model_dump()
        log.info(f"[respond] fallback_generated answer='{_ellipsize(final_dict.get('final_output', {}).get('answer',''))}'")
        return {"final_response": final_dict, "messages": []}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", LoggedToolNode(tools))
    g.add_node("respond", respond)

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "respond": "respond"})
    g.add_edge("tools", "agent")

    memory = MemorySaver()
    app = g.compile(checkpointer=memory)
    log.info("Graph compiled with MemorySaver.")
    return app
