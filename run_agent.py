import sys
from langchain_core.messages import HumanMessage
from graph import build_graph

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_agent.py \"당신의 질문 텍스트\"")
        sys.exit(1)

    question = sys.argv[1]
    graph = build_graph()
    out = graph.invoke({"messages": [HumanMessage(content=question)]})

    final = out.get("final_response")
    if final:
        print("=== FinalResponse ===")
        print(final.model_dump_json(indent=2, ensure_ascii=False))
    else:
        print("No final_response in graph output:", out.keys())

if __name__ == "__main__":
    main()
