# **1) 프로젝트 구조**

```
ai-wearable-agent/
├─ .env.example
├─ requirements.txt
├─ README.md
├─ images/                  # (선택) 샘플 이미지 넣어두면 FastAPI가 q로 검색 가능
│   ├─ street_stop_sign.jpg
│   └─ crosswalk_people.jpg
├─ fastapi_server.py        # 이미지 검색/인코딩 API (data: URI 반환)
├─ tools.py                 # Tavily 검색 도구 + FastAPI 이미지 도구
├─ schema.py                # Pydantic 구조화 출력 스키마
├─ graph.py                 # LangGraph 그래프 (노드/엣지 정의)
└─ run_agent.py             # STT 텍스트(질문) 받아서 그래프 실행

```

# **2) 의존성**

```
# 1) 클론/폴더 이동 (생략)
# cd C:\Gukbi\ai-wearable-agent

# 2) 가상환경
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) 의존성 설치
pip install -r .\requirements.txt

# 4) 환경변수 파일
Copy-Item .\.env .\.env
# .env 열어서 OPENAI_API_KEY / TAVILY_API_KEY 채우기

# 5) (선택) 샘플 이미지 넣기
# images\ 폴더에 파일 몇 장 복사 (파일명으로 검색됨)

# 6) FastAPI 서버 실행 (이미지 제공)
uvicorn fastapi_server:app --host 127.0.0.1 --port 8000 --reload

# 7) 새 터미널에서 에이전트 실행
# 예) “횡단보도 주변 위험 상황 이미지 좀 보여줘”
python .\run_agent.py "횡단보도 이미지 찾아서 설명해줘"

# 예) 일반 QA (Tavily RAG 동작)
python .\run_agent.py "비 오는 날 우산 없이 이동할 때 안전 수칙 알려줘"

```

# **3) Windows/VSCode 실행 방법 (PowerShell 기준)**

````
# Wearable LangGraph Service (Docker)

## 빠른 시작 (Windows PowerShell)

```powershell
# 1) .env 준비
Copy-Item .\.env.example .\.env
# .env 파일에서 OPENAI_API_KEY / TAVILY_API_KEY 채우기
# (IMAGE_API_BASE는 컨테이너 내부 호출이므로 기본값 http://127.0.0.1:8000 그대로 두세요)

# 2) 도커 빌드/실행
docker compose build
docker compose up -d

# 3) 헬스체크 (호스트에서 8010으로 접근)
curl http://127.0.0.1:8010/health

# 4) 그래프 호출 (호스트에서 8010으로 접근)
curl -X POST http://127.0.0.1:8010/invoke `
     -H "Content-Type: application/json" `
     -d "{`"text`": `"비 오는 날 보행 안전 수칙 알려줘`"}"

# 5) 이미지 검색 (호스트에서 8010으로 접근)
curl "http://127.0.0.1:8010/image/search?q=crosswalk"


````

# **4) 로컬 메인앱(시선추적)에서 호출 예시 (Python)**

```
import requests

def ask_agent(text: str):
    r = requests.post(
        "http://127.0.0.1:8000/invoke",
        json={"text": text},
        timeout=15
    )
    r.raise_for_status()
    return r.json()

resp = ask_agent("횡단보도 이미지 찾아서 상황 설명해줘")
print(resp)  # FinalResponse 스키마 JSON

```

# ** 5) Docker 빌드 & 실행**

```
# 빌드
docker compose build

# 실행 (백그라운드)
docker compose up -d

```

- 헬스체크
- curl http://127.0.0.1:8010/health

- 그래프 호출 (예시)

```
# ① JSON 문자열 만들기
$json = '{"text":"시선추적, 아이트래킹이란 뭐야?","session_id":"alpha"}'

# ② UTF-8 바이트로 인코딩
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)

# ③ Content-Type에 charset을 명시하고 전송
Invoke-WebRequest -Uri "http://127.0.0.1:8010/invoke" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $bytes `
  -OutFile .\resp_alpha_1.json

```

- 이미지 검색 (예시)
  curl "http://127.0.0.1:8010/image/search?q=crosswalk"
