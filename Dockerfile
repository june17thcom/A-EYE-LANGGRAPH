# --- Base ---
FROM python:3.11-slim

# 파이썬/로케일/UTF-8 안정화
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# --- System deps ---
#  - curl: 컨테이너 헬스체크에서 사용
#  - ca-certificates: OpenAI/Tavily TLS
#  - build-essential: 일부 파이썬 패키지 빌드 시 필요할 수 있음(경량 유지)
#  - tzdata: 로그 타임존 설정(Compose에서 TZ=Asia/Seoul과 함께)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- App code ---
COPY . .

# 업로드/최근 프레임 저장 디렉토리 보장
RUN mkdir -p /app/images

# 내부 FastAPI를 자기 자신으로 부를 때 기본값 (compose에서 override 가능)
ENV IMAGE_API_BASE=http://127.0.0.1:8000
ENV PERCEPTION_API_BASE=http://127.0.0.1:8000
# 로그 레벨 기본
ENV LOG_LEVEL=INFO

# 포트
EXPOSE 8000

# 실행
CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
