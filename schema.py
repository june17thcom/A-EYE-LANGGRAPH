# schema.py
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

class ImageItem(BaseModel):
    filename: str = Field(description="이미지 파일명")
    data_uri: str = Field(description="data:URI (base64)")

class RAGSource(BaseModel):
    source: str
    url: Optional[str] = None
    snippet: Optional[str] = None

class WearableConversationalResponse(BaseModel):
    intent: Literal["qa", "guide", "status", "control", "other"] = Field(
        description="사용자 의도 태그"
    )
    answer: str = Field(
        description=(
            "사용자의 마지막 발화에 대해 한국어로 바로 답한다(음성 전용). "
            "먼저 핵심 1문장 → 이어서 최대 3개 번호 포인트('첫째, 둘째, 셋째'). "
            "링크/표/코드/데이터 URI/원시 JSON을 읽지 않는다. "
            "금지: '무엇을 도와드릴까요', '주제를 알려달라', '입력이 모호', '잘 안 들려요' 등 요구/사과/유도 멘트. "
            "정말 필요할 때만 맨 끝에 한 줄 확인 질문을 붙인다."
        )
    )
    citations: List[RAGSource] = Field(default_factory=list)

class WearableImageResponse(BaseModel):
    intent: Literal["image-insight", "image-lookup"] = "image-insight"
    answer: str
    images: List[ImageItem] = Field(default_factory=list)
    citations: List[RAGSource] = Field(default_factory=list)

class FinalResponse(BaseModel):
    final_output: Union[WearableConversationalResponse, WearableImageResponse]
