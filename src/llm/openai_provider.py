"""
openai_provider.py — LLMProvider 계약의 'OpenAI' 구현체.

[이 파일의 역할]
base.py가 정의한 추상 계약(LLMProvider)을 OpenAI로 '실제 구현'한다.
구체 의존성(langchain_openai의 ChatOpenAI, 모델명, 온도 등)은 전부 이 파일
'안에만' 갇혀 있어야 한다. 다른 어떤 파일도 ChatOpenAI를 직접 import하지 않는다.
→ "OpenAI에 대한 지식은 이 파일 한 곳에만." 나중에 anthropic_provider.py를
   추가하면 끝나고, 노드/그래프 코드는 손대지 않는다.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.llm.base import LLMProvider
from src.schemas import IntentClassification

# ── 의도 분류용 시스템 프롬프트 ───────────────────────────────
# [왜 프롬프트를 모듈 상수로 빼두나?]
# 분류 품질은 곧 이 프롬프트 품질이다. 코드 로직과 섞지 않고 한 곳에 모아두면,
# 골든셋 평가 결과를 보며 '프롬프트만' 반복 튜닝하기 쉽다. (관심사 분리)
_INTENT_SYSTEM_PROMPT = """당신은 B2B SaaS 고객지원 허브의 라우터입니다.
유저의 질문을 정확히 세 의도 중 하나로 분류하세요.

- TREND_REPORT: 시장/매출/업계의 '트렌드·추세·성장률·전망'에 대한 분석이나
  리포트를 원하는 경우. (예: "올해 매출 트렌드 분석해줘", "시장 성장률 보고서 만들어줘")
- POLICY_INQUIRY: 회사의 '규정·정책·계약 조건'이 무엇인지 묻는 경우.
  환불 규정, 위약금, 계약 갱신/해지 조건, 데이터 보관 기간, 보안 정책 등.
  (예: "환불 규정이 어떻게 되나요?", "계약 해지하려면 언제까지 통지해야 해?")
- SIMPLE_CHAT: 그 외 모든 일반 문의. 사용법, 계정/비밀번호 변경 방법, 장애 신고,
  처리 요청, 잡담 등. (예: "비밀번호 어떻게 바꿔?", "결제를 잘못했는데 환불 처리해 주세요")

판단 기준(애매할 때):
- '규정/정책/조건이 무엇인지'를 물으면 → POLICY_INQUIRY
- '처리를 해달라'는 요청이나 사용법 질문이면 → SIMPLE_CHAT
  (예: "환불 규정 알려줘"=POLICY_INQUIRY, "환불해 주세요"=SIMPLE_CHAT)
- '분석/리포트/추세'를 명시적으로 요구하면 → TREND_REPORT
판단 근거(reasoning)도 한 문장으로 함께 제시하세요."""


class OpenAIProvider(LLMProvider):
    """OpenAI 기반 LLMProvider 구현.

    Args:
        model: 사용할 모델명. 분류·생성 모두에 쓰일 기본 모델.
        temperature: 샘플링 온도. 분류는 결정성이 중요하므로 기본 0.0.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        # [왜 온도 0.0이 기본인가?]
        # 의도 분류는 '창의성'이 아니라 '일관성'이 생명이다. 같은 질문엔 항상 같은
        # 분류가 나와야 골든셋 평가가 의미를 가진다. 그래서 분류 기본은 0.0.
        self._model_name = model
        self._temperature = temperature

        # 구조화 출력 전용 모델: with_structured_output으로 스키마를 강제한다.
        # 이렇게 하면 LLM이 자유 텍스트가 아니라 IntentClassification 객체를 반환.
        self._structured_llm = ChatOpenAI(
            model=model, temperature=temperature
        ).with_structured_output(IntentClassification)

        # 자유 텍스트 생성용 모델(초안 작성/검수 등).
        self._chat_llm = ChatOpenAI(model=model, temperature=temperature)

    def classify_intent(self, user_query: str) -> IntentClassification:
        """LLM을 호출해 의도를 분류한다. 반환 타입은 계약(스키마)이 보장한다."""
        # 시스템 프롬프트(규칙) + 유저 질문(데이터)을 함께 던진다.
        result = self._structured_llm.invoke(
            [
                ("system", _INTENT_SYSTEM_PROMPT),
                ("human", user_query),
            ]
        )
        # with_structured_output 덕분에 result는 이미 IntentClassification 인스턴스다.
        # (타입 체커를 위한 명시적 캐스팅은 생략 — 런타임 보장됨)
        return result  # type: ignore[return-value]

    def get_chat_model(self) -> BaseChatModel:
        """초안/검수 등 자유 텍스트 생성에 쓸 챗 모델을 반환한다."""
        return self._chat_llm
