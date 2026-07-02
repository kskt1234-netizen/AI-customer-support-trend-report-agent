"""
schemas.py — 그래프 '경계'를 넘는 데이터의 엄격한 계약(Contract).

[State(TypedDict)와 무엇이 다른가? — 면접 방어 포인트]
- state.py 의 AgentState: 노드들 사이를 흐르는 '내부 작업 메모리'. 가볍게, 검증 없이.
- schemas.py 의 Pydantic 모델: 그래프 경계를 넘나드는 '공식 계약'. 무겁게, 엄격하게.

여기서 Pydantic을 쓰는 진짜 이유는 '타입 힌트'가 아니라 '런타임 검증'이다.
LLM은 우리가 시켜도 가끔 형식을 어긴다(숫자 대신 문자열, 필드 누락 등).
Pydantic 모델로 한 번 통과시키면, 잘못된 데이터가 다음 단계로 새어 나가는 것을
'경계에서' 차단할 수 있다. → "타입이 아니라 계약이다. 계약 위반은 경계에서 막는다."

이 파일의 두 모델:
  1) IntentClassification : 오케스트레이터가 LLM의 의도 분류 결과를 강제로 담는 그릇.
                            with_structured_output(IntentClassification) 형태로 쓴다.
  2) TrendReportOutput    : 트렌드 서브 그래프의 최종 산출물 계약.
                            요구사항대로 growth_rate(float)/summary/sources를 '반드시' 가진다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────
# 1) 의도(Intent) — 오케스트레이터의 분류 대상
# ─────────────────────────────────────────────────────────────
class IntentType(str, Enum):
    """라우팅 가능한 의도의 '닫힌 집합(closed set)'.

    [왜 Enum인가?]
    의도를 평범한 str로 두면 LLM이 "trend", "리포트", "TrendReport" 처럼
    제멋대로 변형된 값을 뱉을 수 있고, 그러면 라우팅 분기가 깨진다.
    Enum으로 허용값을 못 박으면, with_structured_output이 LLM 출력을
    이 값들 중 하나로 '강제'한다. → 라우팅 안정성 확보.
    """

    SIMPLE_CHAT = "SIMPLE_CHAT"  # 일반 고객 문의/처리 요청/잡담 → 그냥 답변
    TREND_REPORT = "TREND_REPORT"  # 시장/매출 트렌드 분석 요청 → 리포트 서브 그래프
    POLICY_INQUIRY = "POLICY_INQUIRY"  # 회사 규정/정책/계약 조건 문의 → RAG 서브 그래프


class IntentClassification(BaseModel):
    """LLM 의도 분류 결과를 담는 구조화 출력(Structured Output) 그릇.

    오케스트레이터에서 `llm.with_structured_output(IntentClassification)` 으로 쓰면,
    LLM은 자유 텍스트가 아니라 '이 스키마에 맞는 객체'를 반환하도록 강제된다.
    """

    intent: IntentType = Field(
        ...,  # ... = 필수(required)
        description="유저 질문의 의도. SIMPLE_CHAT / TREND_REPORT / POLICY_INQUIRY 중 하나.",
    )
    reasoning: str = Field(
        ...,
        description="왜 그렇게 분류했는지에 대한 한 문장 근거. (디버깅/관찰가능성용)",
    )


# ─────────────────────────────────────────────────────────────
# 2) 트렌드 리포트 최종 산출물 — 서브 그래프의 출력 계약
# ─────────────────────────────────────────────────────────────
class TrendReportOutput(BaseModel):
    """트렌드 리포트 서브 그래프의 최종 결과 계약.

    [요구사항 고정]
    이 모델은 반드시 다음을 가진다:
      - market_growth_rate (float) : 코드가 계산한 성장률(%)
      - summary           (str)    : 사람이 읽을 요약 보고서
      - sources           (list)   : 근거 출처

    Critic 노드는 바로 이 '계약을 만족하는지'를 검사한다. 즉 이 스키마가
    자가 수정 루프의 '합격 기준(spec)' 역할도 겸한다.
    """

    market_growth_rate: float = Field(
        ...,
        description="전년 대비 시장/매출 성장률(%). LLM이 아니라 파이썬 코드가 계산한 값.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="트렌드 분석 요약 보고서 본문.",
    )
    sources: list[str] = Field(
        ...,
        min_length=1,  # 출처가 최소 1개는 있어야 '근거 있는 보고서'다.
        description="보고서 근거가 된 출처 목록. 비어 있으면 Critic이 반려한다.",
    )

    def has_required_evidence(self) -> bool:
        """Critic이 호출할 헬퍼 — '숫자 성장률'과 '출처'가 모두 갖춰졌는지.

        Pydantic 검증을 통과했다면 타입/필수는 이미 보장되지만, '의미적' 합격
        기준(출처가 실제로 1개 이상 존재하는가 등)을 한 곳에 모아두면 Critic
        노드가 읽기 쉬워진다. 계약과 합격 기준을 모델 안에 같이 둔다.
        """
        return bool(self.sources) and isinstance(self.market_growth_rate, float)


# ─────────────────────────────────────────────────────────────
# 3) 규정(RAG) 답변 최종 산출물 — POLICY_INQUIRY 워커의 출력 계약
# ─────────────────────────────────────────────────────────────
class PolicyAnswerOutput(BaseModel):
    """POLICY_INQUIRY(RAG) 워커의 최종 답변 계약.

    [설계 핵심 — 환각 방어를 '계약'에 새긴다]
    grounded 플래그가 이 모델의 심장이다:
      - grounded=True  : 답변이 검색된 사내 문서에 접지(grounding)됐고,
                         접지 검증을 통과했으며, 인용(citations)이 반드시 1개 이상이다.
      - grounded=False : 문서 근거를 확보하지 못한 '정직한 거절/안내' 답변이다.
                         (지어낸 답 대신 모른다고 말하는 것도 정상 산출물이다)
    즉 "근거 있는 답" 또는 "근거 없다고 정직하게 밝힌 답"만 이 계약을 통과한다.
    '근거 없는데 그럴듯한 답'은 아래 validator가 경계에서 차단한다.
    """

    answer: str = Field(
        ...,
        min_length=1,
        description="고객에게 전달할 답변 본문.",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="답변의 근거로 인용된 사내 문서 ID 목록. 거절 답변이면 비어 있을 수 있다.",
    )
    grounded: bool = Field(
        ...,
        description="답변이 검색된 문서에 접지되었는지. False면 정직한 '근거 없음' 안내.",
    )

    @model_validator(mode="after")
    def _grounded_requires_citations(self) -> "PolicyAnswerOutput":
        """grounded=True인데 인용이 없는 모순된 답변은 계약 위반으로 차단한다."""
        if self.grounded and not self.citations:
            raise ValueError("grounded=True인 답변은 최소 1개의 인용(citations)이 필요합니다.")
        return self
