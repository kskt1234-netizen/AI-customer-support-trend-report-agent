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

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# 1) 의도(Intent) — 오케스트레이터의 분류 대상
# ─────────────────────────────────────────────────────────────
class IntentType(str, Enum):
    """라우팅 가능한 의도의 '닫힌 집합(closed set)'.

    [왜 Enum인가?]
    의도를 평범한 str로 두면 LLM이 "trend", "리포트", "TrendReport" 처럼
    제멋대로 변형된 값을 뱉을 수 있고, 그러면 라우팅 분기가 깨진다.
    Enum으로 허용값을 못 박으면, with_structured_output이 LLM 출력을
    이 두 값 중 하나로 '강제'한다. → 라우팅 안정성 확보.
    """

    SIMPLE_CHAT = "SIMPLE_CHAT"  # 일반 고객 문의/잡담 → 그냥 답변
    TREND_REPORT = "TREND_REPORT"  # 시장/매출 트렌드 분석 요청 → 리포트 서브 그래프


class IntentClassification(BaseModel):
    """LLM 의도 분류 결과를 담는 구조화 출력(Structured Output) 그릇.

    오케스트레이터에서 `llm.with_structured_output(IntentClassification)` 으로 쓰면,
    LLM은 자유 텍스트가 아니라 '이 스키마에 맞는 객체'를 반환하도록 강제된다.
    """

    intent: IntentType = Field(
        ...,  # ... = 필수(required)
        description="유저 질문의 의도. SIMPLE_CHAT 또는 TREND_REPORT 중 하나.",
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
