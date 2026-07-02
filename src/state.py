"""
state.py — 그래프 전체가 공유하는 단일 상태(State) 정의.

[설계 의도]
LangGraph의 노드들은 서로를 직접 호출하지 않는다. 대신 '하나의 State 객체'를
입력으로 받아, 일부 필드를 갱신한 dict를 반환한다. 그러면 LangGraph가 그 변경분을
기존 State에 병합(merge)해 다음 노드로 넘긴다.

즉 State는 노드들 사이의 '공용 메모리'이자 '데이터 버스'다. 그래서 어떤 노드가
무엇을 읽고 무엇을 쓰는지가 이 파일 하나만 보면 다 드러나야 한다.

[왜 Pydantic이 아니라 TypedDict인가? — 면접 방어 포인트]
- State(여기): 노드 사이를 매 스텝 흐르는 '내부 작업 메모리'다. 자주, 부분적으로
  갱신된다. TypedDict는 런타임 검증 비용이 없고(그냥 dict라서), LangGraph가
  요구하는 '부분 갱신 후 병합' 패턴과 가장 잘 맞는다.
- schemas.py의 Pydantic: 그래프 '경계(boundary)'를 넘는 데이터다. 여기서는
  엄격한 런타임 검증이 필요하다. → 둘의 역할이 다르므로 도구도 다르게 쓴다.
  ("내부 작업물은 가볍게 TypedDict, 외부로 나가는 계약은 엄격하게 Pydantic")
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """그래프 전역에서 공유되는 상태.

    total=False 인 이유:
      모든 필드가 처음부터 채워져 있지 않다. 예를 들어 intent는 분류 노드를
      통과해야 생기고, market_growth_rate는 연산 노드를 통과해야 생긴다.
      total=False 로 두면 "아직 안 채워진 필드"를 허용하면서도, 각 필드의
      '타입'은 명시할 수 있다. (필수/선택을 섞고 싶으면 Required/NotRequired를
      쓸 수도 있으나, 여기서는 단순성을 위해 전체를 선택적으로 둔다.)
    """

    # ── 입력 ──────────────────────────────────────────────
    user_query: str  # 유저의 원본 질문. 모든 흐름의 출발점.

    # ── 오케스트레이터(메인 그래프)가 채우는 필드 ──────────────
    intent: str  # "SIMPLE_CHAT" 또는 "TREND_REPORT". 라우팅 기준.

    # ── SIMPLE_CHAT 경로의 결과 ───────────────────────────
    chat_response: str  # 단순 답변 텍스트.

    # ── TREND_REPORT 서브 그래프가 채우는 필드 ────────────────
    raw_search_data: dict  # asyncio.gather로 병렬 수집한 원본 더미 데이터.
    market_growth_rate: float  # ⚠️ LLM이 아니라 '파이썬 코드'가 계산한 성장률(%).
    sources: list[str]  # 수집 단계에서 모은 출처 목록.

    # ── 자가 수정(Critic) 루프 관련 ──────────────────────────
    draft_report: str  # LLM이 작성한 보고서 초안.
    critic_passed: bool  # 검수 '합격 도장'. True면 통과, False면 반려. (명시적 신호)
    critic_feedback: str  # 검수자가 초안을 반려할 때 남기는 피드백.
    retry_count: int  # 초안↔검수 핑퐁 횟수. 무한 루프 방어용 카운터.

    # ── 최종 산출 / 예외 ─────────────────────────────────────
    final_output: dict  # Pydantic(TrendReportOutput) 검증을 통과한 최종 결과의 dict.
    error: str  # 저하/탈출 사유. (수집 실패, 수치 불량, LLM 호출 실패, 루프 한도 초과 등)
