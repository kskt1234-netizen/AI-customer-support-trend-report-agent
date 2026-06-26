"""
main_graph.py — 마스터(오케스트레이터) 그래프.

[이 그래프의 단 하나의 책임: 라우팅]
유저 질문을 받아 의도를 분류하고, 그 의도에 맞는 워커로 '방향만' 정한다.
실제 업무(트렌드 분석 등)는 서브 그래프(워커)가 한다. → 마스터-워커 패턴.

[그래프 형태]

        START
          │
          ▼
   ┌──────────────┐
   │ classify_intent │  ← LLMProvider.classify_intent() 호출
   └──────────────┘     (with_structured_output으로 의도를 Enum으로 강제)
          │
          ▼
   ◇ route_by_intent ◇   ← '조건부 엣지'. State.intent 값을 읽어 분기.
        ╱        ╲
       ╱          ╲
 SIMPLE_CHAT   TREND_REPORT
   노드            노드(서브 그래프)
       ╲          ╱
        ╲        ╱
          ▼    ▼
           END

[조건부 엣지(conditional edge)란? — 면접 포인트]
LangGraph에서 노드는 보통 '다음 노드'로 고정 연결된다. 하지만 라우팅은
'상태에 따라' 다음 목적지가 달라져야 한다. 그래서 add_conditional_edges를 쓴다:
  - 라우터 함수가 State를 보고 '문자열 키'를 반환하고,
  - 그 키 → 목적지 노드 매핑(dict)으로 분기한다.
즉 "분기 결정 로직(파이썬 함수)"과 "그래프 위상(엣지)"이 분리된다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.llm.base import LLMProvider
from src.llm.factory import get_provider
from src.logging_config import get_logger
from src.schemas import IntentType
from src.state import AgentState

# 서브 그래프(워커). ④번 덩어리에서 구현한 진짜 트렌드 리포트 그래프를 가져온다.
from src.sub_graphs.trend_report import build_trend_report_graph

_logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# 노드 1) 의도 분류 — 오케스트레이터의 두뇌
# ─────────────────────────────────────────────────────────────
def make_classify_intent_node(provider: LLMProvider):
    """의도 분류 노드를 '생성'하는 팩토리 함수.

    [왜 노드를 함수로 감싸 LLMProvider를 주입하나? — DIP의 실전 적용]
    노드가 내부에서 get_provider()를 직접 부르면, 노드가 다시 공급사 결정에
    묶인다. 대신 '주입받은 provider'를 클로저로 잡아두면:
      - 테스트에서 FakeProvider를 주입해 API 없이 돌릴 수 있고,
      - "이 노드는 OpenAI, 저 노드는 Claude"처럼 노드별 공급사 주입이 가능하다.
    이것이 ②번에서 만든 추상화가 '실제로 값을 발휘하는' 지점이다.
    """

    def classify_intent(state: AgentState) -> dict:
        # 노드는 State를 입력받아, '바꿀 필드만' dict로 반환한다.
        # LangGraph가 이 반환 dict를 기존 State에 병합한다.
        user_query = state["user_query"]
        result = provider.classify_intent(user_query)

        # [관찰가능성] 분류 결과와 '근거(reasoning)'를 로그로 남긴다.
        # 분류는 가장 자주 틀리는 지점이라, 틀렸을 때 '왜 그렇게 판단했는지'를
        # 사후 추적할 수 있도록 reasoning을 DEBUG 레벨로 흘린다.
        # (LOG_LEVEL=DEBUG 로 실행하면 보인다. 운영 기본 INFO에선 결과만 남는다.)
        _logger.info("의도 분류: query=%r → intent=%s", user_query, result.intent.value)
        _logger.debug("분류 근거(reasoning): %s", result.reasoning)

        # result.intent 는 IntentType(Enum). State에는 그 .value(문자열)를 저장한다.
        # → 라우터 함수가 평범한 문자열을 다루게 해 분기를 단순화.
        return {"intent": result.intent.value}

    return classify_intent


# ─────────────────────────────────────────────────────────────
# 노드 2) SIMPLE_CHAT — 단순 답변 워커(여기서는 간단히 처리)
# ─────────────────────────────────────────────────────────────
def make_simple_chat_node(provider: LLMProvider):
    """일반 문의에 자유 텍스트로 답하는 노드."""

    def simple_chat(state: AgentState) -> dict:
        chat_model = provider.get_chat_model()
        response = chat_model.invoke(
            [
                ("system", "당신은 친절한 B2B SaaS 고객지원 어시스턴트입니다. 간결히 답하세요."),
                ("human", state["user_query"]),
            ]
        )
        # LangChain 메시지 객체의 .content가 실제 답변 텍스트.
        return {"chat_response": response.content}

    return simple_chat


# ─────────────────────────────────────────────────────────────
# 라우터) 조건부 엣지가 호출하는 '순수 분기 함수'
# ─────────────────────────────────────────────────────────────
def route_by_intent(state: AgentState) -> str:
    """State.intent를 보고 다음 노드의 '키'를 반환한다.

    [왜 노드가 아니라 '함수'인가?]
    이 함수는 State를 바꾸지 않는다. 오직 '어디로 갈지'만 결정한다(순수 함수).
    부작용이 없으니 단위 테스트가 쉽고, 라우팅 규칙이 한눈에 보인다.
    반환하는 문자열은 아래 add_conditional_edges의 매핑 키와 일치해야 한다.
    """
    intent = state.get("intent")
    if intent == IntentType.TREND_REPORT.value:
        return "trend_report"
    # 분류 실패/누락을 포함한 그 외 모든 경우는 안전하게 단순 답변으로 보낸다.
    # (알 수 없는 의도를 그냥 크래시시키지 않는 방어적 기본값)
    return "simple_chat"


# ─────────────────────────────────────────────────────────────
# 그래프 빌더 — 노드/엣지를 조립해 실행 가능한 그래프로 컴파일
# ─────────────────────────────────────────────────────────────
def build_main_graph(provider: LLMProvider | None = None):
    """메인 오케스트레이터 그래프를 빌드해 컴파일된 그래프를 반환한다.

    Args:
        provider: 주입할 LLM 공급사. None이면 팩토리 기본값(OpenAI)을 쓴다.
                  테스트에서는 FakeProvider를 주입한다. → 투트랙 테스트의 핵심.
    """
    provider = provider or get_provider()

    # StateGraph(AgentState): 이 그래프의 모든 노드가 AgentState를 주고받는다고 선언.
    graph = StateGraph(AgentState)

    # ── 노드 등록 ──
    graph.add_node("classify_intent", make_classify_intent_node(provider))
    graph.add_node("simple_chat", make_simple_chat_node(provider))
    # 서브 그래프도 '하나의 노드'처럼 꽂힌다. 컴파일된 서브 그래프는 호출 가능한
    # 객체라서, 메인 입장에선 내부를 몰라도 된다. (워커의 캡슐화)
    graph.add_node("trend_report", build_trend_report_graph(provider))

    # ── 엣지(흐름) 정의 ──
    # 1) 시작 → 분류
    graph.add_edge(START, "classify_intent")

    # 2) 분류 → (조건부) → 워커
    #    route_by_intent가 반환한 키를, 아래 매핑으로 실제 노드에 연결한다.
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "simple_chat": "simple_chat",
            "trend_report": "trend_report",
        },
    )

    # 3) 각 워커 → 종료
    graph.add_edge("simple_chat", END)
    graph.add_edge("trend_report", END)

    # compile(): 위상이 올바른지 검증하고 실행 가능한 그래프로 만든다.
    return graph.compile()


# 모듈 임포트 시점에 바로 쓰고 싶을 때를 위한 편의 인스턴스.
# (테스트는 build_main_graph에 FakeProvider를 주입하므로 이걸 쓰지 않는다.)
def get_compiled_app():
    """기본 공급사로 컴파일된 메인 앱을 반환하는 편의 함수."""
    return build_main_graph()
