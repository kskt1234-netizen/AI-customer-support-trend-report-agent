"""
rag_worker.py — POLICY_INQUIRY 워커(RAG 서브 그래프).

사내 규정/계약 문서를 검색(Retrieval)해 그 근거로만 답변을 생성(Generation)하고,
접지 검증(Grounding Check)으로 환각을 방어한다.

[핵심 설계 3가지]
  1) 검색 추상화(DIP) : 워커는 Retriever 계약만 안다. 더미 키워드 검색 →
     하이브리드 검색(Vector+BM25) 교체 시 이 파일은 0줄 수정.
  2) 접지 검증은 '코드'로 : 환각을 잡는 검증기가 스스로 환각하면 안 된다.
     인용 존재 + 수치 접지를 결정론적 코드로 검사한다. (LLM은 언어, 코드는 연산)
  3) 정직한 거절도 정상 산출물 : 근거가 없으면 지어내는 대신 "문서에서 근거를
     찾지 못했다"고 답한다. 고객 응대 워커는 어떤 경우에도(장애 포함) 계약
     (PolicyAnswerOutput)을 만족하는 답변을 반환한다 — error는 관찰용 부가 신호.

[전체 흐름]

  START
    │
    ▼
  retrieve_docs (async, Retriever 주입) ── 검색 실패는 error로 기록
    │
    ▼
  ◇ route_after_retrieve ◇
    ├─ 문서 0건 / 검색 실패 → no_evidence ── 정직한 '근거 없음' 답변 (LLM 미호출)
    │ 문서 있음
    ▼
  generate_answer (LLM) ◀──────────────────┐ "문서에 없는 수치" 등 피드백 동반
    │                                       │ retry_count += 1
    ▼                                       │
  grounding_check (순수 코드) ───────────────┘ (불합격 & retry < 한도면 루프)
    │   (a) 검색된 문서 ID가 인용되었는가?
    │   (b) 답변 속 모든 수치가 문서에 실존하는가?
    ▼
  ◇ route_after_grounding ◇
    ├─ 접지 확인          → finalize (Pydantic 계약 확정, grounded=True)
    └─ retry 한도 초과    → give_up  (안전한 거절 답변, grounded=False + error)
"""

from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.llm.base import LLMProvider, content_to_text
from src.logging_config import get_logger
from src.retrieval.base import Retriever
from src.retrieval.factory import get_retriever
from src.schemas import PolicyAnswerOutput
from src.state import AgentState

_logger = get_logger(__name__)

# 접지 실패 시 재작성 루프 한도. 트렌드 워커와 동일한 무한 루프 방어.
_MAX_RETRIES = 3

# ── 방어용 고정 답변 ──────────────────────────────────────────
# "지어낸 답"보다 "정직한 거절"이 낫다 — 이 워커의 존재 이유.
_NO_EVIDENCE_ANSWER = (
    "죄송합니다. 문의하신 내용은 현재 사내 규정 문서에서 근거를 찾지 못했습니다. "
    "부정확한 답변을 드리는 대신, 담당자 확인 후 정확히 안내드리겠습니다."
)
_GIVE_UP_ANSWER = (
    "죄송합니다. 지금은 문서 근거가 확인된 답변을 만들지 못했습니다. "
    "담당자에게 문의를 전달해 정확한 규정을 안내드리겠습니다."
)


# ═════════════════════════════════════════════════════════════
# 1) 검색(Retrieval) — Retriever 계약에만 의존
# ═════════════════════════════════════════════════════════════
def make_retrieve_node(retriever: Retriever):
    """[노드 팩토리] 주입받은 Retriever로 문서를 검색하는 노드를 만든다.

    LLMProvider와 같은 이유로 주입한다(DIP): 테스트에선 고장난 검색기를,
    운영에선 하이브리드 검색기를 꽂아도 이 노드는 그대로다.
    """

    async def retrieve_docs(state: AgentState) -> dict:
        query = (state.get("user_query") or "").strip()

        # [예외 방어] 검색기(벡터 DB/검색 API)는 외부 I/O라 실패한다.
        # 크래시 대신 error를 남기고, 라우터가 '정직한 거절' 경로로 보낸다.
        try:
            docs = await retriever.retrieve(query)
        except Exception as e:
            _logger.exception("문서 검색 실패 — 정직한 거절 경로로 저하")
            return {"error": f"문서 검색 실패: {e}", "retrieved_docs": []}

        _logger.info("문서 검색 완료: %d건 (query=%r)", len(docs), query)
        # State는 순수 데이터 버스 — Pydantic 객체 대신 dict로 실어 보낸다.
        return {"retrieved_docs": [d.model_dump() for d in docs]}

    return retrieve_docs


# ── 조건부 엣지: 근거가 없으면 LLM 진입 전에 끊는 순수 함수 ─────────
def route_after_retrieve(state: AgentState) -> str:
    """검색 결과를 보고 생성(LLM) 단계로 갈지, 정직한 거절로 갈지 결정한다.

    [왜 0건이면 LLM을 안 부르나? — 환각의 원천 차단]
    근거 문서가 없는데 LLM에게 답을 시키면 그 답은 100% 환각이다.
    비용을 아끼는 것을 넘어, 환각이 '생성될 기회 자체'를 없앤다.
    """
    if state.get("error") or not state.get("retrieved_docs"):
        return "no_evidence"
    return "generate_answer"


# ═════════════════════════════════════════════════════════════
# 2) 생성(Generation) — 문서 근거로만 답하도록 강제
# ═════════════════════════════════════════════════════════════
def make_generate_node(provider: LLMProvider):
    """[노드 팩토리] 검색된 문서만 근거로 답변을 생성하는 노드를 만든다."""

    def generate_answer(state: AgentState) -> dict:
        chat = provider.get_chat_model()
        docs = state["retrieved_docs"]
        feedback = state.get("grounding_feedback")

        # 재작성이면 접지 검증의 반려 사유를 프롬프트에 명시한다. (자가 수정 루프)
        feedback_block = ""
        if feedback:
            feedback_block = (
                f"\n\n[직전 검증 실패 사유 — 반드시 수정하세요]\n{feedback}"
            )

        docs_block = "\n\n".join(
            f"[{d['doc_id']}] {d['title']}\n{d['content']}" for d in docs
        )
        system = (
            "당신은 사내 규정 문서에만 근거해 답하는 B2B SaaS 고객지원 어시스턴트입니다. "
            "반드시 지키세요: (1) 아래 제공된 문서에 있는 사실만 답합니다. "
            "(2) 문서에 없는 수치·기간·조건을 절대 지어내지 않습니다. "
            "(3) 근거 문서의 ID를 답변 끝에 '(출처: 문서ID)' 형식으로 인용합니다. "
            "(4) 전문적이고 정중한 '~입니다' 체로 답합니다."
        )
        human = f"[사내 규정 문서]\n{docs_block}\n\n[고객 질문]\n{state['user_query']}{feedback_block}"

        # [예외 방어] 생성 LLM이 죽으면 빈 답변을 반환한다. 빈 답변은 접지
        # 검증이 반드시 반려하므로, 자가 수정 루프가 재시도를 겸한다.
        # 지속 장애면 retry 한도 초과로 give_up(안전 거절)에 수렴한다.
        try:
            response = chat.invoke([("system", system), ("human", human)])
        except Exception:
            _logger.exception("답변 생성 LLM 호출 실패 — 빈 답변 반환(접지 루프가 재시도)")
            return {"policy_answer": ""}

        _logger.debug("답변 생성 완료 (retry_count=%d)", state.get("retry_count", 0))
        return {"policy_answer": content_to_text(response.content)}

    return generate_answer


# ═════════════════════════════════════════════════════════════
# 3) 접지 검증(Grounding Check) — 결정론적 환각 방어 게이트
# ═════════════════════════════════════════════════════════════
def _grounding_gate(state: AgentState) -> tuple[bool, str]:
    """답변이 검색 문서에 접지되었는지 '코드'로 검사한다.

    [왜 LLM이 아니라 코드인가? — 면접 포인트]
    "환각을 잡는 검증기가 스스로 환각하면 방어가 아니다." 그래서 접지 검증은
    결정론적 코드로 한다. 두 가지를 본다:
      (a) 인용 존재  : 검색된 문서 ID가 답변에 하나 이상 인용되었는가.
      (b) 수치 접지  : 답변에 등장하는 '모든 숫자'가 검색 문서 원문에 실존하는가.
                       (규정 답변에서 환각이 가장 치명적인 지점이 바로 수치다 —
                        "환불은 30일 이내"처럼 그럴듯한 거짓 숫자를 여기서 잡는다)
    반환: (통과여부, 실패사유). 트렌드 워커의 _code_gate와 같은 규약.
    """
    answer = state.get("policy_answer", "") or ""
    docs = state.get("retrieved_docs") or []

    if not answer.strip():
        return False, "답변이 비어 있습니다. 문서 근거로 답변을 작성하세요."

    # (a) 인용 존재 검사
    cited = [d["doc_id"] for d in docs if d["doc_id"] in answer]
    if not cited:
        ids = ", ".join(d["doc_id"] for d in docs)
        return False, f"근거 문서 ID가 인용되지 않았습니다. 사용한 문서를 (출처: {ids} 중 해당 ID)로 인용하세요."

    # (b) 수치 접지 검사 — 답변 속 모든 숫자는 문서 어딘가에 있어야 한다.
    corpus = " ".join(f"{d['doc_id']} {d['title']} {d['content']}" for d in docs)
    for num in re.findall(r"\d+(?:\.\d+)?", answer):
        # 숫자 경계 매칭(트렌드 워커의 _growth_mentioned와 같은 이유):
        # "90"이 "190"의 일부에 매칭되는 오탐을 막는다.
        if not re.search(rf"(?<![\d.]){re.escape(num)}(?!\.?\d)", corpus):
            return False, (
                f"문서에 존재하지 않는 수치 '{num}'가 답변에 포함되었습니다. "
                "문서에 있는 수치만 사용하세요."
            )

    return True, ""


def grounding_check(state: AgentState) -> dict:
    """[노드] 접지 게이트를 적용하고 합격/반려 도장을 찍는다."""
    retry = state.get("retry_count", 0)

    ok, reason = _grounding_gate(state)
    if not ok:
        _logger.info("접지 검증 반려 (retry %d→%d): %s", retry, retry + 1, reason)
        return {
            "grounding_passed": False,
            "grounding_feedback": reason,
            "retry_count": retry + 1,
        }

    _logger.info("접지 검증 통과 (retry_count=%d)", retry)
    return {"grounding_passed": True, "grounding_feedback": ""}


# ── 조건부 엣지: 접지 결과에 따른 분기(순수 함수) ─────────────────
def route_after_grounding(state: AgentState) -> str:
    """접지 검증 결과를 보고 다음 목적지 키를 반환한다.

    세 갈래(트렌드 워커의 route_after_critic와 동일한 패턴):
      - 접지 확인                         → "finalize"
      - 반려 & retry 한도 초과            → "give_up"   (무한루프 방어)
      - 반려 & 아직 여유 있음             → "generate_answer" (재작성 루프)
    """
    if state.get("grounding_passed", False):
        return "finalize"
    if state.get("retry_count", 0) >= _MAX_RETRIES:
        return "give_up"
    return "generate_answer"


# ═════════════════════════════════════════════════════════════
# 4) 종료 노드들 — 어떤 경로든 계약을 만족하는 답변을 내보낸다
# ═════════════════════════════════════════════════════════════
def finalize(state: AgentState) -> dict:
    """[노드] 접지 확인된 답변을 PolicyAnswerOutput 계약으로 확정한다."""
    answer = state["policy_answer"]
    docs = state.get("retrieved_docs") or []
    citations = [d["doc_id"] for d in docs if d["doc_id"] in answer]

    try:
        output = PolicyAnswerOutput(answer=answer, citations=citations, grounded=True)
        _logger.info("규정 답변 확정: 인용 %d건 %s", len(citations), citations)
        return {"final_output": output.model_dump()}
    except ValidationError as e:
        _logger.error("규정 답변 계약(Pydantic) 검증 실패: %s", e)
        return {"error": f"최종 산출물 검증 실패: {e}"}


def no_evidence(state: AgentState) -> dict:
    """[노드] 근거 문서가 없을 때(0건 또는 검색 장애) 정직하게 거절한다.

    이것은 '실패'가 아니라 환각 방어의 정상 동작이다. 검색 장애였다면
    retrieve 노드가 남긴 error가 State에 그대로 남아 관찰 신호가 된다.
    """
    _logger.info("근거 문서 없음 — 정직한 거절 답변 반환 (error=%s)", state.get("error"))
    output = PolicyAnswerOutput(answer=_NO_EVIDENCE_ANSWER, citations=[], grounded=False)
    return {"final_output": output.model_dump()}


def give_up(state: AgentState) -> dict:
    """[노드] 접지 실패가 한도를 넘었을 때 안전한 거절 답변으로 탈출한다.

    트렌드 워커의 bail_out과 달리 error만 남기지 않고 '고객에게 줄 답변'을
    반드시 만든다 — 고객 응대 경로는 어떤 경우에도 답변 계약을 만족해야 한다.
    """
    _logger.warning(
        "접지 검증 한도(%d회) 초과 — 안전 거절로 탈출. 마지막 사유: %s",
        _MAX_RETRIES,
        state.get("grounding_feedback", "(없음)"),
    )
    output = PolicyAnswerOutput(answer=_GIVE_UP_ANSWER, citations=[], grounded=False)
    return {
        "final_output": output.model_dump(),
        "error": (
            f"접지 검증 {_MAX_RETRIES}회를 초과해 근거 있는 답변 생성에 실패했습니다. "
            f"마지막 사유: {state.get('grounding_feedback', '(없음)')}"
        ),
    }


# ═════════════════════════════════════════════════════════════
# 그래프 빌더
# ═════════════════════════════════════════════════════════════
def build_rag_worker_graph(provider: LLMProvider, retriever: Retriever | None = None):
    """규정 RAG 워커 서브 그래프를 빌드해 컴파일된 그래프를 반환한다.

    Args:
        provider: 답변 생성에 쓸 LLM 공급사(계약).
        retriever: 문서 검색기(계약). None이면 팩토리 기본값(dummy)을 쓴다.
                   테스트에선 고장난 검색기를 주입해 장애 방어를 검증한다.
    """
    retriever = retriever or get_retriever()

    graph = StateGraph(AgentState)

    # ── 노드 등록 ──
    graph.add_node("retrieve_docs", make_retrieve_node(retriever))   # async, 코드
    graph.add_node("generate_answer", make_generate_node(provider))  # LLM
    graph.add_node("grounding_check", grounding_check)               # 순수 코드
    graph.add_node("finalize", finalize)                             # 계약 확정
    graph.add_node("no_evidence", no_evidence)                       # 정직한 거절
    graph.add_node("give_up", give_up)                               # 한도 초과 탈출

    # ── 엣지(흐름) ──
    graph.add_edge(START, "retrieve_docs")

    # 근거가 없으면 LLM 진입 전에 끊는다(환각 원천 차단 + 비용 0).
    graph.add_conditional_edges(
        "retrieve_docs",
        route_after_retrieve,
        {
            "generate_answer": "generate_answer",
            "no_evidence": "no_evidence",
        },
    )

    graph.add_edge("generate_answer", "grounding_check")

    # 접지 결과에 따라 확정/재작성 루프/안전 탈출로 분기.
    graph.add_conditional_edges(
        "grounding_check",
        route_after_grounding,
        {
            "finalize": "finalize",
            "give_up": "give_up",
            "generate_answer": "generate_answer",  # ← 환각 검출 시 재작성 루프
        },
    )

    graph.add_edge("finalize", END)
    graph.add_edge("no_evidence", END)
    graph.add_edge("give_up", END)

    return graph.compile()
