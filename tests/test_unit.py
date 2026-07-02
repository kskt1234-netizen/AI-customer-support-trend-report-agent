"""
test_unit.py — 단위 테스트. '부품 하나하나'를 격리해 검증한다.

[단위 테스트란? — 면접 포인트]
그래프 전체가 아니라, 개별 함수/클래스 하나의 동작만 본다. 입력→출력이
명확한 순수 함수(연산, 라우팅, 게이트)와 추상화 계약(ABC, 팩토리)이 대상이다.
빠르고, LLM이 필요 없고, 실패하면 '어느 부품'이 깨졌는지 바로 안다.
"""

from __future__ import annotations

import pytest

from src.llm.base import LLMProvider, content_to_text
from src.llm.factory import get_provider
from src.llm.openai_provider import OpenAIProvider
from src.retrieval.base import Retriever
from src.retrieval.dummy_retriever import KeywordDummyRetriever
from src.retrieval.factory import get_retriever
from src.schemas import PolicyAnswerOutput, TrendReportOutput
from src.sub_graphs.rag_worker import (
    _MAX_RETRIES as _RAG_MAX_RETRIES,
    _grounding_gate,
    route_after_grounding,
    route_after_retrieve,
)
from src.sub_graphs.trend_report import (
    _MAX_RETRIES,
    _code_gate,
    _growth_mentioned,
    compute_growth,
    make_llm_gate,
    route_after_compute,
    route_after_critic,
)
from src.main_graph import route_by_intent
from src.schemas import IntentType


# ─────────────────────────────────────────────────────────────
# 테스트 전용 미니 가짜들 — LLM 게이트의 '장애' 시나리오를 통제한다
# ─────────────────────────────────────────────────────────────
class _RaisingChat:
    """invoke가 항상 예외를 던지는 챗 모델 — LLM 장애 시뮬레이션."""

    def invoke(self, messages):
        raise RuntimeError("LLM API down (simulated)")


class _StaticChat:
    """항상 고정 텍스트를 반환하는 챗 모델."""

    def __init__(self, text: str) -> None:
        self._text = text

    def invoke(self, messages):
        class _Msg:
            content = self._text

        return _Msg()


class _ChatOnlyProvider(LLMProvider):
    """주어진 챗 모델만 제공하는 공급사 — 게이트 단위 테스트용."""

    def __init__(self, chat) -> None:
        self._chat = chat

    def classify_intent(self, user_query):  # 이 테스트에선 쓰이지 않음
        raise NotImplementedError

    def get_chat_model(self):
        return self._chat


# ─────────────────────────────────────────────────────────────
# 1) 성장률 연산 — 결정론적 수식 검증 (LLM 무관)
# ─────────────────────────────────────────────────────────────
class TestComputeGrowth:
    def test_basic_growth(self):
        """100 → 115.4 이면 15.4% 여야 한다(고정값 시나리오)."""
        state = {"raw_search_data": {"last_year_revenue": 100.0, "this_year_revenue": 115.4}}
        assert compute_growth(state)["market_growth_rate"] == 15.4

    def test_negative_growth(self):
        """매출이 줄면 음수 성장률이 나와야 한다."""
        state = {"raw_search_data": {"last_year_revenue": 200.0, "this_year_revenue": 150.0}}
        assert compute_growth(state)["market_growth_rate"] == -25.0

    def test_zero_last_year_is_guarded(self):
        """작년 매출 0(0으로 나누기)일 때 크래시 대신 0.0으로 방어해야 한다."""
        state = {"raw_search_data": {"last_year_revenue": 0.0, "this_year_revenue": 50.0}}
        assert compute_growth(state)["market_growth_rate"] == 0.0

    def test_non_numeric_revenue_sets_error(self):
        """수치가 문자열 등 비정상 타입이면 크래시 대신 error로 전환해야 한다."""
        state = {"raw_search_data": {"last_year_revenue": "백억", "this_year_revenue": 115.4}}
        out = compute_growth(state)
        assert "market_growth_rate" not in out
        assert "유효하지 않습니다" in out["error"]

    def test_missing_data_sets_error(self):
        """수집 데이터 자체가 없으면(수집 노드 이상) KeyError 대신 error를 남겨야 한다."""
        out = compute_growth({})
        assert "error" in out

    def test_upstream_error_is_not_overwritten(self):
        """앞 노드가 남긴 error가 있으면 연산을 건너뛰고 아무것도 덮어쓰지 않아야 한다."""
        assert compute_growth({"error": "수집 실패"}) == {}


# ─────────────────────────────────────────────────────────────
# 2) 코드 게이트(1단 Critic) — 형식/존재 검사 검증
# ─────────────────────────────────────────────────────────────
class TestCodeGate:
    def _state(self, draft: str) -> dict:
        return {
            "draft_report": draft,
            "market_growth_rate": 15.4,
            "sources": ["시장조사기관 A 2024 연간 리포트", "업계 뉴스레터 B"],
        }

    def test_pass_when_number_and_source_present(self):
        ok, _ = _code_gate(self._state("성장률은 15.4%이며 시장조사기관 A 2024 연간 리포트를 참고했습니다."))
        assert ok is True

    def test_fail_when_number_missing(self):
        ok, reason = _code_gate(self._state("성장했습니다. 시장조사기관 A 2024 연간 리포트 참고."))
        assert ok is False and "15.4" in reason  # 숫자 누락 사유

    def test_fail_when_source_missing(self):
        ok, reason = _code_gate(self._state("성장률은 15.4%입니다. 출처는 모르겠습니다."))
        assert ok is False and "출처" in reason  # 출처 누락 사유

    def test_fail_when_number_is_substring_of_another(self):
        """'115.4' 속의 '15.4'는 성장률 언급이 아니다 — 부분 문자열 오탐 방지."""
        ok, _ = _code_gate(self._state("올해 매출은 115.4억이며 시장조사기관 A 2024 연간 리포트 참고."))
        assert ok is False

    def test_empty_draft_rejected(self):
        """빈 초안(LLM 호출 실패 등)은 반드시 반려되어야 한다."""
        ok, _ = _code_gate(self._state(""))
        assert ok is False


class TestGrowthMention:
    """성장률 수치 매칭의 엣지 케이스 — 오탐(false positive)과 미탐(false negative)."""

    def test_integer_growth_matches_natural_notation(self):
        """growth=15.0이면 LLM이 자연스럽게 쓰는 '15%'도 인정해야 한다(미탐 방지)."""
        assert _growth_mentioned("성장률은 15%입니다.", 15.0) is True
        assert _growth_mentioned("성장률은 15.0%입니다.", 15.0) is True

    def test_substring_of_bigger_number_is_not_a_match(self):
        """115.4 안의 15.4, 15.43 앞의 15.4는 매칭이 아니다(오탐 방지)."""
        assert _growth_mentioned("매출 115.4억을 달성했습니다.", 15.4) is False
        assert _growth_mentioned("정확히는 15.43%입니다.", 15.4) is False

    def test_negative_growth_is_matched(self):
        """음수 성장률(-25.0%)도 정확히 매칭되어야 한다."""
        assert _growth_mentioned("시장은 -25.0% 역성장했습니다.", -25.0) is True


# ─────────────────────────────────────────────────────────────
# 3) 라우터(조건부 엣지의 순수 함수) — 분기 키 검증
# ─────────────────────────────────────────────────────────────
class TestRouters:
    def test_route_by_intent_trend(self):
        assert route_by_intent({"intent": IntentType.TREND_REPORT.value}) == "trend_report"

    def test_route_by_intent_default_is_simple_chat(self):
        """알 수 없는/누락된 의도는 안전하게 simple_chat으로 가야 한다(방어적 기본값)."""
        assert route_by_intent({"intent": "WHATEVER"}) == "simple_chat"
        assert route_by_intent({}) == "simple_chat"

    def test_route_after_critic_pass(self):
        assert route_after_critic({"critic_passed": True}) == "finalize"

    def test_route_after_critic_retry(self):
        """불합격 & 한도 미만 → 재작성 루프(draft_report)."""
        state = {"critic_passed": False, "retry_count": 1}
        assert route_after_critic(state) == "draft_report"

    def test_route_after_critic_bailout_on_limit(self):
        """불합격 & 한도 도달 → bail_out (무한 루프 방어)."""
        state = {"critic_passed": False, "retry_count": _MAX_RETRIES}
        assert route_after_critic(state) == "bail_out"

    def test_route_after_compute_aborts_on_error(self):
        """수집/연산 단계 error → LLM 단계로 가지 않고 조기 탈출(abort)해야 한다."""
        assert route_after_compute({"error": "시장 데이터 수집 실패"}) == "abort"

    def test_route_after_compute_proceeds_when_clean(self):
        assert route_after_compute({"market_growth_rate": 15.4}) == "draft_report"

    def test_route_by_intent_policy_inquiry(self):
        """POLICY_INQUIRY 의도 → RAG 워커(policy_rag)로 라우팅되어야 한다."""
        assert route_by_intent({"intent": IntentType.POLICY_INQUIRY.value}) == "policy_rag"

    def test_route_after_retrieve_no_docs_goes_to_no_evidence(self):
        """검색 0건 → LLM 진입 전 정직한 거절 경로(환각 원천 차단)."""
        assert route_after_retrieve({"retrieved_docs": []}) == "no_evidence"
        assert route_after_retrieve({}) == "no_evidence"

    def test_route_after_retrieve_error_goes_to_no_evidence(self):
        """검색기 장애(error) → 문서가 있어 보여도 거절 경로로 저하."""
        state = {"error": "검색 실패", "retrieved_docs": [{"doc_id": "X"}]}
        assert route_after_retrieve(state) == "no_evidence"

    def test_route_after_retrieve_proceeds_with_docs(self):
        assert route_after_retrieve({"retrieved_docs": [{"doc_id": "X"}]}) == "generate_answer"

    def test_route_after_grounding_pass(self):
        assert route_after_grounding({"grounding_passed": True}) == "finalize"

    def test_route_after_grounding_retry(self):
        """접지 반려 & 한도 미만 → 재작성 루프."""
        state = {"grounding_passed": False, "retry_count": 1}
        assert route_after_grounding(state) == "generate_answer"

    def test_route_after_grounding_give_up_on_limit(self):
        """접지 반려 & 한도 도달 → 안전 거절로 탈출(무한 루프 방어)."""
        state = {"grounding_passed": False, "retry_count": _RAG_MAX_RETRIES}
        assert route_after_grounding(state) == "give_up"


# ─────────────────────────────────────────────────────────────
# 3.5) LLM 게이트의 장애 방어 — fail-closed 정책 검증
# ─────────────────────────────────────────────────────────────
class TestLLMGateDefense:
    _STATE = {"market_growth_rate": 15.4, "draft_report": "성장률은 15.4%입니다."}

    def test_gate_exception_fails_closed(self):
        """검수기 LLM이 죽으면 '무검수 통과'가 아니라 '반려'여야 한다(fail-closed)."""
        gate = make_llm_gate(_ChatOnlyProvider(_RaisingChat()))
        ok, reason = gate(self._STATE)
        assert ok is False
        assert "fail-closed" in reason

    def test_gate_empty_response_fails_closed(self):
        """검수기가 빈 응답을 주면 PASS로 오인하지 말고 반려해야 한다."""
        gate = make_llm_gate(_ChatOnlyProvider(_StaticChat("")))
        ok, _ = gate(self._STATE)
        assert ok is False

    def test_gate_pass_and_fail_parsing(self):
        """정상 판정 파싱: 'PASS'는 합격, 'FAIL: 사유'는 사유와 함께 반려."""
        assert make_llm_gate(_ChatOnlyProvider(_StaticChat("PASS")))(self._STATE) == (True, "")
        ok, reason = make_llm_gate(_ChatOnlyProvider(_StaticChat("FAIL: 톤이 가볍습니다")))(self._STATE)
        assert ok is False and reason == "톤이 가볍습니다"


# ─────────────────────────────────────────────────────────────
# 3.6) LLM 경계 텍스트 강제 변환 — content 형태 방어
# ─────────────────────────────────────────────────────────────
class TestContentToText:
    def test_plain_string_passthrough(self):
        assert content_to_text("안녕하세요") == "안녕하세요"

    def test_none_becomes_empty(self):
        assert content_to_text(None) == ""

    def test_content_block_list_is_flattened(self):
        """일부 모델은 콘텐츠 블록 리스트를 반환한다 — 텍스트만 이어붙여야 한다."""
        blocks = [{"type": "text", "text": "성장률은 "}, {"type": "text", "text": "15.4%"}]
        assert content_to_text(blocks) == "성장률은 15.4%"

    def test_string_list_is_joined(self):
        assert content_to_text(["가", "나"]) == "가나"


# ─────────────────────────────────────────────────────────────
# 3.7) 검색(Retriever) 추상화 계약 & 더미 구현 검증
# ─────────────────────────────────────────────────────────────
class TestRetrieverAbstraction:
    def test_abc_cannot_be_instantiated(self):
        """추상 계약은 직접 인스턴스화가 막혀야 한다(retrieve 구현 강제)."""
        with pytest.raises(TypeError):
            Retriever()  # type: ignore[abstract]

    def test_factory_returns_retriever_contract(self):
        """팩토리가 Retriever 계약 타입을 반환해야 한다(워커는 구체 타입을 모름)."""
        retriever = get_retriever("dummy")
        assert isinstance(retriever, Retriever)
        assert isinstance(retriever, KeywordDummyRetriever)

    def test_factory_rejects_unknown_retriever(self):
        """미등록 검색기는 조용히 넘기지 않고 ValueError로 즉시 터져야 한다."""
        with pytest.raises(ValueError):
            get_retriever("nonexistent-search")


class TestDummyRetriever:
    async def test_relevant_doc_ranked_first(self):
        """'환불 위약금' 질문 → 환불 정책 문서가 최상위로 나와야 한다."""
        docs = await KeywordDummyRetriever().retrieve("환불 위약금 규정 알려줘")
        assert docs and docs[0].doc_id == "POL-REFUND-001"

    async def test_irrelevant_query_returns_empty(self):
        """관련 없는 질문 → 빈 리스트(예외가 아니라 '근거 없음' 계약)."""
        assert await KeywordDummyRetriever().retrieve("오늘 점심 메뉴 추천해줘") == []

    async def test_empty_query_returns_empty(self):
        assert await KeywordDummyRetriever().retrieve("   ") == []

    async def test_top_k_is_respected(self):
        """여러 문서가 매칭돼도 top_k개만 반환해야 한다."""
        query = "환불이랑 계약 해지, 데이터 보관, 계정 보안 규정 전부 알려줘"
        docs = await KeywordDummyRetriever().retrieve(query, top_k=2)
        assert len(docs) == 2


# ─────────────────────────────────────────────────────────────
# 3.8) 접지 검증(Grounding Gate) — 결정론적 환각 방어
# ─────────────────────────────────────────────────────────────
class TestGroundingGate:
    _REFUND_DOC = {
        "doc_id": "POL-REFUND-001",
        "title": "환불 및 위약금 정책",
        "content": "결제일로부터 14일 이내 전액 환불, 이후 위약금 10% 공제.",
        "score": 1.0,
    }

    def _state(self, answer: str) -> dict:
        return {"policy_answer": answer, "retrieved_docs": [self._REFUND_DOC]}

    def test_grounded_answer_passes(self):
        ok, _ = _grounding_gate(
            self._state("환불은 14일 이내 전액 가능하며, 이후 위약금 10%가 공제됩니다. (출처: POL-REFUND-001)")
        )
        assert ok is True

    def test_missing_citation_rejected(self):
        """문서 ID 인용이 없으면 반려 — 근거 추적이 불가능한 답변."""
        ok, reason = _grounding_gate(self._state("환불은 14일 이내 가능합니다."))
        assert ok is False and "인용" in reason

    def test_hallucinated_number_rejected(self):
        """문서에 없는 수치('30일')는 그럴듯해도 반려 — 핵심 환각 방어."""
        ok, reason = _grounding_gate(
            self._state("환불은 30일 이내 가능합니다. (출처: POL-REFUND-001)")
        )
        assert ok is False and "30" in reason

    def test_empty_answer_rejected(self):
        """빈 답변(LLM 호출 실패 등)은 반드시 반려되어야 한다."""
        ok, _ = _grounding_gate(self._state(""))
        assert ok is False


# ─────────────────────────────────────────────────────────────
# 4) LLM 추상화 계약(ABC) & 팩토리(DIP/OCP) 검증
# ─────────────────────────────────────────────────────────────
class TestLLMAbstraction:
    def test_abc_cannot_be_instantiated(self):
        """추상 계약은 직접 인스턴스화가 막혀야 한다(메서드 구현 강제)."""
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_factory_returns_provider_contract(self, monkeypatch):
        """팩토리가 LLMProvider 계약 타입을 반환해야 한다(노드는 구체 타입을 모름)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-for-construct")
        provider = get_provider("openai")
        assert isinstance(provider, LLMProvider)
        assert isinstance(provider, OpenAIProvider)

    def test_factory_rejects_unknown_provider(self):
        """미등록 공급사는 조용히 넘기지 않고 ValueError로 즉시 터져야 한다."""
        with pytest.raises(ValueError):
            get_provider("nonexistent-llm")


# ─────────────────────────────────────────────────────────────
# 5) Pydantic 경계 계약 검증
# ─────────────────────────────────────────────────────────────
class TestSchemaContract:
    def test_valid_output_passes(self):
        out = TrendReportOutput(market_growth_rate=15.4, summary="요약", sources=["출처1"])
        assert out.has_required_evidence() is True

    def test_empty_sources_rejected(self):
        """출처가 비면 계약 위반(근거 없는 보고서)으로 검증 실패해야 한다."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TrendReportOutput(market_growth_rate=15.4, summary="요약", sources=[])

    def test_policy_grounded_requires_citations(self):
        """grounded=True인데 인용이 없는 모순 답변은 계약 위반으로 차단해야 한다."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PolicyAnswerOutput(answer="규정상 가능합니다.", citations=[], grounded=True)

    def test_policy_honest_refusal_is_valid(self):
        """근거 없음을 정직하게 밝힌 답변(grounded=False, 인용 0)은 정상 계약이다."""
        out = PolicyAnswerOutput(answer="문서에서 근거를 찾지 못했습니다.", citations=[], grounded=False)
        assert out.grounded is False and out.citations == []

    def test_policy_grounded_with_citation_is_valid(self):
        out = PolicyAnswerOutput(answer="14일 이내 환불 가능합니다. (출처: POL-REFUND-001)",
                                 citations=["POL-REFUND-001"], grounded=True)
        assert out.grounded is True
