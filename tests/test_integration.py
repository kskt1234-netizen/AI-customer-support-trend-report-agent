"""
test_integration.py — 통합 테스트. '조립된 전체 그래프'가 의도대로 흐르는지 검증한다.

[단위 vs 통합 — 면접 포인트]
- 단위(test_unit): 부품 하나(compute_growth, _code_gate, 라우터...)를 격리해 본다.
- 통합(여기)    : 그 부품들이 그래프로 '조립'됐을 때, START→END까지 올바른 경로로
                   흐르고 State가 단계마다 제대로 채워지는지 본다.

[왜 여기서도 FakeProvider인가?]
실제 LLM은 비결정적이라 통합 '로직'을 검증하기엔 불안정하다. 대신 ②번 추상화로
만든 가짜 공급사를 주입해, '분류/초안/검수 응답을 우리가 통제'한 채 그래프 위상과
분기·루프·방어가 맞는지를 결정론적으로 검증한다. (실제 API 정확도는 real 모드의 몫)
"""

from __future__ import annotations

import asyncio

import pytest

from src.llm.base import LLMProvider
from src.retrieval.base import RetrievedDoc, Retriever
from src.schemas import IntentClassification, IntentType
from src.main_graph import build_main_graph
from src.sub_graphs.rag_worker import _MAX_RETRIES as _RAG_MAX_RETRIES
from src.sub_graphs.rag_worker import build_rag_worker_graph
from src.sub_graphs.trend_report import build_trend_report_graph, _MAX_RETRIES


# ─────────────────────────────────────────────────────────────
# 테스트용 가짜 공급사들 (LLMProvider 계약을 구현)
# ─────────────────────────────────────────────────────────────
class _ScriptedChat:
    """호출 순서대로 미리 정해둔 텍스트를 반환하는 가짜 챗 모델.

    서브그래프에서 draft_report와 llm_gate가 번갈아 .invoke를 부른다.
    그 순서에 맞춰 초안/판정을 스크립트로 통제한다.
    """

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self._i = 0

    def invoke(self, messages):
        text = self._script[self._i] if self._i < len(self._script) else self._script[-1]
        self._i += 1

        class _Msg:
            content = text

        return _Msg()


class FakeProvider(LLMProvider):
    """분류 의도와 챗 응답을 모두 통제하는 가짜 공급사."""

    def __init__(self, intent: IntentType, chat_script: list[str] | None = None) -> None:
        self._intent = intent
        self._chat = _ScriptedChat(chat_script or ["(가짜) 응답"])

    def classify_intent(self, user_query: str) -> IntentClassification:
        return IntentClassification(intent=self._intent, reasoning="테스트 고정 분류")

    def get_chat_model(self):
        return self._chat


class _RaisingChat:
    """invoke가 항상 예외를 던지는 챗 모델 — LLM 장애 시뮬레이션."""

    def invoke(self, messages):
        raise RuntimeError("LLM API down (simulated)")


class _BrokenClassifierProvider(LLMProvider):
    """분류기만 죽고 챗 모델은 살아 있는 공급사 — 분류 폴백 검증용."""

    def __init__(self, chat_script: list[str]) -> None:
        self._chat = _ScriptedChat(chat_script)

    def classify_intent(self, user_query: str) -> IntentClassification:
        raise RuntimeError("classifier down (simulated)")

    def get_chat_model(self):
        return self._chat


class _DeadProvider(LLMProvider):
    """모든 LLM 호출이 죽는 공급사 — 전면 장애 시뮬레이션.

    또 하나의 용도: 이 공급사를 주입했는데도 테스트가 '고정 문구'를 받았다면,
    그 경로에서 LLM이 아예 호출되지 않았다는 사실 자체가 증명된다.
    """

    def classify_intent(self, user_query: str) -> IntentClassification:
        raise RuntimeError("LLM down (simulated)")

    def get_chat_model(self):
        return _RaisingChat()


_GOOD_DRAFT = (
    "올해 시장 매출 성장률은 15.4%입니다. 업계 SaaS 도입률 증가가 견인했습니다. "
    "(출처: 시장조사기관 A 2024 연간 리포트)"
)
_BAD_DRAFT = "ㅇㅇ 성장률 그런거 잘 모르겠고 대충 씀"

# ── RAG 워커용 고정 시나리오 ──────────────────────────────────
# 더미 코퍼스의 환불 정책 문서(POL-REFUND-001: 14일 전액 환불, 위약금 10%) 기준.
_POLICY_QUERY = "환불 위약금 규정 알려줘"
_GOOD_POLICY_ANSWER = (
    "환불은 결제일로부터 14일 이내에 요청하시면 전액 가능하며, "
    "이후에는 위약금 10%가 공제됩니다. (출처: POL-REFUND-001)"
)
# 숫자 '30일'은 환불 문서에 없다 — 그럴듯하지만 명백한 환각.
_HALLUCINATED_ANSWER = "환불은 30일 이내에만 가능합니다. (출처: POL-REFUND-001)"


# ─────────────────────────────────────────────────────────────
# 1) 메인 그래프 라우팅 통합 — 의도에 따라 올바른 워커로 가는가
# ─────────────────────────────────────────────────────────────
class TestMainGraphRouting:
    def test_simple_chat_route(self):
        """SIMPLE_CHAT 의도 → simple_chat 노드가 답변을 채워야 한다."""
        app = build_main_graph(FakeProvider(IntentType.SIMPLE_CHAT, ["안녕하세요, 도와드릴게요."]))
        out = asyncio.run(app.ainvoke({"user_query": "비밀번호 바꾸는 법", "retry_count": 0}))
        assert out["intent"] == "SIMPLE_CHAT"
        assert out["chat_response"] == "안녕하세요, 도와드릴게요."

    def test_trend_report_route_reaches_subgraph(self):
        """TREND_REPORT 의도 → 서브그래프가 돌아 final_output이 채워져야 한다."""
        app = build_main_graph(FakeProvider(IntentType.TREND_REPORT, [_GOOD_DRAFT, "PASS"]))
        out = asyncio.run(app.ainvoke({"user_query": "매출 트렌드 분석", "retry_count": 0}))
        assert out["intent"] == "TREND_REPORT"
        assert out["final_output"] is not None
        assert out["final_output"]["market_growth_rate"] == 15.4

    def test_policy_route_reaches_rag_worker(self):
        """POLICY_INQUIRY 의도 → RAG 워커가 돌아 접지된 답변이 나와야 한다."""
        app = build_main_graph(FakeProvider(IntentType.POLICY_INQUIRY, [_GOOD_POLICY_ANSWER]))
        out = asyncio.run(app.ainvoke({"user_query": _POLICY_QUERY, "retry_count": 0}))
        assert out["intent"] == "POLICY_INQUIRY"
        assert out["final_output"]["grounded"] is True
        assert "POL-REFUND-001" in out["final_output"]["citations"]


# ─────────────────────────────────────────────────────────────
# 2) 트렌드 서브그래프 전체 흐름 — 수집→연산→초안→검수→마무리
# ─────────────────────────────────────────────────────────────
class TestTrendSubgraph:
    def _run(self, chat_script: list[str]) -> dict:
        provider = FakeProvider(IntentType.TREND_REPORT, chat_script)
        app = build_trend_report_graph(provider)
        return asyncio.run(app.ainvoke({"user_query": "x", "retry_count": 0}))

    def test_happy_path(self):
        """정상: 좋은 초안 + LLM PASS → 합격, final_output 생성."""
        out = self._run([_GOOD_DRAFT, "PASS"])
        assert out["critic_passed"] is True
        assert out["final_output"]["market_growth_rate"] == 15.4
        assert len(out["final_output"]["sources"]) == 2  # 병렬 수집 출처 2개
        assert out.get("error") is None

    def test_self_correction_loop(self):
        """재작성 루프: 1차 불량(코드게이트 반려) → 2차 양호 → 합격."""
        out = self._run([_BAD_DRAFT, _GOOD_DRAFT, "PASS"])
        assert out["retry_count"] == 1  # 한 바퀴 돌았다
        assert out["critic_passed"] is True
        assert out["final_output"] is not None

    def test_infinite_loop_defense(self):
        """무한 루프 방어: 계속 불량 → retry 한도 초과 → bail_out(error)."""
        out = self._run([_BAD_DRAFT])  # 항상 불량 초안
        assert out["retry_count"] == _MAX_RETRIES
        assert out.get("final_output") is None
        assert out.get("error") is not None

    def test_llm_gate_catches_bad_context(self):
        """2단 게이트: 코드게이트는 통과하나(숫자·출처 존재) LLM이 FAIL →
        의미/톤 문제로 반려되어 재작성 후 합격하는지."""
        # 1차 초안: 숫자(15.4)와 출처는 있지만 맥락이 엉뚱 → 코드게이트 통과, LLM FAIL
        bad_context = "퇴사율이 15.4%로 급증했습니다. (출처: 시장조사기관 A 2024 연간 리포트)"
        out = self._run([bad_context, "FAIL: 성장률이 엉뚱한 맥락에 쓰임", _GOOD_DRAFT, "PASS"])
        assert out["retry_count"] == 1  # LLM 게이트에서 한 번 반려됨
        assert out["critic_passed"] is True
        assert out["final_output"] is not None


# ─────────────────────────────────────────────────────────────
# 3) 우아한 저하(Graceful Degradation) — 장애 상황에서도 크래시 없이
# ─────────────────────────────────────────────────────────────
class TestGracefulDegradation:
    """LLM/외부 API가 죽어도 허브는 크래시 대신 '안전한 무언가'를 돌려줘야 한다."""

    def test_classifier_outage_falls_back_to_simple_chat(self):
        """분류기가 죽으면 → SIMPLE_CHAT 폴백 라우팅 → 살아 있는 챗이 답변."""
        app = build_main_graph(_BrokenClassifierProvider(["무엇을 도와드릴까요?"]))
        out = asyncio.run(app.ainvoke({"user_query": "환불 규정 알려줘", "retry_count": 0}))
        assert out["intent"] == "SIMPLE_CHAT"
        assert out["chat_response"] == "무엇을 도와드릴까요?"

    def test_total_llm_outage_returns_polite_fallback(self):
        """전면 장애: 분류도 챗도 죽음 → 크래시 대신 정중한 고정 문구 + error 기록."""
        app = build_main_graph(_DeadProvider())
        out = asyncio.run(app.ainvoke({"user_query": "안녕하세요", "retry_count": 0}))
        assert out["intent"] == "SIMPLE_CHAT"
        assert "죄송합니다" in out["chat_response"]
        assert "LLM 호출 실패" in out["error"]

    def test_empty_query_short_circuits_without_llm(self):
        """빈 질문은 LLM을 아예 호출하지 않고 재질문 안내로 응답해야 한다.

        죽은 공급사(_DeadProvider)를 주입했는데 '고정 안내 문구'가 나왔다면,
        LLM이 호출되지 않았음이 증명된다(호출됐다면 폴백 문구가 나왔을 것).
        """
        app = build_main_graph(_DeadProvider())
        out = asyncio.run(app.ainvoke({"user_query": "   ", "retry_count": 0}))
        assert out["intent"] == "SIMPLE_CHAT"
        assert "질문이 비어 있습니다" in out["chat_response"]
        assert out.get("error") is None  # 예외 경로를 타지 않았다


# ─────────────────────────────────────────────────────────────
# 4) 트렌드 서브그래프의 장애 방어 — 수집 실패·LLM 장애
# ─────────────────────────────────────────────────────────────
class TestTrendSubgraphDefense:
    def test_draft_llm_outage_bails_out_safely(self):
        """초안 LLM 전면 장애 → 빈 초안 반려 루프 → 한도 도달 → bail_out(크래시 없음)."""
        provider = FakeProvider(IntentType.TREND_REPORT)
        provider._chat = _RaisingChat()  # 챗만 장애로 교체
        app = build_trend_report_graph(provider)
        out = asyncio.run(app.ainvoke({"user_query": "x", "retry_count": 0}))
        assert out.get("final_output") is None
        assert out["retry_count"] == _MAX_RETRIES
        assert "자가 수정" in out["error"]

    def test_market_search_failure_aborts_before_llm(self, monkeypatch):
        """필수 입력(시장 데이터) 수집 실패 → LLM 단계 진입 전에 조기 탈출.

        _DeadProvider식 증명: 챗이 죽어 있는데도 error가 '수집 실패'라면,
        draft_report(LLM)까지 가지 않고 상류에서 끊었다는 뜻이다.
        """
        async def _boom(query: str) -> dict:
            raise ConnectionError("search API down (simulated)")

        monkeypatch.setattr("src.sub_graphs.trend_report._search_market_data", _boom)
        provider = FakeProvider(IntentType.TREND_REPORT)
        provider._chat = _RaisingChat()
        app = build_trend_report_graph(provider)
        out = asyncio.run(app.ainvoke({"user_query": "x", "retry_count": 0}))
        assert "시장 데이터 수집 실패" in out["error"]
        assert out.get("final_output") is None
        assert out.get("draft_report") is None  # LLM 초안 단계에 진입하지 않았다
        assert out["retry_count"] == 0  # 재시도 루프도 돌지 않았다

    def test_competitor_failure_degrades_to_partial_report(self, monkeypatch):
        """보조 입력(경쟁사) 수집 실패 → 실패가 아니라 '부분 저하'로 계속 진행.

        시장 데이터만으로 보고서를 완성하고, 출처는 1개(시장)만 남는다.
        """
        async def _boom(query: str) -> dict:
            raise ConnectionError("competitor API down (simulated)")

        monkeypatch.setattr("src.sub_graphs.trend_report._search_competitor_data", _boom)
        provider = FakeProvider(IntentType.TREND_REPORT, [_GOOD_DRAFT, "PASS"])
        app = build_trend_report_graph(provider)
        out = asyncio.run(app.ainvoke({"user_query": "x", "retry_count": 0}))
        assert out["critic_passed"] is True
        assert out["final_output"]["sources"] == ["시장조사기관 A 2024 연간 리포트"]
        assert out.get("error") is None


# ─────────────────────────────────────────────────────────────
# 5) 규정 RAG 워커 — 접지 검증(환각 방어)·정직한 거절·장애 방어
# ─────────────────────────────────────────────────────────────
class TestRAGWorker:
    def _run(self, chat_script: list[str], query: str = _POLICY_QUERY) -> dict:
        provider = FakeProvider(IntentType.POLICY_INQUIRY, chat_script)
        app = build_rag_worker_graph(provider)
        return asyncio.run(app.ainvoke({"user_query": query, "retry_count": 0}))

    def test_happy_path_grounded_answer(self):
        """정상: 문서 근거 답변 + 인용 → 접지 통과, grounded=True로 확정."""
        out = self._run([_GOOD_POLICY_ANSWER])
        assert out["grounding_passed"] is True
        assert out["final_output"]["grounded"] is True
        assert out["final_output"]["citations"] == ["POL-REFUND-001"]
        assert out.get("error") is None

    def test_hallucination_is_caught_and_corrected(self):
        """환각 자가 수정: 문서에 없는 '30일' 답변 → 접지 반려 → 재작성 후 합격."""
        out = self._run([_HALLUCINATED_ANSWER, _GOOD_POLICY_ANSWER])
        assert out["retry_count"] == 1  # 환각이 정확히 한 번 잡혔다
        assert out["final_output"]["grounded"] is True

    def test_persistent_hallucination_gives_safe_refusal(self):
        """환각이 반복되면 → 한도 초과 → 지어낸 답 대신 안전한 거절 답변."""
        out = self._run([_HALLUCINATED_ANSWER])  # 스크립트 소진 후 같은 답 반복
        assert out["retry_count"] == _RAG_MAX_RETRIES
        assert out["final_output"]["grounded"] is False
        assert out["final_output"]["citations"] == []
        assert "접지 검증" in out["error"]

    def test_no_evidence_refuses_honestly_without_llm(self):
        """검색 0건 → LLM을 아예 호출하지 않고 정직하게 거절해야 한다.

        죽은 챗 모델을 꽂았는데도 거절 답변이 정상적으로 나왔다면,
        LLM이 호출되지 않았음(환각이 생성될 기회 자체가 없었음)이 증명된다.
        """
        provider = FakeProvider(IntentType.POLICY_INQUIRY)
        provider._chat = _RaisingChat()
        app = build_rag_worker_graph(provider)
        out = asyncio.run(app.ainvoke({"user_query": "오늘 점심 메뉴 추천해줘", "retry_count": 0}))
        assert out["final_output"]["grounded"] is False
        assert "근거를 찾지 못했습니다" in out["final_output"]["answer"]
        assert out.get("error") is None  # 장애가 아니라 정상적인 '근거 없음'

    def test_retriever_outage_degrades_to_refusal(self):
        """검색기 장애 → 크래시 대신 error 기록 + 정직한 거절 답변."""
        class _BoomRetriever(Retriever):
            async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
                raise ConnectionError("vector DB down (simulated)")

        provider = FakeProvider(IntentType.POLICY_INQUIRY)
        provider._chat = _RaisingChat()  # LLM까지 죽어 있어도 답변은 나와야 한다
        app = build_rag_worker_graph(provider, retriever=_BoomRetriever())
        out = asyncio.run(app.ainvoke({"user_query": _POLICY_QUERY, "retry_count": 0}))
        assert "문서 검색 실패" in out["error"]
        assert out["final_output"]["grounded"] is False

    def test_llm_outage_gives_safe_refusal(self):
        """생성 LLM 전면 장애 → 빈 답변 반려 루프 → 한도 도달 → 안전 거절."""
        provider = FakeProvider(IntentType.POLICY_INQUIRY)
        provider._chat = _RaisingChat()
        app = build_rag_worker_graph(provider)
        out = asyncio.run(app.ainvoke({"user_query": _POLICY_QUERY, "retry_count": 0}))
        assert out["retry_count"] == _RAG_MAX_RETRIES
        assert out["final_output"]["grounded"] is False
        assert out["error"]
