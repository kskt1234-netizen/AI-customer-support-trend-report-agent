"""
test_unit.py — 단위 테스트. '부품 하나하나'를 격리해 검증한다.

[단위 테스트란? — 면접 포인트]
그래프 전체가 아니라, 개별 함수/클래스 하나의 동작만 본다. 입력→출력이
명확한 순수 함수(연산, 라우팅, 게이트)와 추상화 계약(ABC, 팩토리)이 대상이다.
빠르고, LLM이 필요 없고, 실패하면 '어느 부품'이 깨졌는지 바로 안다.
"""

from __future__ import annotations

import pytest

from src.llm.base import LLMProvider
from src.llm.factory import get_provider
from src.llm.openai_provider import OpenAIProvider
from src.schemas import TrendReportOutput
from src.sub_graphs.trend_report import (
    _MAX_RETRIES,
    _code_gate,
    compute_growth,
    route_after_critic,
)
from src.main_graph import route_by_intent
from src.schemas import IntentType


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
