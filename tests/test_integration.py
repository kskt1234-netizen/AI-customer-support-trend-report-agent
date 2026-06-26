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
from src.schemas import IntentClassification, IntentType
from src.main_graph import build_main_graph
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


_GOOD_DRAFT = (
    "올해 시장 매출 성장률은 15.4%입니다. 업계 SaaS 도입률 증가가 견인했습니다. "
    "(출처: 시장조사기관 A 2024 연간 리포트)"
)
_BAD_DRAFT = "ㅇㅇ 성장률 그런거 잘 모르겠고 대충 씀"


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
