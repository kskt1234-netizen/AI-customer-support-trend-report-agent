"""
test_harness.py — 오케스트레이터 의도 분류 자동 채점기(하네스).

[투트랙(Two-track) 구조 — 면접 핵심 어필]
이 하네스는 환경변수 TEST_MODE 하나로 두 모드를 오간다. '테스트 코드는 그대로'
두고 주입할 LLM 공급사만 바꾼다(②번 추상화가 여기서 또 값을 발휘한다).

  - TEST_MODE=mock (기본, CI의 매 Push):
      골든셋의 expected_intent를 그대로 돌려주는 _GoldenMockProvider를 주입한다.
      → API 호출 0, 무료, 빠름, 결정론적. '라우팅/그래프 로직'을 검증한다.
      (분류 '성능'이 아니라, 그래프가 의도대로 흐르는지를 본다.)

  - TEST_MODE=real (수동 트리거/특정 환경):
      진짜 OpenAIProvider를 주입한다. 실제 LLM이 골든셋을 분류하게 해
      '분류 정확도'를 측정한다. OPENAI_API_KEY가 필요하고 비용이 든다.

합격 기준: 정확도 >= pass_threshold(기본 0.9 = 90%).

[왜 이렇게 나눴나?]
실제 API를 매 Push마다 부르면 느리고 비싸고, 외부 장애에 CI가 흔들린다.
평소엔 Mock으로 로직만 빠르게 지키고, 정확도 측정은 의도적으로 가끔(수동) 돈다.
→ 비용 최적화와 안정성을 동시에. (LLMOps)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.llm.base import LLMProvider
from src.schemas import IntentClassification, IntentType

# 골든셋 파일 경로(이 파일과 같은 폴더).
_CASES_PATH = Path(__file__).parent / "test_cases.json"


def _load_golden() -> dict:
    """골든셋 JSON을 읽어 반환한다."""
    with _CASES_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# mock 모드 전용: 골든셋 정답을 그대로 돌려주는 가짜 공급사
# ─────────────────────────────────────────────────────────────
class _GoldenMockProvider(LLMProvider):
    """query -> expected_intent 매핑을 미리 들고 있다가 그대로 반환하는 Mock.

    [무엇을 검증하나?]
    이 Mock은 분류를 '항상 정답'으로 만든다. 그래서 mock 모드의 테스트는
    '분류가 정확한가'가 아니라 '정답 의도가 주어졌을 때 그래프가 올바른
    경로로 흐르고 State에 제대로 반영되는가'(=라우팅/로직)를 검증한다.
    """

    def __init__(self, query_to_intent: dict[str, str]) -> None:
        self._map = query_to_intent

    def classify_intent(self, user_query: str) -> IntentClassification:
        intent_str = self._map[user_query]  # 골든셋에 있는 질문만 들어온다
        return IntentClassification(
            intent=IntentType(intent_str),
            reasoning="(mock) 골든셋 정답을 그대로 반환",
        )

    def get_chat_model(self):  # mock 모드 분류 테스트에선 쓰이지 않음
        raise NotImplementedError("mock 분류 테스트에서는 챗 모델을 사용하지 않습니다.")


def _build_provider(golden: dict) -> LLMProvider:
    """TEST_MODE에 따라 주입할 LLMProvider를 고른다. (투트랙 스위치)"""
    mode = os.getenv("TEST_MODE", "mock").lower().strip()

    if mode == "real":
        # 실제 OpenAI 분류. 키 없으면 명확히 스킵 사유를 남긴다.
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("TEST_MODE=real 인데 OPENAI_API_KEY가 없어 건너뜁니다.")
        from src.llm.factory import get_provider
        return get_provider("openai")

    # 기본 mock: 골든셋 정답 매핑을 주입.
    query_to_intent = {c["query"]: c["expected_intent"] for c in golden["cases"]}
    return _GoldenMockProvider(query_to_intent)


def test_intent_classification_accuracy() -> None:
    """골든셋으로 '분류기(classify_intent)'만 격리해 정확도를 채점한다.

    [왜 그래프 전체를 돌리지 않고 classify_intent만 부르나? — 관심사 분리]
    이 테스트의 목적은 '의도 분류 성능'이다. 그래서 메인 그래프를 invoke해서
    분류 후 서브그래프(초안 작성 LLM 등)까지 끝까지 실행할 필요가 없다. 오히려
    그렇게 하면 (1) 분류와 무관한 서브그래프 실패가 이 테스트를 깨뜨리고,
    (2) real 모드에서 불필요한 LLM 호출 비용이 든다. 단위 테스트는 한 가지만
    검증해야 한다. → 분류기만 콕 집어 부른다. (엔드투엔드는 별도 통합 테스트의 몫)

    정확도 = (분류가 expected_intent와 일치한 케이스 수) / (전체 케이스 수).
    정확도 >= pass_threshold 이면 통과.
    """
    golden = _load_golden()
    threshold = golden.get("pass_threshold", 0.9)
    provider = _build_provider(golden)

    correct = 0
    misses: list[str] = []

    for case in golden["cases"]:
        # 분류기만 직접 호출. 그래프/서브그래프는 거치지 않는다.
        result = provider.classify_intent(case["query"])
        predicted = result.intent.value
        expected = case["expected_intent"]

        if predicted == expected:
            correct += 1
        else:
            misses.append(
                f"  #{case['id']} '{case['query']}' "
                f"→ 예측={predicted}, 정답={expected}"
            )

    total = len(golden["cases"])
    accuracy = correct / total

    # 실패 시 어떤 케이스가 틀렸는지 한눈에 보이도록 상세 메시지를 만든다.
    detail = "\n".join(misses) if misses else "  (모두 정답)"
    assert accuracy >= threshold, (
        f"의도 분류 정확도 {accuracy:.0%} < 기준 {threshold:.0%}\n"
        f"틀린 케이스:\n{detail}"
    )
