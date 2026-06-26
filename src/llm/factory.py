"""
factory.py — 어떤 LLMProvider 구현체를 쓸지 결정해 '주입'하는 단일 지점.

[왜 팩토리가 필요한가? — DIP의 마지막 조각]
base.py가 '계약'을, openai_provider.py가 '구현'을 맡았다. 그런데 노드/그래프가
직접 `OpenAIProvider()`를 생성하면, 결국 노드가 구체 클래스를 다시 import하게 되어
DIP가 깨진다.

그래서 "어떤 구현체를 고를지"라는 결정을 이 파일 한 곳에 가둔다.
  - 노드/그래프는 get_provider()만 호출하고, 돌려받은 LLMProvider 계약만 쓴다.
  - 공급사 교체는 여기 한 줄(또는 환경변수)만 바꾸면 끝난다.
  - "Orchestrator=OpenAI, Critic=Claude"처럼 노드별로 다른 공급사를 주입하고
    싶을 때도, 각 노드가 get_provider("openai") / get_provider("anthropic")를
    호출하면 된다. (노드별 독립 주입 — 우리가 합의한 목표)

[확장 방법]
anthropic_provider.py를 추가하고 _REGISTRY에 한 줄 등록하면 끝.
factory의 분기를 if-elif로 늘리지 않고 '레지스트리(dict)'로 둔 이유:
공급사가 늘어도 함수 본문을 수정할 필요가 없다(개방-폐쇄 원칙 OCP).
"""

from __future__ import annotations

import os
from typing import Callable

from src.llm.base import LLMProvider
from src.llm.openai_provider import OpenAIProvider

# 공급사 이름 → 그 공급사 인스턴스를 만드는 팩토리 함수.
# 새 공급사는 여기에 한 줄만 추가하면 된다. (OCP: 확장엔 열리고 수정엔 닫힘)
_REGISTRY: dict[str, Callable[[], LLMProvider]] = {
    "openai": lambda: OpenAIProvider(),
    # "anthropic": lambda: AnthropicProvider(),  # ← 2차 구현 시 주석 해제
}


def get_provider(name: str | None = None) -> LLMProvider:
    """이름으로 LLMProvider 구현체를 생성해 반환한다.

    Args:
        name: 공급사 이름("openai" 등). None이면 환경변수 LLM_PROVIDER를
              읽고, 그것도 없으면 "openai"를 기본으로 쓴다.
              → 코드 수정 없이 환경변수만으로 공급사를 바꿀 수 있게 한 것.

    Returns:
        LLMProvider: 호출자는 구체 타입을 모른 채 계약만 사용한다.

    Raises:
        ValueError: 등록되지 않은 공급사 이름이 들어온 경우. (조용히 실패하지
                    않고 명시적으로 터뜨려, 오타/오설정을 즉시 드러낸다.)
    """
    resolved = name or os.getenv("LLM_PROVIDER", "openai")
    resolved = resolved.lower().strip()

    if resolved not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"알 수 없는 LLM 공급사: '{resolved}'. 사용 가능한 값: [{available}]"
        )

    return _REGISTRY[resolved]()
