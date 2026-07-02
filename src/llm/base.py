"""
base.py — LLM 공급사(provider)의 추상 인터페이스(계약).

[왜 이 파일이 존재하는가? — DIP(의존성 역전 원칙)]
우리가 합의한 핵심 결정: "각 노드는 구체적인 LLM 공급사(OpenAI/Anthropic)에
직접 의존하면 안 된다."

만약 노드가 `ChatOpenAI(...)`를 직접 생성하면, 그 노드는 OpenAI에 '못 박힌다'.
나중에 Critic 노드만 Claude로 바꾸려 해도 노드 코드를 뜯어고쳐야 한다.

그래서 방향을 뒤집는다(역전한다):
  - 노드는 '구체 클래스(ChatOpenAI)'가 아니라 '추상 계약(LLMProvider)'에만 의존한다.
  - 구체 공급사(OpenAIProvider)는 그 계약을 '구현'한다.
  - 누가 어떤 공급사를 쓸지는 factory.py가 '주입'한다.

결과:
  - 노드 입장에선 "classify_intent / get_chat_model을 제공하는 무언가"만 알면 된다.
  - OpenAI를 Claude로 갈아끼워도 노드 코드는 한 줄도 안 바뀐다. (결합도 ↓)
  - 테스트에선 이 계약을 구현한 'FakeProvider'를 주입해 API 호출 없이 로직만 검증한다.
    → 투트랙 테스트(Mock/실제)의 토대가 바로 이 추상 계약이다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.schemas import IntentClassification


def content_to_text(content: Any) -> str:
    """LangChain 메시지의 content를 안전하게 '순수 텍스트'로 변환한다.

    [왜 필요한가? — LLM 경계의 형태 방어]
    response.content는 보통 str이지만, 모델/버전에 따라 콘텐츠 블록 리스트
    ([{"type": "text", "text": "..."}, ...])로 올 수도 있고 None일 수도 있다.
    각 노드가 이 차이를 개별적으로 처리하면 방어가 누락되기 쉬우므로,
    LLM 경계 모듈인 여기에서 한 번에 흡수한다. (노드는 항상 str만 받는다)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


class LLMProvider(ABC):
    """모든 LLM 공급사가 지켜야 하는 추상 계약.

    [왜 평범한 클래스가 아니라 ABC인가?]
    ABC(Abstract Base Class)로 두고 메서드에 @abstractmethod를 붙이면,
    이 계약을 '구현하지 않은' 자식 클래스는 인스턴스화 자체가 막힌다(런타임 에러).
    즉 "OpenAIProvider를 만들면서 classify_intent를 깜빡 빼먹는" 실수를
    파이썬이 강제로 잡아준다. → 계약 위반을 코드 레벨에서 차단.
    """

    @abstractmethod
    def classify_intent(self, user_query: str) -> IntentClassification:
        """유저 질문을 받아 의도를 분류해 IntentClassification으로 반환한다.

        구현체는 내부적으로 `with_structured_output(IntentClassification)`을 써서
        LLM이 자유 텍스트가 아니라 '스키마에 맞는 객체'를 반환하도록 강제해야 한다.

        Args:
            user_query: 유저의 원본 질문.

        Returns:
            IntentClassification: intent(Enum) + reasoning(근거 문장)을 담은 객체.
        """
        ...

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """초안 작성/검수 등 '자유 텍스트 생성'에 쓸 LangChain 챗 모델을 반환한다.

        [왜 분류와 챗 모델을 메서드로 분리했나?]
        - classify_intent : 출력이 '구조화(스키마 강제)'되어야 하는 작업.
        - get_chat_model  : 출력이 '자유 텍스트'인 작업(초안 작성, 검수 피드백 등).
        두 쓰임새가 다르므로 인터페이스에서도 분리해 둔다. 노드는 필요한 쪽만
        가져다 쓴다. (인터페이스 분리 원칙 ISP의 가벼운 적용)

        Returns:
            BaseChatModel: LangChain의 공통 챗 모델 인터페이스.
                           OpenAI든 Anthropic이든 모두 이 타입을 만족하므로,
                           노드는 구체 공급사를 몰라도 .invoke()를 호출할 수 있다.
        """
        ...
