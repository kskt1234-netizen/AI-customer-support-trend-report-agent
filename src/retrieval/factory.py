"""
factory.py — 어떤 Retriever 구현체를 쓸지 결정해 '주입'하는 단일 지점.

llm/factory.py와 같은 패턴(레지스트리 = OCP). 하이브리드 검색(Vector + BM25)을
도입할 때 HybridRetriever를 만들어 _REGISTRY에 한 줄 등록하면, RAG 워커와
메인 그래프는 코드 0줄 수정으로 검색 엔진이 교체된다.
"""

from __future__ import annotations

import os
from typing import Callable

from src.retrieval.base import Retriever
from src.retrieval.dummy_retriever import KeywordDummyRetriever

# 검색기 이름 → 인스턴스 팩토리. 새 검색기는 여기에 한 줄만 추가한다.
_REGISTRY: dict[str, Callable[[], Retriever]] = {
    "dummy": lambda: KeywordDummyRetriever(),
    # "hybrid": lambda: HybridRetriever(),  # ← Vector + BM25 도입 시 주석 해제
}


def get_retriever(name: str | None = None) -> Retriever:
    """이름으로 Retriever 구현체를 생성해 반환한다.

    Args:
        name: 검색기 이름("dummy" 등). None이면 환경변수 RETRIEVER를 읽고,
              그것도 없으면 "dummy"를 기본으로 쓴다.

    Raises:
        ValueError: 등록되지 않은 검색기 이름. (오타/오설정을 조용히 넘기지 않는다)
    """
    resolved = (name or os.getenv("RETRIEVER", "dummy")).lower().strip()

    if resolved not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"알 수 없는 검색기: '{resolved}'. 사용 가능한 값: [{available}]"
        )

    return _REGISTRY[resolved]()
