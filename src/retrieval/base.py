"""
base.py — 문서 검색기(Retriever)의 추상 계약.

[왜 이 파일이 존재하는가? — llm/base.py와 같은 이유(DIP)]
RAG 워커가 특정 검색 기술(더미 키워드, ChromaDB, BM25, 하이브리드...)에
직접 의존하면, 검색 엔진을 바꿀 때마다 워커 노드를 뜯어고쳐야 한다.
그래서 방향을 역전한다:
  - 워커는 'Retriever 계약'에만 의존한다. (retrieve만 부를 줄 알면 됨)
  - 구체 검색기(KeywordDummyRetriever, 향후 HybridRetriever 등)가 계약을 구현한다.
  - 어떤 검색기를 쓸지는 factory.py가 주입한다.

결과: 더미 검색 → 하이브리드 검색(Vector + BM25) 교체 시 워커 코드 0줄 수정.
테스트에선 '고장난 검색기'를 주입해 장애 방어를 결정론적으로 검증한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    """검색 결과 문서 1건의 계약.

    검색기가 무엇이든(더미/벡터/BM25) 워커에게는 항상 이 형태로 건네야 한다.
    → 검색기 교체가 워커에 새어 나가지 않게 하는 '경계 계약'.
    """

    doc_id: str = Field(..., description="문서 고유 ID. 답변 인용(citation)의 기준 키.")
    title: str = Field(..., description="문서 제목.")
    content: str = Field(..., description="문서 본문(답변 근거가 되는 원문).")
    score: float = Field(0.0, description="검색 관련도 점수. 클수록 관련 높음.")


class Retriever(ABC):
    """모든 문서 검색기가 지켜야 하는 추상 계약.

    [왜 async인가?]
    실제 검색기는 벡터 DB/검색 API를 네트워크로 호출한다(I/O 바운드).
    시그니처를 처음부터 async로 못 박아 두면, 더미를 진짜로 교체해도
    워커의 호출 코드(await retriever.retrieve(...))가 안 바뀐다.
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """질문과 관련된 문서를 관련도 순으로 최대 top_k개 반환한다.

        Args:
            query: 유저의 원본 질문.
            top_k: 반환할 최대 문서 수.

        Returns:
            관련 문서 목록(관련도 내림차순). 관련 문서가 없으면 빈 리스트.
            '없으면 빈 리스트'가 계약이다 — 예외로 표현하지 않는다.
            (근거 없음은 정상 상황이고, 예외는 검색기 장애에만 쓴다)
        """
        ...
