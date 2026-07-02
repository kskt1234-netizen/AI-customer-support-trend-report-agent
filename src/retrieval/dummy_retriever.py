"""
dummy_retriever.py — Retriever 계약의 '더미 키워드 검색' 구현체.

[역할]
무거운 벡터 DB 없이 RAG 워커의 '그래프 로직'을 완성·검증하기 위한 결정론적
검색기. 사내 규정 문서 코퍼스를 코드에 내장하고, 질문에 문서 키워드가
포함되는지로 점수를 매긴다.

[왜 질문에서 키워드를 찾나(역방향 매칭)?]
한국어는 조사("정책이", "환불은")가 붙어 단순 토큰 분리가 잘 안 맞는다.
질문을 쪼개 문서에서 찾는 대신, '문서가 가진 키워드가 질문 문자열에
등장하는가'를 보면 조사에 강건하면서도 완전히 결정론적이다.

[교체 시나리오]
하이브리드 검색(Vector + BM25)을 붙일 때는 HybridRetriever를 새로 만들어
factory 레지스트리에 한 줄 등록하면 끝. 이 파일과 워커는 손대지 않는다.
"""

from __future__ import annotations

import asyncio

from src.retrieval.base import RetrievedDoc, Retriever

# ── 사내 규정 문서 코퍼스(더미) ─────────────────────────────────
# 실제로는 사내 위키/계약 DB에서 오는 데이터. 수치(14일, 10% 등)는 접지 검증
# 테스트의 기준값이므로 바꿀 때는 골든 테스트와 함께 바꿔야 한다.
_POLICY_CORPUS: list[dict] = [
    {
        "doc_id": "POL-REFUND-001",
        "title": "환불 및 위약금 정책",
        "content": (
            "결제일로부터 14일 이내에 요청하시면 전액 환불이 가능합니다. "
            "14일 경과 후에는 남은 계약 기간 요금의 10%를 위약금으로 공제한 뒤 환불됩니다."
        ),
        "keywords": ["환불", "환급", "위약금", "결제 취소"],
    },
    {
        "doc_id": "POL-CONTRACT-002",
        "title": "계약 갱신 및 해지 조건",
        "content": (
            "연간 계약은 만료 30일 전까지 서면으로 통지하지 않으면 동일 조건으로 자동 갱신됩니다. "
            "중도 해지 시에는 잔여 기간 요금의 20%가 청구됩니다."
        ),
        "keywords": ["계약", "갱신", "해지", "만료"],
    },
    {
        "doc_id": "POL-DATA-003",
        "title": "데이터 보관 및 파기 정책",
        "content": (
            "서비스 해지 후 고객 데이터는 90일간 보관되며, "
            "보관 기간이 지나면 복구 불가능한 방식으로 파기됩니다."
        ),
        "keywords": ["데이터", "보관", "파기", "개인정보"],
    },
    {
        "doc_id": "POL-SECURITY-004",
        "title": "계정 보안 정책",
        "content": (
            "관리자 계정의 비밀번호는 90일마다 변경할 것을 권장하며, "
            "접근 권한은 최소 권한 원칙에 따라 부여됩니다."
        ),
        "keywords": ["보안", "비밀번호", "접근 권한", "계정"],
    },
]


class KeywordDummyRetriever(Retriever):
    """키워드 포함 개수로 점수를 매기는 결정론적 더미 검색기."""

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        await asyncio.sleep(0)  # 실제 검색 API/DB 호출 자리(I/O 양보 지점)

        q = (query or "").strip()
        if not q:
            return []

        scored: list[RetrievedDoc] = []
        for doc in _POLICY_CORPUS:
            score = sum(1 for kw in doc["keywords"] if kw in q)
            if score > 0:  # 관련 없는 문서는 아예 반환하지 않는다(빈 리스트 계약)
                scored.append(
                    RetrievedDoc(
                        doc_id=doc["doc_id"],
                        title=doc["title"],
                        content=doc["content"],
                        score=float(score),
                    )
                )

        # 관련도 내림차순. 파이썬 sort는 안정 정렬이라 동점이면 코퍼스 순서 유지(결정론).
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]
