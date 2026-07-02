"""
run_e2e.py — 실제 OpenAI API로 전체 그래프를 끝까지 돌려보는 수동 E2E 스크립트.

[왜 tests/ 가 아니라 scripts/ 에 두나? — 관심사 분리]
이 스크립트는 진짜 LLM을 호출해 '비용'이 든다. pytest가 자동 수집하면 매 CI마다
돈이 새고 외부 장애에 흔들린다. 그래서 자동 테스트(tests/)와 분리해, '사람이 손으로'
실행하는 scripts/ 에 둔다. 자동화된 로직 검증은 mock 통합 테스트의 몫,
이 스크립트는 '진짜로 탈선 없이 도는가'를 눈으로 확인하는 용도다.

[검증하는 것]
1) SIMPLE_CHAT 경로: 일반 질문 → 의도 분류 → simple_chat → 답변
2) TREND_REPORT 경로: 분석 질문 → 의도 분류 → 서브그래프(병렬수집→연산→초안→
   2단 Critic 루프→마무리) → 최종 출력(TrendReportOutput)

실행 전 .env에 유효한 OPENAI_API_KEY가 있어야 한다.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# 이 스크립트는 프로젝트 루트의 scripts/ 안에 있다. src 패키지를 import하려면
# 루트를 모듈 경로에 추가해야 한다. (python scripts/run_e2e.py 로 실행 가능하게)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# .env를 '코드 실행 전에' 로드한다. get_provider()는 os.getenv만 읽으므로,
# 여기서 load_dotenv를 먼저 호출해야 OPENAI_API_KEY가 환경에 올라온다.
load_dotenv(_PROJECT_ROOT / ".env")

from src.logging_config import get_logger  # noqa: E402
from src.main_graph import get_compiled_app  # noqa: E402

_logger = get_logger("e2e")


def _check_api_key() -> None:
    """키가 없거나 플레이스홀더면 명확히 안내하고 종료한다."""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-여기에") or "붙여넣기" in key:
        print("❌ .env의 OPENAI_API_KEY가 비어있거나 예시값입니다. 실제 키로 교체하세요.")
        sys.exit(1)


async def _run_one(app, label: str, query: str) -> None:
    """질문 하나를 전체 그래프에 통과시키고 결과를 보기 좋게 출력한다."""
    print("\n" + "=" * 70)
    print(f"▶ [{label}] 질문: {query}")
    print("=" * 70)

    # retry_count를 0으로 초기화해 넘긴다(서브그래프 루프 카운터의 시작점).
    result = await app.ainvoke({"user_query": query, "retry_count": 0})

    print(f"· 분류된 의도(intent): {result.get('intent')}")

    if result.get("intent") == "SIMPLE_CHAT":
        print(f"· 답변(chat_response):\n{result.get('chat_response')}")
    elif result.get("intent") == "POLICY_INQUIRY":
        # 규정 RAG 경로의 산출물.
        print(f"· 검색된 문서 수: {len(result.get('retrieved_docs') or [])}")
        print(f"· 접지 통과(grounding_passed): {result.get('grounding_passed')}")
        print(f"· 재시도 횟수(retry_count): {result.get('retry_count')}")
        if result.get("error"):
            print(f"· ⚠️ 에러(검색 실패/접지 한도 초과 등): {result['error']}")
        if result.get("final_output"):
            print("· 최종 출력(PolicyAnswerOutput, Pydantic 검증 통과):")
            print(json.dumps(result["final_output"], ensure_ascii=False, indent=2))
    else:
        # TREND_REPORT 경로의 산출물.
        print(f"· 계산된 성장률(코드 연산): {result.get('market_growth_rate')}%")
        print(f"· 검수 통과(critic_passed): {result.get('critic_passed')}")
        print(f"· 재시도 횟수(retry_count): {result.get('retry_count')}")
        if result.get("error"):
            print(f"· ⚠️ 에러(루프 방어 탈출 등): {result['error']}")
        if result.get("final_output"):
            print("· 최종 출력(TrendReportOutput, Pydantic 검증 통과):")
            print(json.dumps(result["final_output"], ensure_ascii=False, indent=2))


async def main() -> None:
    _check_api_key()

    # 기본 공급사(OpenAI)로 컴파일된 메인 앱. 우리 코드 구조와 그대로 싱크.
    app = get_compiled_app()

    # 세 경로를 모두 태워 그래프가 분기마다 올바로 도는지 눈으로 확인.
    await _run_one(app, "SIMPLE_CHAT 경로", "비밀번호는 어떻게 변경하나요?")
    await _run_one(app, "TREND_REPORT 경로", "올해 SaaS 시장 매출 트렌드를 분석해서 리포트로 만들어줘")
    await _run_one(app, "POLICY_INQUIRY 경로", "환불 위약금이 몇 퍼센트인지 규정 기준으로 알려주세요")

    print("\n✅ E2E 완주. 위 출력에서 탈선/누락이 없는지 직접 검수하세요.")


if __name__ == "__main__":
    asyncio.run(main())
