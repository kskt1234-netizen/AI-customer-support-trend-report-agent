"""
trend_report.py — TREND_REPORT 워커(서브 그래프).

이 서브 그래프는 이 프로젝트의 하이라이트로, 4가지 패턴을 한 흐름에 담는다:

  1) 병렬 수집      : asyncio.gather로 더미 검색 2개를 '동시' 실행 (I/O 겹치기)
  2) 연산 분리      : 성장률(%)은 LLM이 아니라 '파이썬 코드'가 결정론적으로 계산
  3) 초안 작성      : LLM이 자유 텍스트 보고서 초안을 작성
  4) 2단 자가수정   : [코드 게이트 → LLM 게이트] 2계층 Critic + retry_count 방어

[전체 흐름]

  START
    │
    ▼
  parallel_gather (async) ── asyncio.gather(시장검색, 경쟁사검색)
    │                         → raw_search_data, sources
    ▼
  compute_growth (순수 파이썬) ── (올해-작년)/작년*100
    │                            → market_growth_rate
    ▼
  draft_report (LLM) ◀────────────────────────┐ "다시 써"(피드백 동반)
    │                                          │ retry_count += 1
    ▼                                          │
  critic ── 2단 게이트:                         │
    │        1단 코드: 숫자·출처 '존재'?  ───────┘ (불합격 & retry<3 이면 루프)
    │        2단 LLM : 맥락·톤 '타당'?
    ▼
  ◇ route_after_critic ◇
    ├─ 합격            → final_output(Pydantic 검증) 담고 END
    └─ retry>=3 (한도) → error 담고 END  (무한 루프 방어 탈출)
"""

from __future__ import annotations

import asyncio

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.llm.base import LLMProvider
from src.schemas import TrendReportOutput
from src.state import AgentState

# 무한 루프 방어: 초안↔검수 핑퐁을 최대 몇 번까지 허용할지.
# 이 횟수를 넘으면 error를 채우고 그래프를 탈출한다. (API 요금 폭탄/크래시 방지)
_MAX_RETRIES = 3


# ═════════════════════════════════════════════════════════════
# ④-1) 병렬 수집 & 연산 분리
# ═════════════════════════════════════════════════════════════

# ── 외부 검색 API를 흉내 낸 더미(Mock) 함수 2개 ──────────────────
# [왜 async 함수인가?]
# 실제로는 외부 검색/DB API를 await로 호출할 자리다. 지금은 더미지만,
# 시그니처를 async로 둬서 '진짜 API로 교체해도 구조가 안 바뀌게' 한다.
# asyncio.sleep(0)은 "여기에 네트워크 I/O가 있다"는 표식이자, 이벤트 루프에
# 제어권을 넘겨 다른 코루틴이 진행될 틈을 주는 역할.

async def _search_market_data(query: str) -> dict:
    """[더미] 시장 매출 데이터 검색. 고정값 반환(결정론적 테스트를 위해)."""
    await asyncio.sleep(0)  # 실제 API 호출 자리(I/O 양보 지점)
    return {
        "last_year_revenue": 100.0,   # 작년 시장 매출(억). 고정값.
        "this_year_revenue": 115.4,   # 올해 시장 매출(억). 고정값. → 성장률 15.4%
        "source": "시장조사기관 A 2024 연간 리포트",
    }


async def _search_competitor_data(query: str) -> dict:
    """[더미] 경쟁사/업계 동향 검색. 고정 출처 반환."""
    await asyncio.sleep(0)  # 실제 API 호출 자리(I/O 양보 지점)
    return {
        "trend_note": "업계 전반의 SaaS 도입률 증가가 매출 성장을 견인.",
        "source": "업계 뉴스레터 B 2024 12월호",
    }


async def parallel_gather(state: AgentState) -> dict:
    """[노드] 두 더미 검색을 asyncio.gather로 '병렬' 실행해 데이터를 모은다.

    [왜 gather(병렬)인가? — 면접 포인트]
    검색을 순차로 하면 총 소요 = (검색A 시간 + 검색B 시간).
    gather로 동시에 던지면 총 소요 = max(검색A, 검색B).
    검색·API 호출은 'CPU가 노는 I/O 대기'라서, 대기 시간을 겹치면 그만큼
    벽시계 시간이 준다. (CPU 연산이 아니라 I/O 바운드일 때 효과적)
    더미라도 '이런 구조로 짠다'를 보여주는 게 핵심.
    """
    query = state["user_query"]

    # gather: 두 코루틴을 동시에 스케줄링하고, 둘 다 끝나면 결과를 순서대로 반환.
    market, competitor = await asyncio.gather(
        _search_market_data(query),
        _search_competitor_data(query),
    )

    return {
        "raw_search_data": {
            "last_year_revenue": market["last_year_revenue"],
            "this_year_revenue": market["this_year_revenue"],
            "trend_note": competitor["trend_note"],
        },
        # 두 검색이 각자 들고 온 출처를 합친다. 보고서의 '근거'가 된다.
        "sources": [market["source"], competitor["source"]],
    }


def compute_growth(state: AgentState) -> dict:
    """[노드] 성장률(%)을 '파이썬 코드'로 계산한다. ⚠️ LLM을 절대 쓰지 않는다.

    [왜 코드인가? — 면접 포인트]
    LLM은 언어 모델이라 산수를 환각으로 틀린다. 정확도가 생명인 수치 계산은
    결정론적 파이썬 코드로 분리한다. "LLM은 언어, 코드는 연산."
    이 노드가 def(동기)이고 provider를 인자로 받지도 않는다는 사실 자체가,
    '여기엔 LLM이 개입할 여지가 없다'는 설계 의도를 코드로 드러낸다.
    """
    data = state["raw_search_data"]
    last = data["last_year_revenue"]
    this = data["this_year_revenue"]

    # 0으로 나누기 방어: 작년 매출이 0이면 성장률 정의가 불가능.
    if last == 0:
        rate = 0.0
    else:
        rate = round((this - last) / last * 100, 1)  # 성장률(%) 수식

    return {"market_growth_rate": rate}


# ═════════════════════════════════════════════════════════════
# ④-2) 초안 작성 & 2단 Critic & 자가 수정 루프
# ═════════════════════════════════════════════════════════════

def make_draft_report_node(provider: LLMProvider):
    """[노드 팩토리] LLM으로 보고서 초안을 작성하는 노드를 만든다.

    Critic이 반려하면(critic_feedback이 있으면) 그 피드백을 프롬프트에 넣어
    '다시' 작성한다. → 자가 수정 루프의 '수정' 쪽.
    """

    def draft_report(state: AgentState) -> dict:
        chat = provider.get_chat_model()

        growth = state["market_growth_rate"]
        sources = state["sources"]
        trend_note = state["raw_search_data"]["trend_note"]
        feedback = state.get("critic_feedback")  # 첫 작성 땐 없음(None).

        # 재작성이면 직전 피드백을 프롬프트에 명시해 같은 실수를 피하게 한다.
        feedback_block = ""
        if feedback:
            feedback_block = (
                f"\n\n[직전 검수자 피드백 — 반드시 반영하세요]\n{feedback}"
            )

        system = (
            "당신은 B2B SaaS 트렌드 분석 보고서를 쓰는 전문 애널리스트입니다. "
            "전문적이고 정중한 '~입니다' 체로, 문단을 나눠 가독성 있게 작성하세요. "
            "반드시 (1) 주어진 성장률 수치를 '시장 매출 성장률'이라는 올바른 맥락으로 "
            "서술하고, (2) 제공된 출처를 본문 끝에 명시하세요. 숫자를 임의로 바꾸지 마세요."
        )
        human = (
            f"다음 데이터로 트렌드 분석 보고서를 작성하세요.\n"
            f"- 시장 매출 성장률: {growth}%\n"
            f"- 업계 동향: {trend_note}\n"
            f"- 출처: {', '.join(sources)}"
            f"{feedback_block}"
        )

        response = chat.invoke([("system", system), ("human", human)])
        return {"draft_report": response.content}

    return draft_report


# ── 1단: 코드 게이트 (싸고 결정론적) ─────────────────────────────
def _code_gate(state: AgentState) -> tuple[bool, str]:
    """초안에 '숫자 성장률'과 '출처'가 '존재'하는지 코드로 검사.

    [역할 — 면접 포인트]
    이건 '의미'가 아니라 '형식/존재'만 보는 싸구려 1차 필터다. 통과 못 하면
    비싼 LLM 게이트를 부를 필요도 없이 즉시 반려한다. → LLM 호출 횟수 절감(비용↓).
    반환: (통과여부, 실패사유). 통과면 사유는 빈 문자열.
    """
    draft = state.get("draft_report", "") or ""
    growth = state["market_growth_rate"]

    # (a) 코드가 계산한 성장률 숫자가 본문에 실제로 등장하는가?
    #     문자열로 변환해 포함 여부를 본다. (예: "15.4"가 본문에 있는지)
    if str(growth) not in draft:
        return False, f"보고서에 계산된 성장률 수치({growth})가 보이지 않습니다."

    # (b) 출처가 본문에 하나라도 인용되었는가?
    sources = state.get("sources", [])
    if not any(src in draft for src in sources):
        return False, "보고서에 제공된 출처가 인용되지 않았습니다."

    return True, ""


# ── 2단: LLM 게이트 (비싸지만 의미 검수) ──────────────────────────
def make_llm_gate(provider: LLMProvider):
    """초안의 '맥락 정확성·비즈니스 톤'을 LLM이 판정하는 검수기를 만든다.

    [역할 — 면접 포인트]
    코드 게이트는 '15.4%가 존재하는가'는 보지만 '15.4%가 시장 성장률이라는
    올바른 맥락에 박혔는가', '톤이 전문적인가'는 못 본다. 그 의미 검수를 여기서
    한다. 1단을 통과한, 즉 '형식은 갖춘' 초안만 여기로 오므로 LLM 낭비가 적다.
    반환: (통과여부, 실패사유).
    """

    def llm_gate(state: AgentState) -> tuple[bool, str]:
        chat = provider.get_chat_model()
        growth = state["market_growth_rate"]
        draft = state["draft_report"]

        # 판정을 'PASS' 또는 'FAIL: 이유' 한 줄로만 받도록 제약해 파싱을 단순화.
        system = (
            "당신은 깐깐한 보고서 품질 검수자입니다. 아래 보고서가 두 기준을 "
            "모두 만족하는지 판단하세요.\n"
            f"1) 성장률 {growth}%가 '시장/매출 성장'이라는 올바른 맥락으로 쓰였는가 "
            "(퇴사율·실패 등 엉뚱한 맥락에 결합되지 않았는가).\n"
            "2) 전문적이고 정중한 비즈니스 톤(~입니다 체)인가.\n"
            "둘 다 만족하면 정확히 'PASS'만 출력하세요. 하나라도 어기면 "
            "'FAIL: <한 문장 사유>' 형식으로 출력하세요."
        )
        response = chat.invoke([("system", system), ("human", draft)])
        verdict = (response.content or "").strip()

        if verdict.upper().startswith("PASS"):
            return True, ""
        # 'FAIL: ...' 에서 사유만 떼어낸다. 형식이 어긋나도 통째로 사유로 쓴다.
        reason = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
        return False, reason or "LLM 검수자가 품질 미달로 반려했습니다."

    return llm_gate


def make_critic_node(provider: LLMProvider):
    """[노드 팩토리] 2단 게이트를 순서대로 적용하는 Critic 노드를 만든다.

    [왜 코드 게이트를 먼저? — 비용 최적화]
    1단(코드)에서 걸리면 2단(LLM)을 아예 호출하지 않는다. 싼 검사로 먼저
    거르고, 통과한 것만 비싼 검사에 보낸다. → 'LLMOps 비용 최적화'의 실체.
    """
    llm_gate = make_llm_gate(provider)

    def critic(state: AgentState) -> dict:
        # 1단: 코드 게이트
        ok, reason = _code_gate(state)
        if not ok:
            # 불합격 도장(False) + 피드백 + retry 카운터 증가. (LLM 게이트는 건너뜀)
            return {
                "critic_passed": False,
                "critic_feedback": f"[형식] {reason}",
                "retry_count": state.get("retry_count", 0) + 1,
            }

        # 2단: LLM 게이트 (1단 통과분만 도달)
        ok, reason = llm_gate(state)
        if not ok:
            return {
                "critic_passed": False,
                "critic_feedback": f"[의미/톤] {reason}",
                "retry_count": state.get("retry_count", 0) + 1,
            }

        # 둘 다 통과 → 합격 도장(True)을 명시적으로 찍는다.
        # (라우터는 이 boolean 플래그만 읽어 합격/불합격을 가린다 — 암묵적 신호 X)
        return {"critic_passed": True, "critic_feedback": ""}

    return critic


def finalize(state: AgentState) -> dict:
    """[노드] 합격한 초안을 Pydantic 계약(TrendReportOutput)으로 검증해 확정.

    [왜 여기서 또 Pydantic? — 경계 검증]
    Critic을 통과했어도, 최종 산출물은 그래프 '경계'를 넘어 바깥(메인/호출자)으로
    나간다. 경계에서 계약을 한 번 더 강제해, 형식이 깨진 결과가 새어 나가지 못하게
    한다. 검증 실패 시엔 error로 돌린다(조용히 잘못된 데이터를 내보내지 않음).
    """
    try:
        output = TrendReportOutput(
            market_growth_rate=state["market_growth_rate"],
            summary=state["draft_report"],
            sources=state["sources"],
        )
        return {"final_output": output.model_dump()}
    except ValidationError as e:
        return {"error": f"최종 산출물 검증 실패: {e}"}


def bail_out(state: AgentState) -> dict:
    """[노드] retry 한도를 초과했을 때 에러를 담고 루프를 탈출한다."""
    return {
        "error": (
            f"자가 수정 {_MAX_RETRIES}회를 초과해 품질 기준을 통과하지 못했습니다. "
            f"마지막 피드백: {state.get('critic_feedback', '(없음)')}"
        )
    }


# ── 조건부 엣지: Critic 이후 어디로 갈지 결정하는 순수 함수 ──────────
def route_after_critic(state: AgentState) -> str:
    """Critic 결과를 보고 다음 목적지 키를 반환한다.

    세 갈래:
      - 합격(critic_passed == True)                  → "finalize"
      - 불합격이지만 retry 한도 초과(>= _MAX_RETRIES) → "bail_out"  (무한루프 방어)
      - 불합격 & 아직 여유 있음                       → "draft_report" (재작성 루프)
    """
    if state.get("critic_passed", False):
        return "finalize"  # 합격 도장이 찍혔으면 마무리로

    # 불합격 상태. 한도를 넘었는지 본다.
    if state.get("retry_count", 0) >= _MAX_RETRIES:
        return "bail_out"  # 한도 초과 → 탈출

    return "draft_report"  # 아직 여유 있음 → 재작성


# ═════════════════════════════════════════════════════════════
# 그래프 빌더
# ═════════════════════════════════════════════════════════════
def build_trend_report_graph(provider: LLMProvider):
    """트렌드 리포트 서브 그래프를 빌드해 컴파일된 그래프를 반환한다.

    Args:
        provider: 주입할 LLM 공급사. 초안 작성과 LLM 게이트가 이걸 사용한다.
                  (수집/연산 노드는 LLM을 쓰지 않으므로 provider가 필요 없다)
    """
    graph = StateGraph(AgentState)

    # ── 노드 등록 ──
    graph.add_node("parallel_gather", parallel_gather)       # async 노드
    graph.add_node("compute_growth", compute_growth)         # 순수 파이썬
    graph.add_node("draft_report", make_draft_report_node(provider))
    graph.add_node("critic", make_critic_node(provider))
    graph.add_node("finalize", finalize)
    graph.add_node("bail_out", bail_out)

    # ── 엣지(흐름) ──
    graph.add_edge(START, "parallel_gather")
    graph.add_edge("parallel_gather", "compute_growth")
    graph.add_edge("compute_growth", "draft_report")
    graph.add_edge("draft_report", "critic")

    # critic 이후는 상태에 따라 분기(조건부 엣지).
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "finalize": "finalize",
            "bail_out": "bail_out",
            "draft_report": "draft_report",  # ← 여기서 루프(재작성)가 형성된다
        },
    )

    # 종료 경로 두 갈래.
    graph.add_edge("finalize", END)
    graph.add_edge("bail_out", END)

    return graph.compile()
