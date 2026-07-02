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
    │                         (경쟁사 실패=부분 저하로 계속 / 시장 실패=error)
    ▼
  compute_growth (순수 파이썬) ── (올해-작년)/작년*100
    │                            → market_growth_rate (데이터 불량이면 error)
    ▼
  ◇ route_after_compute ◇ ── error 있으면 즉시 END (수집/연산 실패 조기 탈출)
    │ 정상
    ▼
  draft_report (LLM) ◀────────────────────────┐ "다시 써"(피드백 동반)
    │                                          │ retry_count += 1
    ▼                                          │
  critic ── 2단 게이트:                         │
    │        1단 코드: 숫자·출처 '존재'?  ───────┘ (불합격 & retry<3 이면 루프)
    │        2단 LLM : 맥락·톤 '타당'? (검수기 자체가 죽으면 fail-closed 반려)
    ▼
  ◇ route_after_critic ◇
    ├─ 합격            → final_output(Pydantic 검증) 담고 END
    └─ retry>=3 (한도) → error 담고 END  (무한 루프 방어 탈출)
"""

from __future__ import annotations

import asyncio
import re

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.llm.base import LLMProvider, content_to_text
from src.logging_config import get_logger
from src.schemas import TrendReportOutput
from src.state import AgentState

_logger = get_logger(__name__)

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
    query = state.get("user_query", "")

    # gather: 두 코루틴을 동시에 스케줄링하고, 둘 다 끝나면 결과를 순서대로 반환.
    # return_exceptions=True: 한쪽이 실패해도 다른 쪽 결과는 살린다.
    # (기본값 False면 첫 예외가 그대로 전파되어 성공한 쪽 데이터까지 버려진다)
    market, competitor = await asyncio.gather(
        _search_market_data(query),
        _search_competitor_data(query),
        return_exceptions=True,
    )

    # [실패 정책 — 소스별로 다르게]
    # 시장 데이터: 성장률 연산의 '필수 입력'. 실패하면 이 워커는 진행 불가 → error.
    if isinstance(market, BaseException):
        _logger.error("시장 데이터 수집 실패 — 워커 진행 불가: %s", market)
        return {"error": f"시장 데이터 수집 실패: {market}"}

    # 경쟁사 데이터: '보조 입력'. 실패해도 시장 데이터만으로 보고서는 쓸 수 있다
    # → 경고만 남기고 부분 저하(degraded)로 계속 진행한다.
    sources: list[str] = []
    if market.get("source"):
        sources.append(market["source"])

    trend_note = ""
    if isinstance(competitor, BaseException):
        _logger.warning("경쟁사 데이터 수집 실패 — 시장 데이터만으로 부분 진행: %s", competitor)
    else:
        trend_note = competitor.get("trend_note", "")
        if competitor.get("source"):
            sources.append(competitor["source"])

    _logger.info("병렬 수집 완료: 출처 %d개 확보", len(sources))
    return {
        "raw_search_data": {
            # 수치의 '유효성'(숫자인가, 0인가)은 다음 노드 compute_growth가 검증한다.
            # 여기는 I/O 실패만 책임진다. (한 노드는 한 가지 실패만 책임 — SRP)
            "last_year_revenue": market.get("last_year_revenue"),
            "this_year_revenue": market.get("this_year_revenue"),
            "trend_note": trend_note,
        },
        # 두 검색이 각자 들고 온 출처를 합친다. 보고서의 '근거'가 된다.
        "sources": sources,
    }


def _is_number(value: object) -> bool:
    """진짜 숫자인지 검사. bool은 int의 서브클래스라 명시적으로 제외한다."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_growth(state: AgentState) -> dict:
    """[노드] 성장률(%)을 '파이썬 코드'로 계산한다. ⚠️ LLM을 절대 쓰지 않는다.

    [왜 코드인가? — 면접 포인트]
    LLM은 언어 모델이라 산수를 환각으로 틀린다. 정확도가 생명인 수치 계산은
    결정론적 파이썬 코드로 분리한다. "LLM은 언어, 코드는 연산."
    이 노드가 def(동기)이고 provider를 인자로 받지도 않는다는 사실 자체가,
    '여기엔 LLM이 개입할 여지가 없다'는 설계 의도를 코드로 드러낸다.
    """
    # 앞 노드(수집)가 이미 error를 남겼으면 연산할 것이 없다. 그대로 통과시켜
    # route_after_compute가 조기 탈출하게 한다. (error를 덮어쓰지 않는 것이 중요)
    if state.get("error"):
        return {}

    data = state.get("raw_search_data") or {}
    last = data.get("last_year_revenue")
    this = data.get("this_year_revenue")

    # [데이터 유효성 방어] 외부 검색 결과는 신뢰할 수 없다. 숫자가 아니거나
    # 누락됐으면 크래시(TypeError) 대신 명시적 error로 전환해 조기 탈출시킨다.
    if not _is_number(last) or not _is_number(this):
        _logger.error(
            "성장률 연산 불가 — 매출 수치가 유효하지 않음: last=%r, this=%r", last, this
        )
        return {"error": f"성장률 연산 불가: 매출 수치가 유효하지 않습니다 (작년={last!r}, 올해={this!r})"}

    # 0으로 나누기 방어: 작년 매출이 0이면 성장률 정의가 불가능.
    if last == 0:
        _logger.warning("작년 매출이 0 — 성장률 정의 불가, 0.0%%로 방어")
        rate = 0.0
    else:
        rate = round((this - last) / last * 100, 1)  # 성장률(%) 수식

    _logger.info("성장률 연산(코드): %s%% (작년=%s → 올해=%s)", rate, last, this)
    return {"market_growth_rate": rate}


# ── 조건부 엣지: 수집/연산 실패 시 조기 탈출하는 순수 함수 ──────────
def route_after_compute(state: AgentState) -> str:
    """수집·연산 단계에서 error가 남았으면 LLM 단계로 가지 않고 즉시 탈출한다.

    [왜 필요한가? — 비용·안전 방어]
    데이터가 없는데 draft_report(LLM)를 호출하면 (1) 돈만 쓰고 (2) 근거 없는
    보고서(환각)를 쓸 위험이 있다. 실패는 '가능한 한 상류에서' 끊는다.
    """
    if state.get("error"):
        return "abort"
    return "draft_report"


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
        # 경쟁사 수집이 실패(부분 저하)했으면 trend_note가 빈 문자열일 수 있다.
        trend_note = state["raw_search_data"].get("trend_note") or "(보조 자료 수집 실패로 정보 없음)"
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

        # [예외 방어] 초안 LLM 호출이 실패하면 '빈 초안'을 반환한다.
        # 빈 초안은 코드 게이트가 반드시 반려하므로, 기존 자가 수정 루프가
        # 그대로 '재시도(retry)' 역할을 겸한다. 일시 장애면 다음 바퀴에 회복되고,
        # 지속 장애면 retry 한도 초과로 bail_out에 수렴한다. (별도 재시도 로직 불필요)
        try:
            response = chat.invoke([("system", system), ("human", human)])
        except Exception:
            _logger.exception("초안 작성 LLM 호출 실패 — 빈 초안 반환(자가 수정 루프가 재시도)")
            return {"draft_report": ""}

        _logger.debug("초안 작성 완료 (retry_count=%d)", state.get("retry_count", 0))
        return {"draft_report": content_to_text(response.content)}

    return draft_report


# ── 1단: 코드 게이트 (싸고 결정론적) ─────────────────────────────
def _growth_mentioned(draft: str, growth: float) -> bool:
    """초안에 성장률 수치가 '독립된 숫자'로 등장하는지 검사한다.

    [왜 단순 부분 문자열(`str(growth) in draft`)이 아닌가? — 엣지 케이스]
      - 오탐: "115.4억" 안에 "15.4"가 부분 문자열로 들어 있어 엉뚱하게 통과한다.
      - 미탐: growth=15.0일 때 LLM이 자연스럽게 "15%"라고 쓰면 탈락시킨다.
    그래서 (1) 정수값이면 "15"/"15.0" 두 표기를 모두 허용하고,
    (2) 정규식으로 숫자 경계(앞뒤에 다른 숫자·소수점 없음)를 강제한다.
    """
    candidates = {str(growth)}
    if float(growth).is_integer():
        candidates.add(str(int(growth)))  # 15.0 → "15"도 허용

    for cand in candidates:
        # (?<![\d.]) : 바로 앞에 숫자/소수점 금지 → "115.4"의 일부 매칭 차단
        # (?!\.?\d)  : 바로 뒤에 (소수점+)숫자 금지 → "15"가 "15.4"에 매칭되는 것 차단
        if re.search(rf"(?<![\d.]){re.escape(cand)}(?!\.?\d)", draft):
            return True
    return False


def _code_gate(state: AgentState) -> tuple[bool, str]:
    """초안에 '숫자 성장률'과 '출처'가 '존재'하는지 코드로 검사.

    [역할 — 면접 포인트]
    이건 '의미'가 아니라 '형식/존재'만 보는 싸구려 1차 필터다. 통과 못 하면
    비싼 LLM 게이트를 부를 필요도 없이 즉시 반려한다. → LLM 호출 횟수 절감(비용↓).
    반환: (통과여부, 실패사유). 통과면 사유는 빈 문자열.
    """
    draft = state.get("draft_report", "") or ""
    growth = state["market_growth_rate"]

    # (a) 코드가 계산한 성장률 숫자가 본문에 '경계가 맞는 독립 숫자'로 등장하는가?
    if not _growth_mentioned(draft, growth):
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
        # [예외 방어 — fail-closed 결정]
        # 검수기 자체가 죽었을 때 '무검수 통과(fail-open)'는 검증 안 된 보고서를
        # 고객에게 내보내는 것이므로 더 위험하다. 반려(fail-closed)로 처리하면
        # 자가 수정 루프가 재시도하고, 지속 장애면 bail_out으로 안전하게 수렴한다.
        try:
            response = chat.invoke([("system", system), ("human", draft)])
        except Exception as e:
            _logger.exception("LLM 게이트 호출 실패 — fail-closed로 반려")
            return False, f"검수기 호출 실패로 반려(fail-closed): {e}"

        verdict = content_to_text(response.content).strip()

        # [엣지 케이스] 빈 응답 — PASS로 오인하지 않도록 명시적으로 반려.
        if not verdict:
            return False, "검수자가 빈 응답을 반환해 판정 불가(fail-closed)."

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
        retry = state.get("retry_count", 0)

        # 1단: 코드 게이트
        ok, reason = _code_gate(state)
        if not ok:
            # 불합격 도장(False) + 피드백 + retry 카운터 증가. (LLM 게이트는 건너뜀)
            _logger.info("Critic 1단(코드 게이트) 반려 (retry %d→%d): %s", retry, retry + 1, reason)
            return {
                "critic_passed": False,
                "critic_feedback": f"[형식] {reason}",
                "retry_count": retry + 1,
            }

        # 2단: LLM 게이트 (1단 통과분만 도달)
        ok, reason = llm_gate(state)
        if not ok:
            _logger.info("Critic 2단(LLM 게이트) 반려 (retry %d→%d): %s", retry, retry + 1, reason)
            return {
                "critic_passed": False,
                "critic_feedback": f"[의미/톤] {reason}",
                "retry_count": retry + 1,
            }

        # 둘 다 통과 → 합격 도장(True)을 명시적으로 찍는다.
        # (라우터는 이 boolean 플래그만 읽어 합격/불합격을 가린다 — 암묵적 신호 X)
        _logger.info("Critic 2단 게이트 모두 통과 (retry_count=%d)", retry)
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
        _logger.info(
            "최종 산출물 확정: 성장률 %s%%, 출처 %d개", output.market_growth_rate, len(output.sources)
        )
        return {"final_output": output.model_dump()}
    except ValidationError as e:
        _logger.error("최종 산출물 계약(Pydantic) 검증 실패: %s", e)
        return {"error": f"최종 산출물 검증 실패: {e}"}


def bail_out(state: AgentState) -> dict:
    """[노드] retry 한도를 초과했을 때 에러를 담고 루프를 탈출한다."""
    _logger.warning(
        "자가 수정 한도(%d회) 초과 — 루프 탈출. 마지막 피드백: %s",
        _MAX_RETRIES,
        state.get("critic_feedback", "(없음)"),
    )
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

    # 수집/연산 실패(error) 시 LLM 단계를 건너뛰고 조기 탈출한다.
    # (데이터 없이 LLM을 부르면 비용 낭비 + 환각 보고서 위험 — 상류에서 차단)
    graph.add_conditional_edges(
        "compute_growth",
        route_after_compute,
        {
            "draft_report": "draft_report",
            "abort": END,
        },
    )

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
