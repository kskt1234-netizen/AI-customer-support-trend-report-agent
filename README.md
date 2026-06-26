# AI Customer Support & Trend Report Agent Hub

> **한 문장 요약**
> LangGraph로 구현한 **마스터-워커 에이전트 허브** — 고객 문의를 의도 분류해 라우팅하고,
> 트렌드 리포트는 **병렬 수집 · 코드 기반 연산 · 자가 수정 루프**로 생성하며,
> **골든셋 기반 자동 평가(CI)** 로 분류 성능을 검증한다.

B2B SaaS 고객지원을 가정한 에이전트 오케스트레이션 프로젝트입니다. 단순 토이가 아니라
**클린 아키텍처(DIP/SRP)** 와 **하네스 엔지니어링(골든셋·투트랙 테스트)** 철학을 코드로 구현하는 데
초점을 맞췄습니다.

---

## 핵심 아키텍처

```
                 ┌──────────────── MAIN GRAPH (오케스트레이터) ────────────────┐
   유저 질문 ──▶ │  classify_intent ──▶ ◇ 조건부 라우팅 ◇                        │
                 │     (LLM, 구조화 출력)      ╱            ╲                      │
                 └────────────────────────── ╱ ────────────  ╲ ──────────────────┘
                                  SIMPLE_CHAT 노드        TREND_REPORT 서브그래프(워커)
                                  (자유 답변)                      │
   ┌────────────────────────────────────────────────────────────┘
   ▼  SUB GRAPH: trend_report
   parallel_gather ──▶ compute_growth ──▶ draft_report ◀──────────┐ "다시 써"
   (asyncio.gather   (⚠️순수 파이썬       (LLM 초안)               │  retry_count+1
    병렬 더미 수집)    성장률 연산)            │                    │
                                              ▼                    │
                                       critic (2단 게이트) ────────┘ (불합격 & retry<3)
                                       1단 코드: 숫자·출처 존재?
                                       2단 LLM : 맥락·톤 타당?
                                              │
                                    ┌─────────┴─────────┐
                              합격→finalize        retry≥3→bail_out
                              (Pydantic 검증)      (무한루프 방어 탈출)
```

---

## 설계 결정과 "왜"

| 결정 | 이유 |
|---|---|
| **마스터-워커 분리** (`main_graph` / `sub_graphs`) | 메인은 라우팅만, 서브는 업무만. SRP·낮은 결합도. 새 워커(예: 환불 처리)를 레고처럼 추가해도 메인 불변. |
| **LLM 공급사 추상화** (`LLMProvider` ABC + 팩토리) | **DIP**: 노드는 구체 LLM(OpenAI)이 아니라 추상 계약에 의존. OpenAI→Claude 교체 시 노드 코드 0줄 수정. 노드별 독립 주입(예: Orchestrator=OpenAI, Critic=Claude) 가능. |
| **성장률을 LLM이 아닌 코드로 연산** (`compute_growth`) | LLM은 언어 모델이라 산수를 환각한다. 정확도가 생명인 수치는 결정론적 파이썬으로. "LLM은 언어, 코드는 연산." |
| **2단 Critic 게이트** (코드 → LLM) | 코드 게이트만으론 형식만 맞춘 쓰레기("0% 같습니다")를 못 거르고, LLM 게이트만으론 검수자도 환각한다. 싼 코드 게이트로 1차 컷(LLM 호출 절감=비용↓), 통과분만 LLM이 맥락·톤 검수. |
| **`retry_count` 무한루프 방어** | 초안↔검수 핑퐁이 무한 반복되면 API 요금 폭탄·크래시. 최대 3회 후 에러로 우아하게 탈출. |
| **경계마다 Pydantic 검증** (`schemas.py`) | 그래프 경계를 넘는 데이터는 런타임 계약 강제. "타입이 아니라 계약. 위반은 경계에서 막는다." 내부 작업 메모리(State)는 가벼운 TypedDict로 구분. |
| **투트랙 테스트** (mock / real) | 매 Push마다 실제 API를 부르면 느리고 비싸다. 평소 CI는 mock으로 로직만 검증, 정확도 측정은 수동 트리거(real)로. 비용 최적화 + 안정성. |

---

## 프로젝트 구조

```
src/
├── state.py              # 그래프 공용 상태(TypedDict) — 노드 간 데이터 버스
├── schemas.py            # 경계 계약(Pydantic) — IntentClassification, TrendReportOutput
├── logging_config.py     # 공용 로거(print 대신 logging, 레벨 제어)
├── main_graph.py         # 오케스트레이터: 의도 분류 + 조건부 라우팅
├── llm/                  # ── LLM 공급사 추상화(DIP) ──
│   ├── base.py           #    LLMProvider(ABC) : 추상 계약
│   ├── openai_provider.py#    OpenAI 구현체
│   └── factory.py        #    get_provider() : 주입 단일 지점(레지스트리=OCP)
└── sub_graphs/
    └── trend_report.py   # 워커: 병렬수집·연산·초안·2단Critic·루프 방어

tests/
├── test_cases.json       # 골든셋 10개(키워드는 겹치나 의도는 반대인 함정 포함)
├── test_harness.py       # 분류 정확도 채점기(투트랙: mock/real)
├── test_unit.py          # 단위 14개(연산·게이트·라우터·ABC·팩토리·스키마)
└── test_integration.py   # 통합 8개(라우팅·서브그래프 4시나리오)

scripts/
└── run_e2e.py            # 실제 OpenAI로 전체 그래프 완주 확인(수동, 비용 발생)

.github/workflows/run_tests.yml   # CI: push=mock 자동, 수동 트리거=real
```

---

## 빠른 시작

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 환경 설정 (.env.example을 복사해 .env로 만들고 키 입력)
cp .env.example .env
#   → .env의 OPENAI_API_KEY를 실제 키로 교체

# 3) 자동 테스트 (mock 모드 — API 키 불필요, 무료/빠름)
pytest -v

# 4) 분류 정확도 실측 (real 모드 — 실제 OpenAI 호출, 비용 발생)
TEST_MODE=real pytest -v -k accuracy        # PowerShell: $env:TEST_MODE="real"; pytest ...

# 5) 전체 그래프 E2E 완주 확인 (실제 API)
python scripts/run_e2e.py
```

> **로그 레벨**: `LOG_LEVEL=DEBUG`로 실행하면 의도 분류의 근거(reasoning)까지 출력됩니다.
> 분류가 가장 자주 틀리는 지점이라, 디버깅을 위해 근거를 관찰 가능하게 남겨뒀습니다.

---

## 테스트 전략 (테스트 피라미드)

| 층 | 파일 | 검증 | 특징 |
|---|---|---|---|
| 단위 | `test_unit.py` | 부품 하나 격리(연산·게이트·라우터·계약) | 빠름, 실패 시 원인 즉시 식별 |
| 통합 | `test_integration.py` | 조립된 그래프 흐름(분기·루프·방어) | FakeProvider로 결정론적 검증 |
| 골든셋/E2E | `test_harness.py`, `scripts/run_e2e.py` | 분류 정확도 / 실제 완주 | 투트랙(mock/real) |

자동 테스트 23개 전부 mock으로 돌아 **CI에서 무료·결정론적**으로 통과합니다.

---

## 기술 스택

LangGraph · LangChain · OpenAI · Pydantic v2 · pytest / pytest-asyncio · GitHub Actions
