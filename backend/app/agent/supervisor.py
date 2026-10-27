"""
Supervisor
──────────────────────────────────────────────────────────────
작업자(worker) 노드 중 다음에 무엇을 실행할지 결정합니다.
고정된 순서 대신 질문 의도와 현재 근거 상태를 보고 판단합니다.

작업자 목록
  researcher          질문에 필요한 근거 자료 수집 (문서 + 웹), 근거 충족 여부 판단
  competitor_analyst  분석 대상 기업별 현황 분석 (경쟁사 비교 질문에서만)
  writer              보고서 작성

판단 순서
  1. 질문 의도 분류 (최초 1회, LLM)   fact | compare | latest
  2. 근거가 없으면 researcher
  3. 근거가 부족하면 재검색 (재검색 상한, 웹 검색 가능 여부 확인) / 더 할 수 없으면 한계를 남기고 진행
  4. 경쟁사 비교 질문이면 competitor_analyst
  5. writer
  비용 상한에 도달하면 재검색·경쟁사 분석을 생략하고 writer 로 갑니다.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import INTENT_SYSTEM, parse_json
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.config import settings
from app.events import Progress, Route, emit
from app.llm import BudgetExceeded, ainvoke_tracked, budget_exceeded
from app.retrieval import web_search_enabled

WORKERS = ["researcher", "competitor_analyst", "writer"]
INTENTS = ["fact", "compare", "latest"]
INTENT_LABELS = {"fact": "단일 사실 확인", "compare": "경쟁사 비교", "latest": "최신 정보"}


async def classify_intent(state: AgentState) -> str:
    """질문 의도를 LLM으로 분류합니다. 해석할 수 없으면 전체 작업을 수행하는 compare 로 봅니다."""
    messages = [SystemMessage(content=INTENT_SYSTEM), HumanMessage(content=f"질문: {state['query']}")]
    try:
        result = parse_json(await ainvoke_tracked(messages, node="supervisor")) or {}
    except BudgetExceeded as e:
        # 비용 상한: 경쟁사 분석이 없는 가장 짧은 경로로 진행합니다.
        emit(Progress(node="supervisor", message=f"[비용 상한] 질문 의도 분류 생략, 단일 사실 확인 경로로 진행합니다: {e}"))
        return "fact"
    intent = result.get("intent")
    if intent not in INTENTS:
        emit(Progress(node="supervisor", message="질문 의도를 해석하지 못해 경쟁사 비교 경로로 진행합니다."))
        return "compare"
    emit(Progress(node="supervisor", message=f"질문 의도: {INTENT_LABELS[intent]} ({result.get('reason', '')})"))
    return intent


def decide_next(state: AgentState, intent: str) -> tuple[str, str, dict]:
    """(다음 노드, 분기 이유, 추가로 바꿀 상태)를 돌려줍니다."""
    attempts = state.get("search_attempts", 0)
    if attempts == 0:
        return "researcher", "근거 자료가 아직 없음", {}

    # 비용 상한 도달: 재검색·경쟁사 분석을 생략하고 지금까지의 근거로 작성합니다.
    if budget_exceeded():
        return "writer", "비용 상한 도달로 남은 작업(재검색·경쟁사 분석)을 생략하고 작성", {}

    if not state.get("evidence_sufficient", False):
        note = state.get("evidence_note", "")
        if intent == "latest" and not web_search_enabled():
            limit = "웹 검색을 사용할 수 없어 최신 근거를 확보하지 못함"
            if limit not in note:
                emit(Progress(node="supervisor", message=f"재검색 생략: {limit}"))
                return _after_research(state, intent, {"evidence_note": f"{note} / 근거 한계: {limit}".strip(" /")})
        elif attempts < settings.max_search_attempts:
            return "researcher", f"근거 부족으로 재검색 ({attempts + 1}/{settings.max_search_attempts}회): {note}", {}
        else:
            limit = f"재검색 상한({settings.max_search_attempts}회) 후에도 근거 부족"
            if limit not in note:
                emit(Progress(node="supervisor", message=f"재검색 종료: {limit}"))
                return _after_research(state, intent, {"evidence_note": f"{note} / 근거 한계: {limit}".strip(" /")})

    return _after_research(state, intent, {})


def _after_research(state: AgentState, intent: str, update: dict) -> tuple[str, str, dict]:
    if intent == "compare" and "competitor_findings" not in state:
        return "competitor_analyst", "경쟁사 비교 질문이라 기업별 분석 필요", update
    if intent == "compare":
        return "writer", "근거와 기업별 분석 결과 준비 완료", update
    return "writer", f"{INTENT_LABELS[intent]} 질문이라 경쟁사 분석 없이 작성", update


@traced("supervisor")
async def supervisor(state: AgentState) -> dict:
    intent = state.get("intent") or await classify_intent(state)
    next_worker, reason, update = decide_next(state, intent)

    emit(Route(source="supervisor", target=next_worker, reason=reason))
    return {"intent": intent, "next": next_worker, **update}


def route_from_supervisor(state: AgentState) -> str:
    """조건부 엣지: Supervisor가 정한 다음 노드 이름을 돌려줍니다."""
    return state["next"]
