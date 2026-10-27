"""
작업자(worker) 노드와 평가 노드
──────────────────────────────────────────────────────────────
  researcher          근거 자료 수집
  competitor_analyst  분석 대상 기업별 현황 분석
  writer              보고서 작성 (답변 스트리밍)
  step_evaluator      작성된 보고서 평가
"""
from __future__ import annotations

import asyncio
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    COMPETITOR_ANALYST_SYSTEM,
    EVALUATOR_SYSTEM,
    WRITER_SYSTEM,
    format_evidence,
    format_findings,
)
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.config import settings
from app.events import BranchEnd, BranchStart, Evaluation, Progress, Token, emit
from app.llm import ainvoke_tracked, astream_tracked
from app.retrieval import index_exists, search_docs, search_web, web_search_enabled


# ──────────────────────────────────────────────────────────────
# researcher : 근거 자료 수집
# ──────────────────────────────────────────────────────────────
@traced("researcher")
async def researcher(state: AgentState) -> dict:
    item_id, query = state["item_id"], state["query"]

    if not index_exists(item_id):
        emit(Progress(node="researcher", message=f"'{item_id}' 문서 인덱스가 없습니다. scripts/ingest.py 를 먼저 실행하세요."))
    docs = await search_docs(item_id, query, k=4)
    emit(Progress(node="researcher", message=f"문서 검색 {len(docs)}건"))

    web = []
    if web_search_enabled():
        web = await search_web(f"{state['item_name']} {query}", max_results=3)
        emit(Progress(node="researcher", message=f"웹 검색 {len(web)}건"))
    else:
        emit(Progress(node="researcher", message="TAVILY_API_KEY 가 없어 웹 검색을 건너뜁니다."))

    return {"evidence": docs + web}


# ──────────────────────────────────────────────────────────────
# competitor_analyst : 분석 대상 기업별 현황 분석
#  - 기업마다 근거를 추가로 찾고, LLM으로 현황과 TRL을 추정합니다.
#  - 기업 하나의 분석을 "갈래(branch)"라고 부르며, 갈래마다 시작/종료 이벤트를 남깁니다.
# ──────────────────────────────────────────────────────────────
async def analyze_competitor(state: AgentState, competitor: str) -> dict:
    """기업 하나를 분석해 {"competitor": 기업명, "analysis": 분석 내용, "sources": [...]} 를 돌려줍니다."""
    item_name = state["item_name"]
    search_query = f"{competitor} {item_name}"

    docs = await search_docs(state["item_id"], search_query, k=3)
    web = await search_web(search_query, max_results=2)
    evidence = docs + web

    messages = [
        SystemMessage(content=COMPETITOR_ANALYST_SYSTEM),
        HumanMessage(content=(
            f"분석 아이템: {item_name}\n분석 대상 기업: {competitor}\n\n"
            f"근거 자료:\n{format_evidence(evidence)}"
        )),
    ]
    analysis = await ainvoke_tracked(messages, node="competitor_analyst")
    return {
        "competitor": competitor,
        "analysis": analysis,
        "sources": [e["source"] for e in evidence],
    }


async def run_branch(state: AgentState, competitor: str) -> dict:
    """갈래 하나를 실행하고 시작/종료 이벤트와 소요 시간을 남깁니다."""
    emit(BranchStart(node="competitor_analyst", branch=competitor))
    started = time.perf_counter()

    finding = await analyze_competitor(state, competitor)

    elapsed = int((time.perf_counter() - started) * 1000)
    emit(BranchEnd(node="competitor_analyst", branch=competitor, duration_ms=elapsed))
    return finding


@traced("competitor_analyst")
async def competitor_analyst(state: AgentState) -> dict:
    competitors = state.get("competitors", [])
    emit(Progress(node="competitor_analyst", message=f"분석 대상 {len(competitors)}개 기업: {', '.join(competitors)}"))

    # Fan-out: 기업별 갈래를 동시에 실행 / Fan-in: 모든 기업의 결과를 기업 순서대로 모음
    findings = await asyncio.gather(*(run_branch(state, c) for c in competitors))
    emit(Progress(node="competitor_analyst", message=f"기업별 분석 결과 {len(findings)}건 수집"))

    return {"competitor_findings": list(findings)}


# ──────────────────────────────────────────────────────────────
# writer : 보고서 작성 (작성 중인 내용을 Frontend로 스트리밍)
# ──────────────────────────────────────────────────────────────
POOR_OUTPUT_SAMPLE = (
    "## 1. 요약\n해당 기술은 업계에서 많은 관심을 받고 있으며 향후 성장이 기대됩니다. "
    "여러 기업이 개발을 진행하고 있고 시장 상황에 따라 경쟁이 치열해질 것으로 보입니다.\n\n"
    "## 2. 기술 개요\n이 기술은 성능과 효율을 높이기 위한 차세대 기술입니다. "
    "다양한 분야에서 활용될 수 있으며 중요성이 계속 커지고 있습니다.\n\n"
    "## 3. 경쟁사별 현황\n주요 기업들은 각자의 전략에 따라 기술 개발에 힘쓰고 있습니다. "
    "각 기업의 기술 수준은 상당히 높은 편으로 판단됩니다.\n\n"
    "## 4. 시사점\n지속적인 투자와 연구 개발이 필요합니다.\n\n"
    "## 5. 한계와 불확실성\n시장 상황은 변할 수 있습니다.\n"
)


@traced("writer")
async def writer(state: AgentState) -> dict:
    # 점검용 테스트 모드: 근거 없이 일반론만 담긴 "부실한 결과"를 만듭니다.
    if state.get("test_mode"):
        emit(Progress(node="writer", message="[테스트 모드] 부실한 중간 결과를 생성합니다."))
        emit(Token(node="writer", text=POOR_OUTPUT_SAMPLE))
        return {"draft": POOR_OUTPUT_SAMPLE}

    content = (
        f"사용자 질문: {state['query']}\n"
        f"분석 아이템: {state['item_name']}\n"
        f"분석 대상 기업: {', '.join(state.get('competitors', []))}\n\n"
        f"근거 자료:\n{format_evidence(state.get('evidence', []))}\n\n"
        f"경쟁사 분석 결과:\n{format_findings(state.get('competitor_findings', []))}"
    )
    # 재작성: 이전 평가의 실패 사유와 개선 요구를 반영합니다.
    if state.get("status") == "eval_failed" and state.get("eval_reason"):
        emit(Progress(node="writer", message="이전 평가 피드백을 반영해 다시 작성합니다."))
        content += f"\n\n이전 보고서 평가 피드백 (반드시 반영):\n{state['eval_reason']}"

    messages = [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=content)]
    draft = await astream_tracked(messages, node="writer")
    return {"draft": draft}


# ──────────────────────────────────────────────────────────────
# step_evaluator : 보고서 평가
#  - 분량을 먼저 확인하고, 질문 충족·근거 일치·근거 없는 추정을 LLM으로 평가합니다.
#  - 통과하면 종료, 통과하지 못하면 피드백과 함께 writer 가 다시 작성합니다.
#  - 평가 횟수가 상한(settings.max_eval_attempts)에 도달하면 실패 사유를 남기고 종료합니다.
# ──────────────────────────────────────────────────────────────
MIN_REPORT_CHARS = 300


def parse_evaluation(text: str) -> tuple[bool, str]:
    """평가 LLM 응답(JSON)을 (통과 여부, 사유+개선 요구)로 바꿉니다. 해석할 수 없으면 실패로 봅니다."""
    raw = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return False, "평가 결과를 해석하지 못했습니다. 근거 [n]를 붙여 질문에 직접 답하도록 다시 작성하세요."
    passed = result.get("passed") is True
    reason = str(result.get("reason", "")).strip()
    feedback = str(result.get("feedback", "")).strip()
    if not passed and feedback:
        reason = f"{reason} / 개선 요구: {feedback}"
    return passed, reason


@traced("step_evaluator")
async def step_evaluator(state: AgentState) -> dict:
    draft = state.get("draft", "")
    attempt = state.get("eval_attempts", 0) + 1

    if len(draft.strip()) < MIN_REPORT_CHARS:
        passed, reason = False, f"보고서 분량 부족 ({len(draft.strip())}자) / 개선 요구: {MIN_REPORT_CHARS}자 이상으로 작성하세요."
    else:
        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM),
            HumanMessage(content=(
                f"사용자 질문: {state['query']}\n\n"
                f"근거 자료:\n{format_evidence(state.get('evidence', []))}\n\n"
                f"경쟁사 분석 결과:\n{format_findings(state.get('competitor_findings', []))}\n\n"
                f"평가할 보고서:\n{draft}"
            )),
        ]
        passed, reason = parse_evaluation(await ainvoke_tracked(messages, node="step_evaluator"))

    emit(Evaluation(stage="report", passed=passed, reason=reason, attempt=attempt))

    if passed:
        status = "completed"
    elif attempt >= settings.max_eval_attempts:
        status = "max_attempts"
        emit(Progress(
            node="step_evaluator",
            message=f"재시도 상한({settings.max_eval_attempts}회) 도달로 평가 미통과 상태로 종료합니다. 사유: {reason}",
        ))
    else:
        status = "eval_failed"

    return {"eval_passed": passed, "eval_reason": reason, "eval_attempts": attempt, "status": status}


def route_after_evaluation(state: AgentState) -> str:
    return "writer" if state.get("status") == "eval_failed" else "end"
