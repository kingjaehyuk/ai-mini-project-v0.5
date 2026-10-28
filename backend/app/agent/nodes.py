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
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    COMPETITOR_ANALYST_SYSTEM,
    EVALUATOR_SYSTEM,
    SUFFICIENCY_SYSTEM,
    WRITER_SYSTEM,
    WRITER_SYSTEM_BY_INTENT,
    format_evidence,
    format_findings,
    parse_json,
)
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.config import settings
from app.events import BranchEnd, BranchStart, Evaluation, Progress, Token, emit
from app.llm import BudgetExceeded, ainvoke_tracked, astream_tracked, budget_exceeded
from app.retrieval import index_exists, search_docs, search_web, web_search_enabled


# ──────────────────────────────────────────────────────────────
# researcher : 근거 자료 수집
#  - 재검색이면 보완된 쿼리(search_query)로 찾고, 기존 근거에 중복 없이 더합니다.
#  - 수집 후 근거가 질문에 답하기에 충분한지 LLM으로 판단합니다.
#  - 근거 수집 단계 평가(check_research) 결과를 evaluation 이벤트로 남기고 Supervisor 에 넘깁니다.
#    미통과면 Supervisor 가 재검색하며, 미통과 사유는 다음 시도의 근거 판단 입력에 들어갑니다.
# ──────────────────────────────────────────────────────────────
MIN_EVIDENCE = 3


def check_research(evidence: list[dict], sufficient: bool, note: str) -> tuple[bool, str]:
    """근거 수집 단계 통과 기준: 근거 MIN_EVIDENCE건 이상, 모든 근거에 출처, LLM 근거 충족 판단 "충분" """
    problems = []
    if len(evidence) < MIN_EVIDENCE:
        problems.append(f"근거 {len(evidence)}건 ({MIN_EVIDENCE}건 미만)")
    no_source = sum(1 for e in evidence if not str(e.get("source", "")).strip())
    if no_source:
        problems.append(f"출처 없는 근거 {no_source}건")
    if not sufficient:
        problems.append(f"근거 부족: {note}")
    if problems:
        return False, " / ".join(problems)
    return True, f"근거 {len(evidence)}건, 모두 출처 있음, 근거 충분: {note}"


@traced("researcher")
async def researcher(state: AgentState) -> dict:
    item_id = state["item_id"]
    query = state.get("search_query") or state["query"]
    attempt = state.get("search_attempts", 0) + 1
    previous_reason = state.get("evidence_note", "") if attempt > 1 else ""
    if attempt > 1:
        emit(Progress(node="researcher", message=f"보완 쿼리로 재검색: {query} (직전 평가 사유 반영: {previous_reason})"))

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

    evidence = list(state.get("evidence", []))
    seen = {(e["source"], e["content"]) for e in evidence}
    evidence += [e for e in docs + web if (e["source"], e["content"]) not in seen]

    # 점검용 테스트 모드: 출처 없는 근거 1건만 남긴 "부실한 결과"를 만듭니다.
    if state.get("test_mode"):
        emit(Progress(node="researcher", message="[테스트 모드] 출처 없는 근거 1건만 남깁니다."))
        sample = (docs + web)[:1] or [{"kind": "doc", "title": "(테스트)", "content": "근거 내용 없음"}]
        evidence = [{**sample[0], "source": ""}]

    content = (
        f"사용자 질문: {state['query']}\n질문 의도: {state.get('intent', '')}\n\n"
        f"근거 자료:\n{format_evidence(evidence)}"
    )
    # 재검색: 직전 평가의 미통과 사유를 근거 판단과 보완 쿼리 작성에 반영합니다.
    if previous_reason:
        content += f"\n\n직전 근거 수집 평가 미통과 사유 (해결됐는지 확인하고, 남은 부족을 보완 쿼리에 반영):\n{previous_reason}"
    messages = [SystemMessage(content=SUFFICIENCY_SYSTEM), HumanMessage(content=content)]
    try:
        result = parse_json(await ainvoke_tracked(messages, node="researcher")) or {}
    except BudgetExceeded as e:
        emit(Progress(node="researcher", message=f"[비용 상한] 근거 충족 판단 생략: {e}"))
        result = {"reason": "비용 상한 도달로 근거 충족 여부를 판단하지 못함"}
    sufficient = result.get("sufficient") is True
    note = str(result.get("reason", "")).strip() or "근거 충족 여부를 판단하지 못함"

    passed, reason = check_research(evidence, sufficient, note)
    emit(Evaluation(stage="researcher", passed=passed, reason=reason, attempt=attempt))

    return {
        "evidence": evidence,
        "search_attempts": attempt,
        "search_query": str(result.get("next_query", "")).strip() or query,
        "evidence_sufficient": passed,
        "evidence_note": reason,
    }


# ──────────────────────────────────────────────────────────────
# competitor_analyst : 분석 대상 기업별 현황 분석
#  - 기업마다 근거를 추가로 찾고, LLM으로 현황과 TRL을 추정합니다.
#  - 기업 하나의 분석을 "갈래(branch)"라고 부르며, 갈래마다 시작/종료 이벤트를 남깁니다.
#  - 기업별 분석 단계 평가(check_analysis) 결과를 evaluation 이벤트로 남기고 Supervisor 에 넘깁니다.
#    미통과면 Supervisor 가 재분석하며, 미통과 기업만 그 사유를 입력에 넣어 다시 분석합니다.
# ──────────────────────────────────────────────────────────────
POOR_ANALYSIS_SAMPLE = (
    "- 현황 요약: 해당 기업은 기술 개발을 활발히 진행하고 있으며 기술 수준이 높은 편입니다.\n"
    "- 불확실한 점: 시장 상황에 따라 달라질 수 있습니다."
)


async def analyze_competitor(state: AgentState, competitor: str, feedback: str = "") -> dict:
    """기업 하나를 분석해 {"competitor": 기업명, "analysis": 분석 내용, "sources": [...]} 를 돌려줍니다."""
    item_name = state["item_name"]
    search_query = f"{competitor} {item_name}"

    docs = await search_docs(state["item_id"], search_query, k=3)
    web = await search_web(search_query, max_results=2)
    evidence = docs + web
    sources = [e["source"] for e in evidence]

    # 점검용 테스트 모드: TRL과 근거 번호가 없는 "부실한 결과"를 만듭니다.
    if state.get("test_mode"):
        emit(Progress(node="competitor_analyst", message=f"[테스트 모드] {competitor} 부실한 분석 결과를 생성합니다."))
        return {"competitor": competitor, "analysis": POOR_ANALYSIS_SAMPLE, "sources": sources}

    content = (
        f"분석 아이템: {item_name}\n분석 대상 기업: {competitor}\n\n"
        f"근거 자료:\n{format_evidence(evidence)}"
    )
    # 재분석: 직전 평가의 미통과 사유를 반영합니다.
    if feedback:
        content += f"\n\n이전 분석 평가 피드백 (반드시 반영):\n{feedback}"
    messages = [SystemMessage(content=COMPETITOR_ANALYST_SYSTEM), HumanMessage(content=content)]
    try:
        analysis = await ainvoke_tracked(messages, node="competitor_analyst")
    except BudgetExceeded as e:
        emit(Progress(node="competitor_analyst", message=f"[비용 상한] {competitor} 분석 생략: {e}"))
        analysis = "근거 부족: 비용 상한 도달로 분석하지 못함"
    return {
        "competitor": competitor,
        "analysis": analysis,
        "sources": sources,
    }


async def run_branch(state: AgentState, competitor: str, feedback: str = "") -> dict:
    """갈래 하나를 실행하고 시작/종료 이벤트와 소요 시간을 남깁니다."""
    emit(BranchStart(node="competitor_analyst", branch=competitor))
    started = time.perf_counter()

    finding = await analyze_competitor(state, competitor, feedback)

    elapsed = int((time.perf_counter() - started) * 1000)
    emit(BranchEnd(node="competitor_analyst", branch=competitor, duration_ms=elapsed))
    return finding


TRL_PATTERN = re.compile(r"TRL\s*추정\s*[:：]\s*(.*)")
REASON_PATTERN = re.compile(r"(?:추정 근거|불확실한 점)\s*[:：]\s*(.*)")


def check_finding(finding: dict) -> list[str]:
    """기업 하나의 분석 결과에서 부족한 점을 찾습니다. (없으면 빈 목록)"""
    analysis = finding.get("analysis", "")
    problems = []
    match = TRL_PATTERN.search(analysis)
    value = match.group(1).strip() if match else ""
    citations = [int(n) for n in re.findall(r"\[(\d+)\]", analysis)]

    if "판단 불가" in value:
        if not any(len(r.strip(" -.")) >= 2 for r in REASON_PATTERN.findall(analysis)):
            problems.append("TRL 판단 불가 이유 없음")
    elif re.search(r"\b[1-9]\b", value):
        if not citations:
            problems.append("TRL 근거 번호 [n] 없음")
    else:
        problems.append("TRL 누락")

    limit = len(finding.get("sources", []))
    out_of_range = sorted({n for n in citations if not 1 <= n <= limit})
    if out_of_range:
        problems.append(f"근거 번호 {out_of_range} 범위 밖 (근거 {limit}건)")
    return problems


def check_analysis(competitors: list[str], findings: list[dict]) -> tuple[bool, str, dict[str, str]]:
    """기업별 분석 단계 통과 기준: 설정 기업 수 = 결과 기업 수, 기업마다 check_finding 통과
    (통과 여부, 사유, {미통과 기업: 사유}) 를 돌려줍니다."""
    names = [f["competitor"] for f in findings]
    failed = {c: "분석 결과 누락" for c in competitors if c not in names}
    for f in findings:
        problems = check_finding(f)
        if problems:
            failed[f["competitor"]] = ", ".join(problems)

    problems = []
    if len(names) != len(competitors) or set(names) != set(competitors):
        problems.append(f"기업 수 불일치 (설정 {len(competitors)}개, 결과 {len(names)}개)")
    problems += [f"{c} {r}" for c, r in failed.items()]
    if problems:
        return False, " / ".join(problems), failed
    return True, f"{len(names)}개 기업 모두 TRL과 근거 번호 확인", {}


@traced("competitor_analyst")
async def competitor_analyst(state: AgentState) -> dict:
    competitors = state.get("competitors", [])
    attempt = state.get("analysis_attempts", 0) + 1
    previous = {f["competitor"]: f for f in state.get("competitor_findings", [])}
    feedback = state.get("analysis_feedback", {}) if attempt > 1 else {}

    # 재분석이면 미통과 기업만 다시 분석하고, 통과한 기업의 결과는 유지합니다.
    targets = [c for c in competitors if c in feedback or c not in previous] if attempt > 1 else competitors
    if attempt > 1:
        emit(Progress(node="competitor_analyst", message=f"미통과 기업 재분석 (직전 평가 사유 반영): "
                      + " / ".join(f"{c}: {feedback.get(c, '분석 결과 누락')}" for c in targets)))
    else:
        emit(Progress(node="competitor_analyst", message=f"분석 대상 {len(competitors)}개 기업: {', '.join(competitors)}"))

    # Fan-out: 기업별 갈래를 동시에 실행 / Fan-in: 모든 기업의 결과를 기업 순서대로 모음
    results = await asyncio.gather(*(run_branch(state, c, feedback.get(c, "")) for c in targets))
    merged = {**previous, **{f["competitor"]: f for f in results}}
    findings = [merged[c] for c in competitors if c in merged]
    emit(Progress(node="competitor_analyst", message=f"기업별 분석 결과 {len(findings)}건 수집"))

    passed, reason, failed = check_analysis(competitors, findings)
    emit(Evaluation(stage="competitor_analyst", passed=passed, reason=reason, attempt=attempt))

    return {
        "competitor_findings": findings,
        "analysis_attempts": attempt,
        "analysis_passed": passed,
        "analysis_reason": reason,
        "analysis_feedback": failed,
    }


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
    # 근거가 부족한 채로 진행하는 경우, 그 한계를 보고서에 밝히도록 전달합니다.
    if not state.get("evidence_sufficient", True) and state.get("evidence_note"):
        content += f"\n\n근거 한계 (보고서의 한계와 불확실성에 반드시 명시):\n{state['evidence_note']}"
    # 기업별 분석이 평가를 통과하지 못한 채 진행하는 경우, 그 한계를 보고서에 밝히도록 전달합니다.
    if not state.get("analysis_passed", True) and state.get("analysis_reason"):
        content += f"\n\n기업별 분석 한계 (보고서의 한계와 불확실성에 반드시 명시):\n{state['analysis_reason']}"
    # 재작성: 이전 평가의 실패 사유와 개선 요구를 반영합니다.
    if state.get("status") == "eval_failed" and state.get("eval_reason"):
        emit(Progress(node="writer", message="이전 평가 피드백을 반영해 다시 작성합니다."))
        content += f"\n\n이전 보고서 평가 피드백 (반드시 반영):\n{state['eval_reason']}"

    system = WRITER_SYSTEM_BY_INTENT.get(state.get("intent", ""), WRITER_SYSTEM)
    messages = [SystemMessage(content=system), HumanMessage(content=content)]
    try:
        draft = await astream_tracked(messages, node="writer")
    except BudgetExceeded as e:
        # 비용 상한: 이전 보고서가 있으면 그대로 두고, 없으면 수집한 근거 목록으로 최소 보고서를 만듭니다.
        if state.get("draft"):
            emit(Progress(node="writer", message=f"[비용 상한] 재작성 생략, 이전 보고서를 유지합니다: {e}"))
            return {}
        emit(Progress(node="writer", message=f"[비용 상한] 보고서 작성 생략, 수집한 근거 목록으로 대신합니다: {e}"))
        draft = budget_fallback_report(state)
        emit(Token(node="writer", text=draft))
    return {"draft": draft}


def budget_fallback_report(state: AgentState) -> str:
    """비용 상한으로 LLM 보고서를 만들지 못했을 때 내보내는 최소 보고서"""
    sources = "\n".join(
        f"- [{i}] {e['title']} | {e['source']}" for i, e in enumerate(state.get("evidence", []), start=1)
    ) or "- (수집한 근거 없음)"
    return (
        f"# {state['query']}\n\n"
        "## 비용 상한 도달\n"
        f"실행 1회의 LLM 비용 상한(${settings.cost_limit_usd})에 도달해 보고서를 작성하지 못했습니다.\n\n"
        f"## 수집한 근거 자료\n{sources}\n"
    )


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

    # 비용 상한: 이미 상한에 도달했거나 평가 호출을 예약할 수 없으면 평가를 생략하고 현재 보고서로 종료합니다.
    budget_end = {"eval_passed": False, "status": "budget_exceeded",
                  "eval_reason": "비용 상한 도달로 평가를 생략했습니다."}
    if budget_exceeded():
        emit(Progress(node="step_evaluator", message="[비용 상한] 평가를 생략하고 현재 보고서로 종료합니다."))
        return budget_end

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
        try:
            passed, reason = parse_evaluation(await ainvoke_tracked(messages, node="step_evaluator"))
        except BudgetExceeded as e:
            emit(Progress(node="step_evaluator", message=f"[비용 상한] 평가를 생략하고 현재 보고서로 종료합니다: {e}"))
            return budget_end

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
