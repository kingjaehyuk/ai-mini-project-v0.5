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
import re
import time
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import (
    COMPETITOR_ANALYST_SYSTEM,
    REPORT_JUDGE_SYSTEM,
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
#  - 수집 후 근거가 질문에 답하기에 충분한지 LLM으로 판단합니다. (경쟁사 비교 질문은 코드 기준만 사용)
#  - 근거 수집 단계 평가(check_research) 결과를 evaluation 이벤트로 남기고 Supervisor 에 넘깁니다.
#    미통과면 Supervisor 가 재검색하며, 미통과 사유는 다음 시도의 근거 판단 입력에 들어갑니다.
# ──────────────────────────────────────────────────────────────
MIN_EVIDENCE = 3


def check_evidence_rules(evidence: list[dict]) -> list[str]:
    """근거 수집 단계의 코드 기준: 근거 MIN_EVIDENCE건 이상, 모든 근거에 출처 (미통과 사유 목록, 없으면 빈 목록)"""
    problems = []
    if len(evidence) < MIN_EVIDENCE:
        problems.append(f"근거 {len(evidence)}건 ({MIN_EVIDENCE}건 미만)")
    no_source = sum(1 for e in evidence if not str(e.get("source", "")).strip())
    if no_source:
        problems.append(f"출처 없는 근거 {no_source}건")
    return problems


def check_research(evidence: list[dict], sufficient: bool, note: str) -> tuple[bool, str]:
    """근거 수집 단계 통과 기준: 코드 기준(check_evidence_rules) + LLM 근거 충족 판단 "충분" """
    problems = check_evidence_rules(evidence)
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
        f"사용자 질문: {state['query']}\n질문 의도: {state.get('intent', '')}\n"
        f"이번 검색 쿼리: {query}\n\n"
        f"근거 자료:\n{format_evidence(evidence)}"
    )
    # 코드 기준 미통과(근거 수·출처)를 알려, 그 부족을 메울 보완 쿼리를 만들게 합니다.
    rule_problems = check_evidence_rules(evidence)
    if rule_problems:
        content += f"\n\n코드 기준 미통과 (보완 쿼리 필수): {' / '.join(rule_problems)}"
    # 재검색: 직전 평가의 미통과 사유를 근거 판단과 보완 쿼리 작성에 반영합니다.
    if previous_reason:
        content += f"\n\n직전 근거 수집 평가 미통과 사유 (해결됐는지 확인하고, 남은 부족을 보완 쿼리에 반영):\n{previous_reason}"
    # 경쟁사 비교(compare): 기업별 근거는 기업별 분석에서 따로 모으므로 코드 기준만으로 판정합니다.
    # LLM은 코드 기준 미통과일 때 보완 쿼리를 만드는 데만 씁니다.
    compare = state.get("intent") == "compare"
    if compare and not rule_problems:
        result = {"reason": "공통 배경 근거 확보 (기업별 근거는 기업별 분석에서 수집)"}
    else:
        messages = [SystemMessage(content=SUFFICIENCY_SYSTEM), HumanMessage(content=content)]
        try:
            result = parse_json(await ainvoke_tracked(messages, node="researcher")) or {}
        except BudgetExceeded as e:
            emit(Progress(node="researcher", message=f"[비용 상한] 근거 충족 판단 생략: {e}"))
            result = {"reason": "비용 상한 도달로 근거 충족 여부를 판단하지 못함"}
    sufficient = compare or result.get("sufficient") is True
    note = str(result.get("reason", "")).strip() or "근거 충족 여부를 판단하지 못함"

    passed, reason = check_research(evidence, sufficient, note)
    emit(Evaluation(stage="researcher", passed=passed, reason=reason, attempt=attempt))

    next_query = str(result.get("next_query", "")).strip()
    # 재검색이 이어지는 경우에만 (재검색 상한 전, 비용 상한 미도달) 같은 쿼리 사용을 알립니다.
    if not passed and next_query in ("", query) and attempt < settings.max_search_attempts and not budget_exceeded():
        emit(Progress(node="researcher", message="보완 쿼리를 만들지 못해 재검색 시 같은 쿼리를 사용합니다."))

    return {
        "evidence": evidence,
        "search_attempts": attempt,
        "search_query": next_query or query,
        "evidence_sufficient": passed,
        "evidence_note": reason,
    }


# ──────────────────────────────────────────────────────────────
# competitor_analyst : 분석 대상 기업별 현황 분석
#  - 기업마다 근거를 추가로 찾고, LLM으로 현황과 TRL을 추정합니다.
#  - 기업 하나의 분석을 "갈래(branch)"라고 부르며, 갈래마다 시작/종료 이벤트를 남깁니다.
#  - 기업별로 찾은 근거는 공유 근거 목록(evidence)에 중복 없이 합치고,
#    분석 결과의 근거 번호 [n]을 공유 목록 번호로 바꿉니다. (writer 와 보고서 평가가 같은 번호를 보도록)
#  - 기업별 분석 단계 평가(check_analysis) 결과를 evaluation 이벤트로 남기고 Supervisor 에 넘깁니다.
#    미통과면 Supervisor 가 재분석하며, 미통과 기업만 그 사유를 입력에 넣어 다시 분석합니다.
# ──────────────────────────────────────────────────────────────
POOR_ANALYSIS_SAMPLE = (
    "- 현황 요약: 해당 기업은 기술 개발을 활발히 진행하고 있으며 기술 수준이 높은 편입니다.\n"
    "- 불확실한 점: 시장 상황에 따라 달라질 수 있습니다."
)


async def analyze_competitor(state: AgentState, competitor: str, feedback: str = "") -> dict:
    """기업 하나를 분석해 {"competitor": 기업명, "analysis": 분석 내용, "sources": [...], "evidence": [...]} 를 돌려줍니다.
    analysis 의 근거 번호 [n]은 이 기업의 evidence 기준이며, competitor_analyst 가 공유 목록 번호로 바꿉니다."""
    item_name = state["item_name"]
    search_query = f"{competitor} {item_name}"

    docs = await search_docs(state["item_id"], search_query, k=3)
    web = await search_web(search_query, max_results=2)
    evidence = docs + web
    sources = [e["source"] for e in evidence]

    # 점검용 테스트 모드: TRL과 근거 번호가 없는 "부실한 결과"를 만듭니다.
    if state.get("test_mode"):
        emit(Progress(node="competitor_analyst", message=f"[테스트 모드] {competitor} 부실한 분석 결과를 생성합니다."))
        return {"competitor": competitor, "analysis": POOR_ANALYSIS_SAMPLE, "sources": sources, "evidence": evidence}

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
        "evidence": evidence,
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
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(text: str) -> list[int]:
    """본문의 근거 번호 [n], [n, m] 을 모두 꺼냅니다."""
    return [int(n) for group in CITATION_PATTERN.findall(text) for n in re.split(r"\s*,\s*", group)]


def merge_branch_evidence(evidence: list[dict], finding: dict) -> None:
    """기업 하나의 근거를 공유 근거 목록에 중복 없이 합치고, 분석의 [n]을 공유 목록 번호로 바꿉니다.
    그 기업의 근거 범위를 벗어난 번호는 [?] 로 바꾸고 citation_error 에 남깁니다."""
    index = {(e["source"], e["content"]): i for i, e in enumerate(evidence, start=1)}
    local = finding.pop("evidence", [])
    mapping = {}
    for n, e in enumerate(local, start=1):
        key = (e["source"], e["content"])
        if key not in index:
            evidence.append(e)
            index[key] = len(evidence)
        mapping[n] = index[key]

    bad = sorted({n for n in cited_numbers(finding["analysis"]) if n not in mapping})
    finding["citation_error"] = f"근거 번호 {bad} 범위 밖 (근거 {len(local)}건)" if bad else ""

    def to_global(m: re.Match) -> str:
        numbers = [mapping[int(n)] for n in re.split(r"\s*,\s*", m.group(1)) if int(n) in mapping]
        return f"[{', '.join(map(str, numbers))}]" if numbers else "[?]"

    finding["analysis"] = CITATION_PATTERN.sub(to_global, finding["analysis"])


def check_finding(finding: dict, evidence_count: int) -> list[str]:
    """기업 하나의 분석 결과에서 부족한 점을 찾습니다. (없으면 빈 목록)"""
    analysis = finding.get("analysis", "")
    problems = []
    match = TRL_PATTERN.search(analysis)
    value = match.group(1).strip() if match else ""
    citations = cited_numbers(analysis)

    if "판단 불가" in value:
        if not any(len(r.strip(" -.")) >= 2 for r in REASON_PATTERN.findall(analysis)):
            problems.append("TRL 판단 불가 이유 없음")
    elif re.search(r"\b[1-9]\b", value):
        if not citations:
            problems.append("TRL 근거 번호 [n] 없음")
    else:
        problems.append("TRL 누락")

    if finding.get("citation_error"):
        problems.append(finding["citation_error"])
    out_of_range = sorted({n for n in citations if not 1 <= n <= evidence_count})
    if out_of_range:
        problems.append(f"근거 번호 {out_of_range} 범위 밖 (근거 {evidence_count}건)")
    return problems


def check_analysis(competitors: list[str], findings: list[dict], evidence_count: int) -> tuple[bool, str, dict[str, str]]:
    """기업별 분석 단계 통과 기준: 설정 기업 수 = 결과 기업 수, 기업마다 check_finding 통과
    (통과 여부, 사유, {미통과 기업: 사유}) 를 돌려줍니다."""
    names = [f["competitor"] for f in findings]
    failed = {c: "분석 결과 누락" for c in competitors if c not in names}
    for f in findings:
        problems = check_finding(f, evidence_count)
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

    # 기업별 근거를 공유 근거 목록에 합치고, 분석의 근거 번호를 공유 목록 번호로 바꿉니다.
    evidence = list(state.get("evidence", []))
    before = len(evidence)
    for f in results:
        merge_branch_evidence(evidence, f)
    merged = {**previous, **{f["competitor"]: f for f in results}}
    findings = [merged[c] for c in competitors if c in merged]
    emit(Progress(node="competitor_analyst",
                  message=f"기업별 분석 결과 {len(findings)}건 수집, 기업별 근거 {len(evidence) - before}건을 근거 목록에 추가 (총 {len(evidence)}건)"))

    passed, reason, failed = check_analysis(competitors, findings, len(evidence))
    emit(Evaluation(stage="competitor_analyst", passed=passed, reason=reason, attempt=attempt))

    return {
        "evidence": evidence,
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
# step_evaluator : 최종 보고서 평가
#  - 5개 항목을 판정합니다. 모두 통과해야 보고서가 통과합니다.
#      Answer Relevance        LLM Judge                 질문의 핵심 요구에 답함 (1~5점 중 4점 이상)
#      Faithfulness            LLM Judge (+ 코드 비율)    근거로 뒷받침되는 주장 비율 ≥ 0.8
#      Citation Accuracy       코드 (번호 범위) + LLM Judge (번호 누락 · 뒷받침 여부)
#      Completeness            코드                      설정한 기업 수 = TRL과 근거가 있는 기업 수 (경쟁사 비교 질문)
#      Uncertainty Disclosure  코드 (한계 절 유무) + LLM Judge (한계 서술, 근거 없는 TRL 단정 없음)
#  - LLM Judge 는 평가 1회에 한 번 호출합니다. (REPORT_JUDGE_SYSTEM)
#  - 항목마다 evaluation 이벤트(stage="final:<항목>")를 남기고, 미통과 항목의 사유를 writer 재작성에 전달합니다.
#  - 평가 횟수가 상한(settings.max_eval_attempts)에 도달하면 실패 사유를 남기고 종료합니다.
# ──────────────────────────────────────────────────────────────
MIN_RELEVANCE_SCORE = 4
MIN_FAITHFULNESS = 0.8
REPORT_ITEMS = ["Answer Relevance", "Faithfulness", "Citation Accuracy", "Completeness", "Uncertainty Disclosure"]
NOT_JUDGED = "비용 상한으로 LLM Judge 미평가"


def split_sections(report: str) -> dict[str, str]:
    """보고서를 {## 제목: 본문} 으로 나눕니다."""
    sections, title = {}, ""
    for line in report.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            sections[title] = ""
        elif title:
            sections[title] += line + "\n"
    return sections


def check_citation_range(report: str, evidence_count: int) -> tuple[bool, str]:
    """Citation Accuracy 코드 부분: 인용 번호가 있고, 모든 [n]이 1 ~ 근거 수 범위 안"""
    citations = cited_numbers(report)
    if not citations:
        return False, "근거 번호 [n] 없음"
    out_of_range = sorted({n for n in citations if not 1 <= n <= evidence_count})
    if out_of_range:
        return False, f"근거 번호 {out_of_range} 범위 밖 (근거 {evidence_count}건)"
    return True, f"인용 {len(citations)}개 모두 근거 범위(1~{evidence_count}) 안"


def check_completeness(report: str, intent: str, competitors: list[str]) -> tuple[bool, str]:
    """Completeness (코드): 설정한 기업 수 = TRL 추정과 근거 [n]이 있는 기업 수 ("판단 불가"는 근거 없이 인정)"""
    if intent != "compare":
        return True, "경쟁사 비교 질문이 아니라 기업별 TRL 해당 없음"
    lines = report.splitlines()
    has_trl_header = "TRL" in report
    missing = []
    for c in competitors:
        ok = False
        for i, line in enumerate(lines):
            if c.lower() not in line.lower():
                continue
            # 표 형식이면 그 행만, 아니면 기업명이 나온 줄부터 3줄 아래까지를 그 기업의 서술로 봅니다.
            # (다른 기업명이나 제목이 나오면 거기서 끊습니다.)
            window_lines = [line]
            if not line.lstrip().startswith("|"):
                for nxt in lines[i + 1:i + 4]:
                    if nxt.startswith("#") or any(o.lower() in nxt.lower() for o in competitors if o != c):
                        break
                    window_lines.append(nxt)
            window = "\n".join(window_lines)
            has_trl = "TRL" in window or (line.lstrip().startswith("|") and has_trl_header and re.search(r"\b[1-9]\b", line))
            if "판단 불가" in window or (has_trl and CITATION_PATTERN.search(window)):
                ok = True
                break
        if not ok:
            missing.append(c)
    if missing:
        return False, f"TRL과 근거가 있는 기업 {len(competitors) - len(missing)}개 / 설정 {len(competitors)}개 (없음: {', '.join(missing)})"
    return True, f"설정한 {len(competitors)}개 기업 모두 TRL과 근거 있음"


def check_limitation_section(report: str) -> tuple[bool, str]:
    """Uncertainty Disclosure 코드 부분: 한계 절이 있고 내용이 있음"""
    body = next((b for t, b in split_sections(report).items() if "한계" in t), None)
    if body is None:
        return False, "한계 절 없음"
    if not body.strip():
        return False, "한계 절 내용 없음"
    return True, "한계 절 있음"


def judge_items(result: dict | None) -> dict[str, tuple[bool, str]]:
    """LLM Judge 응답(JSON)을 항목별 판정으로 바꿉니다.
    Answer Relevance, Faithfulness 는 최종 판정, Citation Accuracy, Uncertainty Disclosure 는 Judge 부분 판정입니다."""
    if result is None:
        reason = "Judge 응답을 해석하지 못함"
        return {name: (False, reason) for name in ["Answer Relevance", "Faithfulness", "Citation Accuracy", "Uncertainty Disclosure"]}

    relevance = result.get("answer_relevance") or {}
    try:
        score = int(relevance.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    items = {"Answer Relevance": (score >= MIN_RELEVANCE_SCORE,
                                  f"{score}점 (기준 {MIN_RELEVANCE_SCORE}점): {relevance.get('reason', '')}")}

    claims = [c for c in result.get("claims") or [] if isinstance(c, dict)]
    supported = [c for c in claims if c.get("supported") is True]
    ratio = len(supported) / len(claims) if claims else 0.0
    unsupported = [str(c.get("claim", ""))[:40] for c in claims if c.get("supported") is not True]
    reason = f"지지 비율 {ratio:.2f} ({len(supported)}/{len(claims)}, 기준 {MIN_FAITHFULNESS})"
    if unsupported:
        reason += f" / 근거 없는 주장 {len(unsupported)}건: {'; '.join(unsupported[:3])}"
    items["Faithfulness"] = (bool(claims) and ratio >= MIN_FAITHFULNESS, reason)

    no_number = [str(c.get("claim", ""))[:40] for c in claims if not c.get("citations")]
    mismatch = [str(c.get("claim", ""))[:40] for c in claims if c.get("citations") and c.get("citation_ok") is not True]
    problems = []
    if no_number:
        problems.append(f"번호 누락 {len(no_number)}건: {'; '.join(no_number[:3])}")
    if mismatch:
        problems.append(f"인용 근거가 뒷받침하지 않음 {len(mismatch)}건: {'; '.join(mismatch[:3])}")
    if not claims:
        problems.append("판정한 사실 주장 없음")
    items["Citation Accuracy"] = (not problems, " / ".join(problems) or f"주장 {len(claims)}건 모두 인용 근거가 뒷받침")

    uncertainty = result.get("uncertainty") or {}
    items["Uncertainty Disclosure"] = (uncertainty.get("passed") is True, str(uncertainty.get("reason", "")).strip())
    return items


def combine(code: tuple[bool, str], judge: tuple[bool, str]) -> tuple[bool, str]:
    """코드 판정과 Judge 판정을 함께 써야 하는 항목: 둘 다 통과해야 통과"""
    return code[0] and judge[0], f"코드: {code[1]} / Judge: {judge[1]}"


def code_items(state: AgentState) -> dict[str, tuple[bool, str]]:
    """코드로 판정하는 부분 (Citation Accuracy 번호 범위, Completeness, Uncertainty Disclosure 한계 절)"""
    draft = state.get("draft", "")
    return {
        "Citation Accuracy": check_citation_range(draft, len(state.get("evidence", []))),
        "Completeness": check_completeness(draft, state.get("intent", ""), state.get("competitors", [])),
        "Uncertainty Disclosure": check_limitation_section(draft),
    }


def merge_items(code: dict[str, tuple[bool, str]], judge: dict[str, tuple[bool, str]]) -> dict[str, tuple[bool, str]]:
    return {
        "Answer Relevance": judge["Answer Relevance"],
        "Faithfulness": judge["Faithfulness"],
        "Citation Accuracy": combine(code["Citation Accuracy"], judge["Citation Accuracy"]),
        "Completeness": code["Completeness"],
        "Uncertainty Disclosure": combine(code["Uncertainty Disclosure"], judge["Uncertainty Disclosure"]),
    }


def is_excluded_section(title: str) -> bool:
    """주장 추출에서 빼는 절 (시사점·한계)"""
    return "시사점" in title or "한계" in title


def report_sentences(text: str) -> list[str]:
    """보고서 텍스트를 비교용 문장으로 나눕니다. (목록·표 기호, 인용 번호, 공백 제거)"""
    sentences = []
    for line in text.splitlines():
        line = re.sub(r"^[#\-*\s|]+", "", line)
        for s in re.split(r"(?<=[.다])\s+", line):
            s = re.sub(r"\[[\d,\s]+\]|\s", "", s)
            if len(s) > 5:
                sentences.append(s)
    return sentences


def drop_excluded_claims(result: dict | None, report: str) -> int:
    """Judge가 시사점·한계 절에서 뽑은 주장을 claims에서 뺍니다. (프롬프트 지시와 관계없이 코드로 보장)
    주장과 가장 비슷한 보고서 문장이 시사점·한계 절(또는 본문의 "불확실한 점" 줄)에 있으면 제외 대상으로 봅니다.
    뺀 주장 수를 돌려줍니다."""
    if not result or not isinstance(result.get("claims"), list):
        return 0
    main_lines, excluded_lines = [], []
    for title, body in split_sections(report).items():
        for line in body.splitlines():
            if is_excluded_section(title) or "불확실한 점" in line:
                excluded_lines.append(line)
            else:
                main_lines.append(line)
    main = report_sentences("\n".join(main_lines))
    excluded = report_sentences("\n".join(excluded_lines))
    if not main or not excluded:
        return 0

    def best(claim: str, sentences: list[str]) -> float:
        claim = re.sub(r"\[[\d,\s]+\]|\s", "", claim)
        return max(SequenceMatcher(None, claim, s).ratio() for s in sentences)

    kept = [c for c in result["claims"]
            if not isinstance(c, dict) or best(str(c.get("claim", "")), excluded) <= best(str(c.get("claim", "")), main)]
    dropped = len(result["claims"]) - len(kept)
    result["claims"] = kept
    return dropped


def report_for_judge(report: str) -> str:
    """보고서를 Judge 입력용으로 나눕니다. 시사점·한계 절은 주장 추출에서 빠지도록 따로 둡니다."""
    sections = split_sections(report)
    if not sections:
        return f"보고서 본문 (주장 추출 대상):\n{report}"
    main = [f"## {t}\n{b}" for t, b in sections.items() if not is_excluded_section(t)]
    excluded = [f"## {t}\n{b}" for t, b in sections.items() if is_excluded_section(t)]
    return (
        f"보고서 본문 (주장 추출 대상):\n{''.join(main) or '(없음)'}\n\n"
        f"시사점·한계 절 (주장 추출 제외, 불확실성 판정에 사용):\n{''.join(excluded) or '(없음)'}"
    )


async def evaluate_report(state: AgentState) -> tuple[dict[str, tuple[bool, str]], str]:
    """최종 보고서를 5개 항목으로 판정해 ({항목: (통과 여부, 사유)}, Judge 개선 요구) 를 돌려줍니다.
    Judge 호출이 비용 상한에 막히면 BudgetExceeded 를 그대로 올립니다."""
    limits = []
    if not state.get("evidence_sufficient", True) and state.get("evidence_note"):
        limits.append(f"근거 한계: {state['evidence_note']}")
    if not state.get("analysis_passed", True) and state.get("analysis_reason"):
        limits.append(f"기업별 분석 한계: {state['analysis_reason']}")
    messages = [
        SystemMessage(content=REPORT_JUDGE_SYSTEM),
        HumanMessage(content=(
            f"사용자 질문: {state['query']}\n질문 의도: {state.get('intent', '')}\n\n"
            f"근거 자료:\n{format_evidence(state.get('evidence', []))}\n\n"
            f"경쟁사 분석 결과:\n{format_findings(state.get('competitor_findings', []))}\n\n"
            f"실행 중 확인된 한계:\n{chr(10).join(limits) or '(없음)'}\n\n"
            f"{report_for_judge(state.get('draft', ''))}"
        )),
    ]
    result = parse_json(await ainvoke_tracked(messages, node="step_evaluator"))
    feedback = str((result or {}).get("feedback", "")).strip()
    dropped = drop_excluded_claims(result, state.get("draft", ""))
    if dropped:
        emit(Progress(node="step_evaluator", message=f"시사점·한계 절에서 뽑힌 주장 {dropped}건은 Faithfulness·Citation Accuracy 집계에서 제외"))
    return merge_items(code_items(state), judge_items(result)), feedback


@traced("step_evaluator")
async def step_evaluator(state: AgentState) -> dict:
    attempt = state.get("eval_attempts", 0) + 1

    # 비용 상한: 이미 상한에 도달했거나 평가 호출을 예약할 수 없으면 평가를 생략하고 현재 보고서로 종료합니다.
    budget_end = {"eval_passed": False, "status": "budget_exceeded",
                  "eval_reason": "비용 상한 도달로 평가를 생략했습니다."}
    if budget_exceeded():
        emit(Progress(node="step_evaluator", message="[비용 상한] 평가를 생략하고 현재 보고서로 종료합니다."))
        return budget_end

    try:
        items, feedback = await evaluate_report(state)
    except BudgetExceeded as e:
        # 코드 항목만 판정해 기록하고, Judge 항목은 미평가로 남긴 뒤 현재 보고서로 종료합니다.
        emit(Progress(node="step_evaluator", message=f"[비용 상한] LLM Judge 평가를 생략하고 현재 보고서로 종료합니다: {e}"))
        not_judged = {name: (False, NOT_JUDGED) for name in REPORT_ITEMS}
        for name, (item_passed, item_reason) in merge_items(code_items(state), not_judged).items():
            emit(Evaluation(stage=f"final:{name}", passed=item_passed, reason=item_reason, attempt=attempt))
        return budget_end

    for name, (item_passed, item_reason) in items.items():
        emit(Evaluation(stage=f"final:{name}", passed=item_passed, reason=item_reason, attempt=attempt))

    failed = [f"{name}: {r}" for name, (ok, r) in items.items() if not ok]
    passed = not failed
    if passed:
        reason = "5개 항목 모두 통과"
    else:
        reason = f"미통과 {len(failed)}개 항목 - " + " / ".join(failed)
        if feedback:
            reason += f" / 개선 요구: {feedback}"
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
