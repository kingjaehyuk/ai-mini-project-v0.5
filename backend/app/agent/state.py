"""
그래프 상태(State)
──────────────────────────────────────────────────────────────
모든 노드가 함께 읽고 쓰는 공유 데이터입니다.
노드는 상태 전체를 돌려주지 않고, "바꾸고 싶은 항목만" 담은 dict를 돌려줍니다.
LangGraph가 그 dict를 기존 상태에 반영합니다.
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ── 입력 ─────────────────────────────
    run_id: str
    item_id: str
    item_name: str
    competitors: list[str]      # 분석 대상 기업 (items.yaml)
    query: str                  # 사용자 질문
    test_mode: bool             # 점검용 테스트 모드 (True면 writer가 부실한 결과를 만듦)

    # ── Supervisor ──────────────────────
    intent: str                 # 질문 의도: "fact"(단일 사실) | "compare"(경쟁사 비교) | "latest"(최신 정보)
    next: str                   # 다음에 실행할 노드 이름

    # ── 작업 결과 ────────────────────────
    evidence: list[dict]                # researcher 가 모은 근거 자료
    search_query: str                   # 다음 검색에 쓸 쿼리 (재검색 시 보완된 쿼리)
    search_attempts: int                # 검색 실행 횟수 (재검색 상한 판단용)
    evidence_sufficient: bool           # 근거 수집 단계 평가 통과 여부 (check_research)
    evidence_note: str                  # 근거 수집 평가 사유 (다음 재검색 입력) / 근거 한계 (writer 에 전달)
    competitor_findings: list[dict]     # competitor_analyst 의 경쟁사별 분석 결과
    draft: str                          # writer 가 작성한 보고서 초안

    # ── 평가 ─────────────────────────────
    eval_passed: bool
    eval_reason: str
    eval_attempts: int
    status: str                 # "completed"(평가 통과) | "eval_failed"(재작성 예정) | "max_attempts"(재시도 상한 도달로 종료) | "budget_exceeded"(비용 상한 도달로 종료)
