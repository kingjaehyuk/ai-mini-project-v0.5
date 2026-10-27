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
    plan: list[str]             # 실행할 작업 목록
    plan_index: int             # 지금까지 진행한 작업 수
    next: str                   # 다음에 실행할 노드 이름

    # ── 작업 결과 ────────────────────────
    evidence: list[dict]                # researcher 가 모은 근거 자료
    competitor_findings: list[dict]     # competitor_analyst 의 경쟁사별 분석 결과
    draft: str                          # writer 가 작성한 보고서 초안

    # ── 평가 ─────────────────────────────
    eval_passed: bool
    eval_reason: str
    eval_attempts: int
    status: str                 # "completed"(평가 통과) | "eval_failed"(재작성 예정) | "max_attempts"(재시도 상한 도달로 종료)
