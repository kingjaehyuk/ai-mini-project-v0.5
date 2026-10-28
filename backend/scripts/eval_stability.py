"""
최종 보고서 평가 판정 안정성 확인
──────────────────────────────────────────────────────────────
사용법 (backend 폴더에서 실행)
    uv run python scripts/eval_stability.py              # HBF Q2
    uv run python scripts/eval_stability.py --item hbf --q q2

처리 순서
  1) 질문을 정상 모드로 1회 실행해 최종 상태(보고서, 근거, 기업별 분석)를 얻습니다.
  2) 같은 최종 상태의 보고서를 evaluate_report() 로 2회 판정합니다.
  3) 항목별 판정과 달라진 항목을 출력합니다.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))   # backend/ 를 import 경로에 추가

from app.agent.graph import graph
from app.agent.nodes import evaluate_report
from app.config import get_item, settings
from app.llm import Budget


async def main(item_id: str, q: str) -> None:
    item = get_item(item_id)
    query = item["test_questions"][q]
    state = await graph.ainvoke(
        {"run_id": "stability", "item_id": item_id, "item_name": item["name"],
         "competitors": item["competitors"], "query": query, "test_mode": False},
        config={"recursion_limit": settings.recursion_limit,
                "configurable": {"budget": Budget(limit_usd=settings.cost_limit_usd)}},
    )
    print(f"질문: {query}")
    print(f"실행 결과: status={state.get('status')}, 평가 {state.get('eval_attempts')}회\n")

    # 그래프 밖에서 평가하므로 비용 상한은 적용되지 않습니다. (판정 2회 = Judge 호출 2회)
    runs = [(await evaluate_report(state))[0] for _ in range(2)]
    changed = 0
    for name in runs[0]:
        (p1, r1), (p2, r2) = runs[0][name], runs[1][name]
        mark = "같음" if p1 == p2 else "다름"
        changed += p1 != p2
        print(f"[{name}] 1회 {'통과' if p1 else '미통과'} / 2회 {'통과' if p2 else '미통과'} ({mark})")
        print(f"  1회 사유: {r1}\n  2회 사유: {r2}")
    print(f"\n판정이 다른 항목 수: {changed}개 / {len(runs[0])}개")
    print("\n--- 평가한 보고서 ---\n" + state.get("draft", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", default="hbf")
    parser.add_argument("--q", default="q2")
    args = parser.parse_args()
    asyncio.run(main(args.item, args.q))
