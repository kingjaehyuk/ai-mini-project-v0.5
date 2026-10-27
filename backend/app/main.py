"""
FastAPI 서버
──────────────────────────────────────────────────────────────
실행 (backend 폴더에서)
    uv run uvicorn app.main:app --reload --port 8000

API
  GET  /api/config                  아이템 목록, 분석 대상 기업, 테스트 질문
  POST /api/chat                    질문 실행 → 이벤트를 SSE(Server-Sent Events)로 스트리밍
  GET  /api/runs/{run_id}/report    실행이 끝난 보고서(Markdown) 조회
"""
from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from app.agent.graph import graph
from app.config import get_item, load_items, settings
from app.events import Done, ErrorEvent, Progress, Report, RunStart
from app.llm import Budget

app = FastAPI(title="AI Service Mini Project")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 실행이 끝난 보고서를 run_id 별로 보관 (서버를 재시작하면 사라짐)
REPORTS: dict[str, str] = {}


class ChatRequest(BaseModel):
    item_id: str
    query: str
    test_mode: bool | None = None   # 화면에서 선택한 실행 모드 (없으면 .env 의 POOR_OUTPUT_TEST 사용)


@app.get("/api/config")
def get_config():
    return {
        item_id: {
            "name": item["name"],
            "competitors": item["competitors"],
            "test_questions": item["test_questions"],
        }
        for item_id, item in load_items().items()
    }


def to_sse(event: dict) -> str:
    """이벤트 dict 하나를 SSE 형식의 문자열로 바꿉니다."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        item = get_item(req.item_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    run_id = uuid.uuid4().hex[:12]
    test_mode = settings.poor_output_test if req.test_mode is None else req.test_mode
    initial_state = {
        "run_id": run_id,
        "item_id": req.item_id,
        "item_name": item["name"],
        "competitors": item["competitors"],
        "query": req.query,
        "test_mode": test_mode,
    }

    async def event_stream():
        started = time.perf_counter()
        totals = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0}
        final_draft = ""
        budget = Budget(limit_usd=settings.cost_limit_usd)   # 이 실행의 모든 노드가 함께 쓰는 비용 예산

        yield to_sse(RunStart(
            run_id=run_id, item_id=req.item_id, item_name=item["name"],
            competitors=item["competitors"], query=req.query, test_mode=test_mode,
        ).model_dump())
        yield to_sse(Progress(node="budget", message=f"LLM 비용 상한: ${settings.cost_limit_usd} (실행 1회)").model_dump())

        try:
            # stream_mode="values" : 각 단계가 끝난 뒤의 상태 전체
            # stream_mode="custom" : 노드 안에서 emit() 으로 보낸 이벤트
            async for mode, payload in graph.astream(
                initial_state,
                stream_mode=["custom", "values"],
                config={"recursion_limit": settings.recursion_limit, "configurable": {"budget": budget}},
            ):
                if mode == "values":
                    final_draft = payload.get("draft", final_draft)
                    continue
                if payload.get("type") == "llm_usage":
                    totals["llm_calls"] += 1
                    totals["input_tokens"] += payload["input_tokens"]
                    totals["output_tokens"] += payload["output_tokens"]
                    totals["total_cost_usd"] += payload["cost_usd"]
                yield to_sse(payload)

            REPORTS[run_id] = final_draft
            yield to_sse(Report(run_id=run_id, markdown=final_draft).model_dump())

        except GraphRecursionError:
            yield to_sse(ErrorEvent(
                message=f"실행 단계가 상한({settings.recursion_limit})을 넘어 중단되었습니다."
            ).model_dump())
        except Exception as e:  # noqa: BLE001
            yield to_sse(ErrorEvent(message=f"{type(e).__name__}: {e}").model_dump())

        yield to_sse(Done(
            run_id=run_id,
            total_duration_ms=int((time.perf_counter() - started) * 1000),
            **totals,
        ).model_dump())

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/runs/{run_id}/report", response_class=PlainTextResponse)
def get_report(run_id: str):
    if run_id not in REPORTS:
        raise HTTPException(status_code=404, detail="보고서를 찾을 수 없습니다.")
    return REPORTS[run_id]
