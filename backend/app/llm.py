"""
LLM / 임베딩 생성과 사용량 기록
- 모든 LLM 호출은 이 파일의 ainvoke_tracked / astream_tracked 를 거칩니다.
- 호출이 끝날 때마다 llm_usage 이벤트(토큰 수, 예상 비용)가 발생합니다.
- 실행별 비용 상한(Budget)이 있으면 호출 전에 예상 비용을 예약하고, 호출 후 실제 비용으로 정산합니다.
  예약 후 상한을 넘으면 호출하지 않고 BudgetExceeded 를 발생시킵니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.config import get_config

from app.config import PRICING_PER_1M, settings
from app.events import LlmUsage, Token, emit


@lru_cache
def get_llm() -> ChatOpenAI:
    # stream_usage=True : 스트리밍 응답에서도 토큰 사용량을 받기 위한 옵션
    return ChatOpenAI(model=settings.llm_model, temperature=0, stream_usage=True)


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICING_PER_1M.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _emit_usage(node: str, message) -> float:
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = estimate_cost(settings.llm_model, input_tokens, output_tokens)
    emit(LlmUsage(
        node=node,
        model=settings.llm_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
    ))
    return cost


# ──────────────────────────────────────────────────────────────
# 실행별 비용 상한 (공유 예산)
#  - main.py 가 실행마다 Budget 을 만들어 graph config 의 configurable["budget"] 으로 넘깁니다.
#  - 병렬 호출도 같은 Budget 을 쓰므로, 호출 전 예약(reserve)으로 동시에 상한을 넘지 않게 합니다.
#  - 예약 금액 = 입력 토큰 수 × 입력 단가 + 최대 출력 토큰 수 × 출력 단가
#    (최대 출력 토큰 수를 호출에 함께 지정하므로 실제 비용은 예약 금액을 넘지 않습니다.)
# ──────────────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS = {"writer": 2000}    # 노드별 최대 출력 토큰 수
DEFAULT_MAX_OUTPUT_TOKENS = 600


class BudgetExceeded(Exception):
    """예약하면 비용 상한을 넘는 경우 발생합니다. (LLM 은 호출되지 않음)"""


@dataclass
class Budget:
    limit_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    exceeded: bool = False

    def reserve(self, amount: float) -> None:
        if self.spent_usd + self.reserved_usd + amount > self.limit_usd:
            self.exceeded = True
            raise BudgetExceeded(
                f"비용 상한 ${self.limit_usd} 도달 "
                f"(사용 ${self.spent_usd:.5f} + 진행 중 예약 ${self.reserved_usd:.5f} + 이번 호출 예상 ${amount:.5f})"
            )
        self.reserved_usd += amount

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved_usd -= reserved
        self.spent_usd += actual


def get_budget() -> Budget | None:
    """현재 실행의 Budget 을 돌려줍니다. 그래프 밖이거나 상한이 없으면 None."""
    try:
        return get_config().get("configurable", {}).get("budget")
    except Exception:
        return None


def budget_exceeded() -> bool:
    budget = get_budget()
    return budget is not None and budget.exceeded


def _reserve(messages: list[BaseMessage], max_tokens: int) -> float:
    budget = get_budget()
    if budget is None:
        return 0.0
    try:
        input_tokens = get_llm().get_num_tokens_from_messages(messages)
    except Exception:
        input_tokens = sum(len(str(m.content)) for m in messages)   # 토큰 계산 실패 시 글자 수로 보수적으로 추정
    amount = estimate_cost(settings.llm_model, input_tokens, max_tokens)
    budget.reserve(amount)
    return amount


def _settle(reserved: float, actual: float) -> None:
    budget = get_budget()
    if budget is not None:
        budget.settle(reserved, actual)


async def ainvoke_tracked(messages: list[BaseMessage], node: str) -> str:
    """LLM을 한 번 호출하고 응답 텍스트를 돌려줍니다. (사용량 이벤트 자동 발생, 비용 상한 확인)"""
    max_tokens = MAX_OUTPUT_TOKENS.get(node, DEFAULT_MAX_OUTPUT_TOKENS)
    reserved = _reserve(messages, max_tokens)
    cost = 0.0
    try:
        response = await get_llm().bind(max_tokens=max_tokens).ainvoke(messages)
        cost = _emit_usage(node, response)
    finally:
        _settle(reserved, cost)
    return response.content


async def astream_tracked(messages: list[BaseMessage], node: str) -> str:
    """
    LLM 응답을 조각(token) 단위로 Frontend에 흘려보내면서 전체 텍스트를 돌려줍니다.
    조각들을 더해(+) 하나의 메시지로 합치면 마지막에 사용량 정보가 들어 있습니다.
    """
    max_tokens = MAX_OUTPUT_TOKENS.get(node, DEFAULT_MAX_OUTPUT_TOKENS)
    reserved = _reserve(messages, max_tokens)
    cost = 0.0
    full = None
    try:
        async for chunk in get_llm().bind(max_tokens=max_tokens).astream(messages):
            if chunk.content:
                emit(Token(node=node, text=chunk.content))
            full = chunk if full is None else full + chunk
        if full is not None:
            cost = _emit_usage(node, full)
    finally:
        _settle(reserved, cost)
    return "" if full is None else full.content
