"""
설정 모듈
- 프로젝트 루트의 .env 환경 변수와 config/items.yaml 의 아이템 설정을 한곳에서 읽어옵니다.
- 다른 모듈은 이 파일의 `settings`, `load_items()` 만 사용합니다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent   # backend/
ROOT_DIR = BASE_DIR.parent                          # 프로젝트 루트 (.env 위치)
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    recursion_limit: int = int(os.getenv("RECURSION_LIMIT", "25"))
    max_eval_attempts: int = int(os.getenv("MAX_EVAL_ATTEMPTS", "3"))    # 보고서 평가 최대 횟수 (최초 작성 + 재작성)
    max_search_attempts: int = int(os.getenv("MAX_SEARCH_ATTEMPTS", "2"))  # 근거 검색 최대 횟수 (최초 검색 + 재검색)
    max_analysis_attempts: int = int(os.getenv("MAX_ANALYSIS_ATTEMPTS", "2"))  # 기업별 분석 최대 횟수 (최초 분석 + 재분석)
    cost_limit_usd: float = float(os.getenv("COST_LIMIT_USD", "0.015"))     # 실행 1회의 LLM 비용 상한 (USD)
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
    poor_output_test: bool = os.getenv("POOR_OUTPUT_TEST", "0") == "1"   # 화면에서 모드를 보내지 않을 때의 기본값

    data_dir: Path = BASE_DIR / "data"
    index_dir: Path = BASE_DIR / "index"
    items_file: Path = BASE_DIR / "config" / "items.yaml"


settings = Settings()

# ──────────────────────────────────────────────────────────────
# LLM 단가표 (USD / 토큰 100만 개 기준: 입력, 출력)
#  - 사용량 이벤트(llm_usage)에서 호출별 예상 비용을 계산할 때 사용합니다.
#  - 단가는 바뀔 수 있으므로 사용하는 모델의 최신 단가로 수정하세요.
#  - 표에 없는 모델은 비용이 0으로 표시됩니다.
# ──────────────────────────────────────────────────────────────
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


@lru_cache
def load_items() -> dict:
    """config/items.yaml 을 읽어 {item_id: {...}} 형태로 돌려줍니다."""
    with open(settings.items_file, encoding="utf-8") as f:
        return yaml.safe_load(f)["items"]


def get_item(item_id: str) -> dict:
    items = load_items()
    if item_id not in items:
        raise KeyError(f"알 수 없는 아이템입니다: {item_id}")
    return items[item_id]
