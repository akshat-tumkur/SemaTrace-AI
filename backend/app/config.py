from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_PATH = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_PATH / ".env")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_ENABLED = env_flag("TAVILY_SEARCH_ENABLED", default=True)
TAVILY_MAX_UNITS = max(0, int(os.getenv("TAVILY_MAX_UNITS", "3")))
TAVILY_MAX_RESULTS = max(1, min(5, int(os.getenv("TAVILY_MAX_RESULTS", "3"))))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
