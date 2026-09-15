from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_SEARCH_ENABLED

TAVILY_ENDPOINT = "https://api.tavily.com/search"


@dataclass(frozen=True)
class WebResult:
    source_id: str
    title: str
    url: str
    source_text: str
    web_score: float


class TavilySearch:
    def __init__(self, api_key: str = TAVILY_API_KEY, enabled: bool = TAVILY_SEARCH_ENABLED) -> None:
        self.api_key = api_key
        self.enabled = enabled and bool(api_key)
        self.last_error: str | None = None

    def search(self, query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[WebResult]:
        if not self.enabled or not query.strip():
            return []
        payload = json.dumps(
            {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        request = Request(TAVILY_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=12) as response:
                results = json.loads(response.read().decode("utf-8")).get("results", [])
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            self.last_error = type(error).__name__
            return []
        return [
            WebResult(
                source_id=f"web-{index}-{abs(hash(result.get('url', ''))) % 1_000_000}",
                title=result.get("title", "Untitled web source"),
                url=result.get("url", ""),
                source_text=result.get("content", ""),
                web_score=float(result.get("score", 0.0) or 0.0),
            )
            for index, result in enumerate(results)
            if result.get("url") and result.get("content")
        ]
