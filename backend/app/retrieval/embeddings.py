from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY
from app.retrieval.bm25 import CorpusDocument, LocalCorpus

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
OPENAI_EMBEDDINGS_ENDPOINT = "https://api.openai.com/v1/embeddings"


@dataclass(frozen=True)
class DenseMatch:
    source_id: str
    title: str
    url: str
    source_text: str
    semantic_score: float


class EmbeddingIndex:
    """Dense retrieval backed by OpenAI embeddings with a local fallback."""

    def __init__(self, corpus: LocalCorpus, dimensions: int = 256, api_key: str = OPENAI_API_KEY, model: str = EMBEDDING_MODEL) -> None:
        self.corpus = corpus
        self.dimensions = dimensions
        self.api_key = api_key
        self.model = model or "text-embedding-3-small"
        self.provider = "openai"
        self.last_error: str | None = None
        try:
            self._vectors = self._openai_embeddings([document.text for document in corpus.documents])
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as error:
            self._use_hash_fallback(error)

    def search(self, query: str, limit: int = 3) -> list[DenseMatch]:
        try:
            query_vector = self._openai_embeddings([query])[0]
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as error:
            self._use_hash_fallback(error)
            query_vector = self._hash_embedding(query)
        ranked = sorted(
            ((self._cosine(query_vector, vector), index) for index, vector in enumerate(self._vectors)),
            reverse=True,
        )
        return [
            DenseMatch(
                source_id=self.corpus.documents[index].source_id,
                title=self.corpus.documents[index].title,
                url=self.corpus.documents[index].url,
                source_text=self.corpus.documents[index].text,
                semantic_score=round(score, 4),
            )
            for score, index in ranked[:limit]
            if score > 0
        ]

    def _openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        body = json.dumps({"model": self.model, "input": texts, "encoding_format": "float"}).encode("utf-8")
        request = Request(OPENAI_EMBEDDINGS_ENDPOINT, data=body, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        data = sorted(payload["data"], key=lambda item: item["index"])
        return [list(map(float, item["embedding"])) for item in data]

    def _hash_embedding(self, text: str) -> list[float]:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)

    def _use_hash_fallback(self, error: Exception) -> None:
        self.provider = "deterministic-hash-fallback"
        self.last_error = _safe_error(error)
        self._vectors = [self._hash_embedding(document.text) for document in self.corpus.documents]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return max(0.0, sum(a * b for a, b in zip(left, right)))


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    return [value / magnitude for value in vector] if magnitude else vector


def _safe_error(error: Exception) -> str:
    if isinstance(error, HTTPError):
        return f"HTTP {error.code}"
    return type(error).__name__
