from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from app.retrieval.bm25 import CorpusDocument, LocalCorpus

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class DenseMatch:
    source_id: str
    title: str
    url: str
    source_text: str
    semantic_score: float


class EmbeddingIndex:
    """Dense retrieval with an optional sentence-transformers provider.

    The deterministic hashing fallback keeps the MVP runnable without downloading
    a model; it is a retrieval signal, not a semantic plagiarism verdict.
    """

    def __init__(self, corpus: LocalCorpus, dimensions: int = 256) -> None:
        self.corpus = corpus
        self.dimensions = dimensions
        self.provider = "deterministic-hash"
        self._model: Any | None = None
        self._vectors = [self._embed(document.text) for document in corpus.documents]

    def search(self, query: str, limit: int = 3) -> list[DenseMatch]:
        query_vector = self._embed(query)
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

    def _embed(self, text: str) -> list[float]:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return max(0.0, sum(a * b for a, b in zip(left, right)))


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    return [value / magnitude for value in vector] if magnitude else vector
