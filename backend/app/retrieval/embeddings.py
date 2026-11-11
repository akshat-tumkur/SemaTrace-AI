from __future__ import annotations

import hashlib
import json
import math
import re
from http.client import HTTPException
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY
from app.retrieval.bm25 import LocalCorpus
from app.corpus.models import SourceOccurrence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
OPENAI_EMBEDDINGS_ENDPOINT = "https://api.openai.com/v1/embeddings"


@dataclass(frozen=True)
class DenseMatch:
    source_id: str
    title: str
    url: str
    source_text: str
    semantic_score: float
    source_document_id: str | None = None
    source_document_sha256: str | None = None
    source_start_char: int | None = None
    source_end_char: int | None = None
    passage_id: str | None = None
    source_occurrences: tuple[SourceOccurrence, ...] = ()
    generation_id: str | None = None
    vector_space_id: str | None = None


class EmbeddingIndex:
    """Dense retrieval backed by OpenAI embeddings with a local fallback."""

    def __init__(
        self,
        corpus: LocalCorpus,
        dimensions: int = 256,
        api_key: str = OPENAI_API_KEY,
        model: str = EMBEDDING_MODEL,
    ) -> None:
        if (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions <= 0
        ):
            raise ValueError("hash_embedding_dimensions")
        self.corpus = corpus
        self.dimensions = dimensions
        self.api_key = api_key
        self.model = model or "text-embedding-3-small"
        self.provider = "openai"
        self.last_error: str | None = None
        try:
            vectors = self._openai_embeddings(
                [document.text for document in corpus.documents]
            )
            self._validate_vectors(vectors, len(corpus.documents))
            self._vectors = vectors
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            HTTPException,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            self._use_hash_fallback(error)

    def search(self, query: str, limit: int = 3) -> list[DenseMatch]:
        if not self._vectors:
            return []
        if self.provider == "deterministic-hash-fallback":
            # Fallback is sticky for this index. Recovery requires a NEW complete index.
            query_vector = self._hash_embedding(query)
        else:
            try:
                vectors = self._query_embeddings([query])
                self._validate_vectors(vectors, 1, len(self._vectors[0]))
                query_vector = vectors[0]
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                HTTPException,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as error:
                self._use_hash_fallback(error)
                query_vector = self._hash_embedding(query)
        ranked = sorted(
            (
                (self._cosine(query_vector, vector), index)
                for index, vector in enumerate(self._vectors)
            ),
            reverse=True,
        )
        return [
            DenseMatch(
                source_id=self.corpus.documents[index].source_id,
                title=self.corpus.documents[index].title,
                url=self.corpus.documents[index].url,
                source_text=self.corpus.documents[index].text,
                semantic_score=round(score, 4),
                source_document_id=self.corpus.documents[index].source_document_id,
                source_document_sha256=self.corpus.documents[
                    index
                ].source_document_sha256,
                source_start_char=self.corpus.documents[index].source_start_char,
                source_end_char=self.corpus.documents[index].source_end_char,
                passage_id=self.corpus.documents[index].passage_id,
                source_occurrences=self.corpus.documents[index].source_occurrences,
            )
            for score, index in ranked[:limit]
            if score > 0
        ]

    def _query_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._openai_embeddings(texts)

    def _openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        body = json.dumps(
            {"model": self.model, "input": texts, "encoding_format": "float"}
        ).encode("utf-8")
        request = Request(
            OPENAI_EMBEDDINGS_ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.last_response_usage = payload.get("usage")
        if payload.get("model", self.model) != self.model:
            raise ValueError("embedding_response_model")
        data = payload["data"]
        if any(
            isinstance(item["index"], bool) or not isinstance(item["index"], int)
            for item in data
        ):
            raise ValueError("embedding_response_indices")
        data = sorted(data, key=lambda item: item["index"])
        if [item["index"] for item in data] != list(range(len(texts))):
            raise ValueError("embedding_response_indices")
        if any(
            not isinstance(item["embedding"], list)
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in item["embedding"]
            )
            for item in data
        ):
            raise ValueError("embedding_response_coordinates")
        return [list(map(float, item["embedding"])) for item in data]

    @staticmethod
    def _validate_vectors(
        vectors: list[list[float]], count: int, dimensions: int | None = None
    ) -> None:
        if len(vectors) != count:
            raise ValueError("embedding_vector_count")
        if not count:
            return
        expected = dimensions if dimensions is not None else len(vectors[0])
        if expected <= 0 or any(len(vector) != expected for vector in vectors):
            raise ValueError("embedding_vector_shape")
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("embedding_nonfinite_coordinate")

    def _hash_embedding(self, text: str) -> list[float]:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)

    def _use_hash_fallback(self, error: Exception) -> None:
        # Build before publishing any state; an index is never partly converted.
        vectors = [
            self._hash_embedding(document.text) for document in self.corpus.documents
        ]
        self._validate_vectors(vectors, len(self.corpus.documents), self.dimensions)
        self._vectors = vectors
        self.provider = "deterministic-hash-fallback"
        self.last_error = _safe_error(error)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            raise ValueError("embedding_incompatible_dimensions")
        if any(
            not math.isfinite(value) for vector in (left, right) for value in vector
        ):
            raise ValueError("embedding_nonfinite_coordinate")
        return max(0.0, sum(a * b for a, b in zip(left, right)))


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    return [value / magnitude for value in vector] if magnitude else vector


def _safe_error(error: Exception) -> str:
    if isinstance(error, HTTPError):
        return f"HTTP {error.code}"
    return type(error).__name__
