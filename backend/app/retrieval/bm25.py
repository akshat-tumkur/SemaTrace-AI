from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping
from app.corpus.models import SourceOccurrence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CorpusDocument:
    source_id: str
    title: str
    url: str
    text: str
    source_document_id: str | None = None
    source_document_sha256: str | None = None
    source_start_char: int | None = None
    source_end_char: int | None = None
    passage_id: str | None = None
    source_occurrences: tuple[SourceOccurrence, ...] = ()


@dataclass(frozen=True)
class LexicalMatch:
    source_id: str
    title: str
    url: str
    source_text: str
    score: float
    source_document_id: str | None = None
    source_document_sha256: str | None = None
    source_start_char: int | None = None
    source_end_char: int | None = None
    passage_id: str | None = None
    source_occurrences: tuple[SourceOccurrence, ...] = ()


class LocalCorpus:
    def __init__(
        self,
        documents: list[CorpusDocument],
        source_documents: Mapping[str, str] | None = None,
        *,
        reuse_statistics: bool = False,
    ) -> None:
        self.documents = documents
        self.source_documents = MappingProxyType(dict(source_documents or {}))
        if reuse_statistics:
            self._tokens, self._average_length, self._document_frequency = (
                _cached_statistics(tuple(d.text for d in documents))
            )
            return
        self._tokens = [_tokenize(document.text) for document in documents]
        self._average_length = sum(map(len, self._tokens)) / max(len(self._tokens), 1)
        self._document_frequency: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self._document_frequency[token] = (
                    self._document_frequency.get(token, 0) + 1
                )

    def search(self, query: str, limit: int = 3) -> list[LexicalMatch]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        # Identical BM25 formula/tie order, using postings rather than scanning
        # every document's token list for every query term.
        postings = _postings(tuple(d.text for d in self.documents))
        scores = defaultdict(float)
        count = len(self.documents)
        for token in query_tokens:
            entries = postings.get(token, ())
            if not entries:
                continue
            idf = math.log(1 + (count - len(entries) + 0.5) / (len(entries) + 0.5))
            for index, frequency in entries:
                ratio = len(self._tokens[index]) / max(self._average_length, 1)
                scores[index] += idf * (
                    (frequency * 2.0) / (frequency + 1.5 * (0.25 + 0.75 * ratio))
                )
        scored = sorted(
            ((score, index) for index, score in scores.items()), reverse=True
        )
        return [
            LexicalMatch(
                source_id=self.documents[index].source_id,
                title=self.documents[index].title,
                url=self.documents[index].url,
                source_text=self.documents[index].text,
                score=round(score, 4),
                source_document_id=self.documents[index].source_document_id,
                source_document_sha256=self.documents[index].source_document_sha256,
                source_start_char=self.documents[index].source_start_char,
                source_end_char=self.documents[index].source_end_char,
                passage_id=self.documents[index].passage_id,
                source_occurrences=self.documents[index].source_occurrences,
            )
            for score, index in scored[:limit]
            if score > 0
        ]

    def _score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        if not document_tokens:
            return 0.0
        document_count = len(self.documents)
        length_ratio = len(document_tokens) / max(self._average_length, 1)
        score = 0.0
        for token in query_tokens:
            frequency = document_tokens.count(token)
            if not frequency:
                continue
            document_frequency = self._document_frequency.get(token, 0)
            idf = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += idf * (
                (frequency * 2.0) / (frequency + 1.5 * (0.25 + 0.75 * length_ratio))
            )
        return score


def _tokenize(text: str) -> list[str]:
    return [sys.intern(t) for t in _TOKEN_PATTERN.findall(text.lower())]


@lru_cache(maxsize=1)
def _postings(texts):
    result = defaultdict(list)
    for i, text in enumerate(texts):
        for token, frequency in Counter(_tokenize(text)).items():
            result[token].append((i, frequency))
    return dict(result)


@lru_cache(maxsize=1)
def _cached_statistics(texts: tuple[str, ...]):
    """Reuse only immutable lexical statistics, never unchecked source snapshots."""
    tokens = tuple(tuple(_tokenize(text)) for text in texts)
    frequency: dict[str, int] = {}
    for passage in tokens:
        for token in set(passage):
            frequency[token] = frequency.get(token, 0) + 1
    return (
        tokens,
        sum(map(len, tokens)) / max(len(tokens), 1),
        MappingProxyType(frequency),
    )
