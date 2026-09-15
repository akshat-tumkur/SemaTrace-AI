from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CorpusDocument:
    source_id: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class LexicalMatch:
    source_id: str
    title: str
    url: str
    source_text: str
    score: float


class LocalCorpus:
    def __init__(self, documents: list[CorpusDocument]) -> None:
        self.documents = documents
        self._tokens = [_tokenize(document.text) for document in documents]
        self._average_length = sum(map(len, self._tokens)) / max(len(self._tokens), 1)
        self._document_frequency: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

    def search(self, query: str, limit: int = 3) -> list[LexicalMatch]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scored = [(self._score(query_tokens, tokens), index) for index, tokens in enumerate(self._tokens)]
        scored.sort(reverse=True)
        return [
            LexicalMatch(
                source_id=self.documents[index].source_id,
                title=self.documents[index].title,
                url=self.documents[index].url,
                source_text=self.documents[index].text,
                score=round(score, 4),
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
            idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            score += idf * ((frequency * 2.0) / (frequency + 1.5 * (0.25 + 0.75 * length_ratio)))
        return score


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())
