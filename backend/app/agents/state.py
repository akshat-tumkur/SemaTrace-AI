from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.parsing.sentences import SentenceUnit
from app.retrieval.bm25 import LocalCorpus
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.tavily import TavilySearch
from app.agents.reasoner import OpenAIReasoner


@dataclass
class InvestigationState:
    filename: str
    file_type: str
    text: str
    corpus: LocalCorpus
    embedding_index: EmbeddingIndex | None = None
    web_search: TavilySearch | None = None
    reasoner: OpenAIReasoner | None = None
    units: list[SentenceUnit] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    verified_matches: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    citation_matches: list[dict[str, Any]] = field(default_factory=list)
    audit_log: list[dict[str, str]] = field(default_factory=list)
    final_verdict: dict[str, Any] = field(default_factory=dict)

    def event(self, agent: str, event: str, details: str, unit_id: str | None = None) -> None:
        record = {"agent": agent, "event": event, "details": details}
        if unit_id:
            record["unit_id"] = unit_id
        self.audit_log.append(record)
