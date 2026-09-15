from __future__ import annotations

from typing import Any

from app.agents.reasoner import OpenAIReasoner
from app.graph.workflow import build_workflow
from app.agents.state import InvestigationState
from app.agents.verifier import verify_candidates
from app.retrieval.bm25 import LocalCorpus
from app.retrieval.tavily import TavilySearch


def run_investigation(
    filename: str,
    file_type: str,
    text: str,
    corpus: LocalCorpus,
    web_search: TavilySearch | None = None,
    reasoner: OpenAIReasoner | None = None,
) -> dict[str, Any]:
    state = InvestigationState(filename=filename, file_type=file_type, text=text, corpus=corpus, web_search=web_search, reasoner=reasoner)
    state.event("Workflow", "started", "Beginning agentic investigation")
    state = build_workflow().invoke({"investigation": state})["investigation"]
    return {
        **state.final_verdict,
        "filename": filename,
        "file_type": file_type,
        "status": "completed",
        "units_analyzed": len(state.units),
        "units": [unit.__dict__ for unit in state.units],
        "mode": "LangGraph agentic workflow",
        "audit_log": state.audit_log,
    }
