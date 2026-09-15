from __future__ import annotations

from app.agents.state import InvestigationState
from app.retrieval.merger import merge_candidates
from app.config import TAVILY_MAX_UNITS
from app.retrieval.queries import query_variants


def investigate(state: InvestigationState) -> InvestigationState:
    lexical_candidates: list[dict] = []
    dense_candidates: list[dict] = []
    web_candidates: list[dict] = []
    if state.embedding_index is None:
        from app.retrieval.embeddings import EmbeddingIndex

        state.embedding_index = EmbeddingIndex(state.corpus)
    for unit in state.units:
        for match in state.corpus.search(unit.text, limit=2):
            lexical_candidates.append(
                {
                    "unit_id": unit.id,
                    "submission_text": unit.text,
                    "source_id": match.source_id,
                    "source_text": match.source_text,
                    "source_title": match.title,
                    "source_url": match.url,
                    "lexical_score": match.score,
                }
            )
        for match in state.embedding_index.search(unit.text, limit=2):
            dense_candidates.append(
                {
                    "unit_id": unit.id,
                    "submission_text": unit.text,
                    "source_id": match.source_id,
                    "source_text": match.source_text,
                    "source_title": match.title,
                    "source_url": match.url,
                    "semantic_score": match.semantic_score,
                }
            )
    if state.web_search is not None:
        web_units = sorted(state.units, key=lambda unit: len(unit.text.split()), reverse=True)[:TAVILY_MAX_UNITS]
        query_count = 0
        for unit in web_units:
            for query in query_variants(unit.text):
                query_count += 1
                for result in state.web_search.search(query):
                    web_candidates.append(
                        {
                            "unit_id": unit.id,
                            "submission_text": unit.text,
                            "source_id": result.source_id,
                            "source_text": result.source_text,
                            "source_title": result.title,
                            "source_url": result.url,
                            "web_score": result.web_score,
                        }
                    )
        if state.web_search.last_error:
            state.event("Web Investigator Agent", "search_failed", "Tavily unavailable; continuing with local retrieval")
        else:
            state.event("Web Investigator Agent", "searched_web", f"Tavily searched {query_count} bounded query variants and returned {len(web_candidates)} candidate passages")
    else:
        state.event("Web Investigator Agent", "skipped", "Web search is not configured")
    state.candidates = merge_candidates(lexical_candidates, dense_candidates + web_candidates)
    state.event("Source Investigator Agent", "searched_local_corpus", f"BM25 returned {len(lexical_candidates)} candidates")
    state.event("Embedding Retriever", "searched_dense_index", f"Dense provider: {state.embedding_index.provider}; returned {len(dense_candidates)} candidates")
    state.event("Candidate Merger", "merged_retrieval_signals", f"Kept {len(state.candidates)} ranked candidates")
    return state
