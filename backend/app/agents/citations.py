from __future__ import annotations

import re

from app.agents.state import InvestigationState

_CITATION_PATTERN = re.compile(r"(?:\([^()]{2,40},?\s*\d{4}\)|\[\d+\])")


def check_citations(state: InvestigationState) -> InvestigationState:
    for match in state.verified_matches:
        if _CITATION_PATTERN.search(match["submission_text"]):
            match["citation_present"] = True
            state.citation_matches.append(match)
        else:
            match["citation_present"] = False
    state.event("Citation Checker Agent", "checked_attribution", f"Found {len(state.citation_matches)} cited evidence matches")
    return state
