from __future__ import annotations

from app.agents.state import InvestigationState

_GENERIC_PREFIXES = ("in conclusion", "according to the study", "this paper", "the results show")


def filter_false_positives(state: InvestigationState) -> InvestigationState:
    kept: list[dict] = []
    for match in state.verified_matches:
        text = match["submission_text"].lower()
        if len(text.split()) < 8 or any(text.startswith(prefix) for prefix in _GENERIC_PREFIXES):
            match["match_type"] = "LIKELY_FALSE_POSITIVE"
            state.false_positives.append(match)
        else:
            kept.append(match)
    state.verified_matches = kept
    state.event("False Positive Filter Agent", "filtered_matches", f"Downgraded {len(state.false_positives)} generic or short matches")
    return state
