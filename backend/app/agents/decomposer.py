from __future__ import annotations

from app.agents.state import InvestigationState
from app.parsing.sentences import split_into_units


def decompose(state: InvestigationState) -> InvestigationState:
    state.units = split_into_units(state.text)
    state.event("Decomposer Agent", "extracted_units", f"Identified {len(state.units)} sentence units")
    return state
