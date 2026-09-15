from __future__ import annotations

import re

from app.agents.state import InvestigationState

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "without",
}


def verify_candidates(state: InvestigationState) -> InvestigationState:
    for candidate in state.candidates:
        submission_tokens = _content_tokens(candidate["submission_text"])
        source_tokens = _content_tokens(candidate["source_text"])
        overlap = len(submission_tokens & source_tokens) / max(len(submission_tokens), 1)
        lexical_score = candidate.get("lexical_score", 0.0)
        verification_score = round(min(1.0, overlap * 0.7 + min(lexical_score / 2.0, 1.0) * 0.3), 4)
        candidate["token_overlap"] = round(overlap, 4)
        candidate["verification_score"] = verification_score
        if verification_score >= 0.35 and overlap >= 0.35:
            candidate["match_type"] = "EXACT_OR_LEXICAL_OVERLAP" if overlap >= 0.65 else "POSSIBLE_PARAPHRASE"
            state.verified_matches.append(candidate)
    state.event("Evidence Verifier Agent", "verified_candidates", f"Confirmed {len(state.verified_matches)} evidence candidates")
    return state


def _content_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOPWORDS}
