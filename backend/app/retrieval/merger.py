from __future__ import annotations

from typing import Any


def merge_candidates(lexical: list[dict[str, Any]], dense: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in lexical + dense:
        key = (candidate["unit_id"], candidate["source_id"])
        current = merged.setdefault(key, candidate.copy())
        current.update({key: value for key, value in candidate.items() if key not in {"lexical_score", "semantic_score"}})
        if "lexical_score" in candidate:
            current["lexical_score"] = max(current.get("lexical_score", 0.0), candidate["lexical_score"])
        if "semantic_score" in candidate:
            current["semantic_score"] = max(current.get("semantic_score", 0.0), candidate["semantic_score"])
        if "web_score" in candidate:
            current["web_score"] = max(current.get("web_score", 0.0), candidate["web_score"])
    for candidate in merged.values():
        lexical_score = min(candidate.get("lexical_score", 0.0) / 2.0, 1.0)
        semantic_score = candidate.get("semantic_score", 0.0)
        web_score = candidate.get("web_score", 0.0)
        candidate["retrieval_score"] = round(0.45 * lexical_score + 0.35 * semantic_score + 0.2 * web_score, 4)
    return sorted(merged.values(), key=lambda candidate: candidate["retrieval_score"], reverse=True)[:limit]
