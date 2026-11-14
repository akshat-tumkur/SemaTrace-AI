"""Additive evidence annotations. Never decides retention, coverage, citations, or risk."""

from __future__ import annotations

import re
from typing import Any

from app.compat.dual_assessment import DualAssessment, grounded_locations


def literal_locations(text: str, quote: str) -> dict[str, Any]:
    """Locate exact occurrences without choosing which repeated occurrence was intended."""
    starts = (
        [m.start() for m in re.finditer(f"(?={re.escape(quote)})", text)]
        if quote.strip()
        else []
    )
    return {
        "ranges": [[start, start + len(quote)] for start in starts[:16]],
        "occurrence_count": len(starts),
        "truncated": len(starts) > 16,
    }


def copy_error_is_retriable(spans: list[dict], submission: str, source: str) -> bool:
    """Admit copy diagnostics only. This never accepts, normalizes, or returns evidence."""
    words = lambda text: re.findall(r"\w+", text.casefold())
    invalid = []
    for pair in spans:
        for key, target in (("submission_span", submission), ("source_span", source)):
            quote = pair[key]
            if quote not in target:
                needle, haystack = words(quote), words(target)
                invalid.append(
                    bool(needle)
                    and any(
                        haystack[i : i + len(needle)] == needle
                        for i in range(len(haystack) - len(needle) + 1)
                    )
                )
    return bool(invalid) and all(invalid)


def annotate_evidence(candidate: dict[str, Any]) -> None:
    """Separate lexical retention from model conclusions and unavailable verification."""
    if candidate.get("dual_assessment") is not None:
        _annotate_dual(candidate)
        return
    # A rerun of deterministic classification clears llm_verdict; ignore stale results.
    result = (
        (candidate.get("llm_verification") or {})
        if candidate.get("llm_verdict")
        else {}
    )
    verdict, status = result.get("verdict"), candidate.get("final_match_status")
    locations = []
    for index, pair in enumerate(result.get("supporting_spans", [])):
        left = literal_locations(candidate["submission_text"], pair["submission_span"])
        right = literal_locations(candidate["source_text"], pair["source_span"])
        locations.append(
            {
                "span_index": index,
                "submission": left,
                "source": right,
                "ambiguous": left["occurrence_count"] > 1
                or right["occurrence_count"] > 1,
            }
        )
    invalid = any(
        not row[side]["occurrence_count"]
        for row in locations
        for side in ("submission", "source")
    )
    error = (candidate.get("verification_provenance") or {}).get("error")
    if not verdict:
        semantic = "UNAVAILABLE" if error else "NOT_EVALUATED"
    elif invalid:
        semantic = "INVALID_EVIDENCE"
    elif verdict == "INSUFFICIENT_CONTEXT" or result.get("confidence") == "LOW":
        semantic = "UNCERTAIN"
    elif verdict == "UNRELATED":
        semantic = (
            "CONTRADICTION_REPORTED"
            if result.get("contradictions")
            else "NO_SPECIFIC_SUPPORT_REPORTED"
        )
    elif status == "MANUAL_REVIEW":
        semantic = "CONFLICT_REQUIRES_REVIEW"
    elif verdict == "PARTIAL_OVERLAP":
        semantic = (
            "PARTIAL_SUPPORT_WITH_CONTRADICTIONS"
            if result.get("contradictions")
            else "MODEL_REPORTED_PARTIAL_SUPPORT"
        )
    else:
        semantic = "MODEL_REPORTED_EQUIVALENCE"
    if status == "FILTERED_FALSE_POSITIVE":
        semantic = "FILTERED_GENERIC_OR_SHORT"
    lexical = (
        "STRONG_LEXICAL_EVIDENCE"
        if candidate.get("deterministic_outcome") == "ACCEPT"
        else (
            "AMBIGUOUS_LEXICAL_EVIDENCE"
            if candidate.get("legacy_accepted")
            else "LIMITED_LEXICAL_EVIDENCE"
        )
    )
    candidate["evidence_assessment"] = {
        "lexical_status": lexical,
        "semantic_status": semantic,
        "manual_review_required": status == "MANUAL_REVIEW"
        or semantic
        in {
            "UNAVAILABLE",
            "UNCERTAIN",
            "INVALID_EVIDENCE",
            "CONFLICT_REQUIRES_REVIEW",
            "PARTIAL_SUPPORT_WITH_CONTRADICTIONS",
        },
        "citation_support_status": candidate.get(
            "citation_support_status", "NOT_ASSESSED"
        ),
        "offset_unit": "unicode_code_points",
        "end_exclusive": True,
        "supporting_span_locations": locations,
        "explanation": f"{lexical}; semantic assessment: {semantic}. Citation markers do not establish source attribution or claim support.",
        # Additive presentation contract. Neither token overlap nor literal model
        # quotes establish independently verified meaning or logical contradiction.
        "semantic_support": {
            "status": "UNKNOWN"
            if error
            or semantic
            in {
                "INVALID_EVIDENCE",
                "UNCERTAIN",
                "CONFLICT_REQUIRES_REVIEW",
                "CONTRADICTION_REPORTED",
                "PARTIAL_SUPPORT_WITH_CONTRADICTIONS",
                "NO_SPECIFIC_SUPPORT_REPORTED",
            }
            else (
                "PARTIAL"
                if semantic == "MODEL_REPORTED_PARTIAL_SUPPORT"
                else "UNVERIFIED"
            ),
            "basis": "MODEL_REPORTED" if verdict else "NOT_EVALUATED",
            "independently_validated": False,
        },
    }


def _annotate_dual(candidate: dict[str, Any]) -> None:
    """Annotations are model reports, never proof of semantic attachment or authorship."""
    from app.compat.effective_support import effective_support

    effective = effective_support(
        candidate["dual_assessment"],
        candidate["submission_text"],
        candidate["source_text"],
        candidate.get("deterministic_conflicts", ()),
    )
    candidate["effective_semantic_support"] = effective
    candidate["explanation"] = (
        f"Effective semantic support: {effective['status']} (not independently validated). Raw model explanations remain in dual_assessment."
    )
    try:
        result = DualAssessment.model_validate(
            candidate["dual_assessment"],
            context={
                "submission": candidate["submission_text"],
                "source": candidate["source_text"],
            },
        )
        locations = grounded_locations(
            result, candidate["submission_text"], candidate["source_text"]
        )
    except ValueError:
        candidate.pop("dual_assessment_locations", None)
        candidate["evidence_assessment"] = {
            "textual_reuse": {"status": "UNKNOWN", "basis": "INVALID_EVIDENCE"},
            "semantic_support": {
                "status": "UNKNOWN",
                "basis": "INVALID_EVIDENCE",
                "independently_validated": False,
            },
            "lexical_status": "UNCLASSIFIED",
            "semantic_status": "INVALID_EVIDENCE",
            "manual_review_required": True,
        }
        return
    reuse, support = result.textual_reuse, result.semantic_support
    support_pending = effective["status"] == "UNKNOWN"
    pending = (
        support_pending or reuse.confidence == "LOW" or reuse.verdict == "UNCERTAIN"
    )
    candidate["dual_assessment_locations"] = {
        **locations,
        "offset_unit": "unicode_code_points",
        "end_exclusive": True,
        "basis": "ORIGINAL_PASSAGES",
        "semantic_attachment_validated": False,
    }
    # Map every selected quote through the pinned passage occurrence. Never
    # substitute an arbitrary source alias or choose the first repeated quote.
    origin_unit = (candidate.get("verification_provenance") or {}).get(
        "submission_unit"
    )
    origins = candidate.get("source_occurrences") or [candidate]
    for rows in (locations["reuse"], locations["support"], locations["unsupported"]):
        for row in rows:
            row["submission_document"] = (
                [origin_unit["start"] + n for n in row["submission"]]
                if origin_unit
                else None
            )
            row["source_documents"] = []
            for origin in origins:
                start, end = (
                    origin.get("source_start_char"),
                    origin.get("source_end_char"),
                )
                if (
                    row["source"] is not None
                    and type(start) is int
                    and type(end) is int
                    and start >= 0
                    and end - start == len(candidate["source_text"])
                    and origin.get("source_document_id")
                ):
                    row["source_documents"].append(
                        {
                            "source_document_id": origin["source_document_id"],
                            "occurrence_id": origin.get("occurrence_id"),
                            "range": [start + n for n in row["source"]],
                        }
                    )
    candidate["evidence_assessment"] = {
        "lexical_status": "STRONG_LEXICAL_EVIDENCE"
        if candidate.get("deterministic_outcome") == "ACCEPT"
        else "AMBIGUOUS_OR_LIMITED_LEXICAL_EVIDENCE",
        "semantic_status": "UNKNOWN"
        if support_pending
        else "MODEL_REPORTED_" + support.verdict,
        "textual_reuse": {
            "status": "UNKNOWN"
            if reuse.confidence == "LOW" or reuse.verdict == "UNCERTAIN"
            else reuse.verdict,
            "category": reuse.category,
            "basis": "MODEL_REPORTED",
            "independently_validated": False,
        },
        "semantic_support": {
            "status": "UNKNOWN" if support_pending else support.verdict,
            "basis": "MODEL_REPORTED",
            "independently_validated": False,
        },
        "manual_review_required": pending,
        "citation_support_status": candidate.get(
            "citation_support_status", "NOT_ASSESSED"
        ),
        "offset_unit": "unicode_code_points",
        "end_exclusive": True,
        "coverage_basis": "EXCLUDED_FROM_LEGACY_METRICS",
        "explanation": "Possible source reuse and semantic support are separate model judgments. Literal offsets do not validate meaning, authorship or citation support.",
    }


def refresh_report_support(report):
    """Read-time safety projection for older dual records; no database rewrite."""
    for candidate in [
        *report.get("matches", []),
        *report.get("verification", {}).get("candidates", []),
    ]:
        if candidate.get("dual_assessment") is not None:
            annotate_evidence(candidate)
    return report
