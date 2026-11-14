"""One-way semantic quarantine; never decides reuse, retention or scoring."""

from app.compat.dual_assessment import DualAssessment


def effective_support(raw, submission, source, conflicts=()):
    result = dict(
        status="UNKNOWN",
        basis="UNAVAILABLE_OR_INVALID",
        independently_validated=False,
        reason_codes=[],
        raw_verdict=None,
        raw_verdict_reference="dual_assessment.semantic_support.verdict",
        semantic_support_coverage_permitted=False,
        literal_evidence_validated=False,
    )
    if isinstance(raw, dict):
        support_raw = raw.get("semantic_support")
        result["raw_verdict"] = (
            support_raw.get("verdict") if isinstance(support_raw, dict) else None
        )
    try:
        assessment = DualAssessment.model_validate(
            raw, context=dict(submission=submission, source=source)
        )
    except (ValueError, TypeError):
        result["reason_codes"] = ["INVALID_DUAL_ASSESSMENT"]
        return result
    from app.compat.legacy_conflicts import _detail_conflicts

    warnings = set(_detail_conflicts(submission, source)) | set(conflicts or ())
    support = assessment.semantic_support
    result.update(
        basis="MODEL_REPORTED_LITERAL_GROUNDED", literal_evidence_validated=True
    )
    reasons = []
    if support.confidence == "LOW" or support.verdict == "UNCERTAIN":
        reasons.append("MODEL_UNCERTAIN")
    if support.verdict in {"FULL", "PARTIAL"} and warnings:
        reasons.extend(
            {
                "different_numbers": "NUMERIC_DETAIL_MISMATCH",
                "different_negation": "POLARITY_WARNING",
            }.get(w, "DETERMINISTIC_CONFLICT_WARNING")
            for w in sorted(warnings)
        )
    if reasons:
        result.update(
            basis="QUARANTINED_MODEL_REPORT", reason_codes=sorted(set(reasons))
        )
    else:
        result["status"] = support.verdict
    # FULL means model-reported only, never independently VERIFIED; no path
    # authorizes semantic coverage, including identical or equal-number passages.
    return result
