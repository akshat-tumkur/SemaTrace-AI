"""Migrated V2 mock evidence-contract regressions; no accuracy claim."""

import copy


import hashlib


import json


from pathlib import Path


from unittest.mock import MagicMock


import pytest


from pydantic import ValidationError


from app.compat.dual_assessment import (
    DualAssessment,
    exact_range,
    grounded_locations,
    DUAL_SYSTEM_PROMPT,
)


from app.compat.evidence import annotate_evidence


from app.retrieval.bm25 import LocalCorpus


from app.storage.database import AnalysisStore


def pair(sub, src, **extra):
    return dict(
        submission_span=sub,
        source_span=src,
        submission_occurrence=0,
        source_occurrence=0,
        **extra,
    )


def assessment(sub, src, reuse="YES", category="MINOR_EDIT", support="NONE"):
    details = []
    spans = []
    if support == "FULL":
        spans = [pair(sub, src)]
    elif support == "PARTIAL":
        spans = [pair(sub.split(";")[0], src)]
        details = [
            dict(
                submission_span=sub.split("; ")[1],
                submission_occurrence=0,
                source_span=None,
                source_occurrence=None,
                relationship="NOT_STATED",
                explanation="The additional clause is absent, not contradicted.",
            )
        ]
    elif reuse == "YES" and support == "NONE":
        details = [
            dict(
                submission_span=sub,
                submission_occurrence=0,
                source_span=src,
                source_occurrence=0,
                relationship="CHANGED",
                explanation="The details differ; this does not independently prove a logical contradiction.",
            )
        ]
    return dict(
        textual_reuse=dict(
            verdict=reuse,
            category=category,
            confidence="HIGH",
            evidence=[pair(sub.split(";")[0] if category == "PARTIAL" else sub, src)]
            if reuse == "YES"
            else [],
            explanation="A synthetic response for contract testing only.",
        ),
        semantic_support=dict(
            verdict=support,
            confidence="HIGH",
            supporting_spans=spans,
            unsupported_or_changed_details=details,
            explanation="Support is independent of reuse.",
        ),
        limitations=[],
    )




def test_explicit_repeated_unicode_occurrences_roundtrip_without_normalization():
    sub, src = "🌍 café; café.", "👩‍🔬 café; café."
    output = assessment(sub, src, category="PARTIAL", support="UNCERTAIN")
    output["textual_reuse"]["evidence"] = [
        dict(pair("café", "café"), submission_occurrence=1, source_occurrence=1)
    ]
    result = DualAssessment.model_validate(
        output, context=dict(submission=sub, source=src)
    )
    locations = grounded_locations(result, sub, src)
    for side, text in (("submission", sub), ("source", src)):
        a, b = locations["reuse"][0][side]
        assert text[a:b] == "café" and a == text.rfind("café")
    assert exact_range("aaa", "aa", 1) == (1, 3)
    with pytest.raises(ValueError):
        exact_range(sub, "cafe\u0301", 0)


@pytest.mark.parametrize(
    "failure",
    [
        "overlap",
        "missing_qualifier",
        "full_changed",
        "missing_details",
        "source_absence",
        "none_quotes",
    ],
)
def test_support_cannot_claim_unsupported_clauses_or_missing_qualifiers(failure):
    sub, src = (
        "Alice may send data; the server always responds instantly.",
        "Alice may send data.",
    )
    output = assessment(sub, src, category="PARTIAL", support="PARTIAL")
    s = output["semantic_support"]
    if failure == "overlap":
        s["supporting_spans"] = [pair(sub, src)]
    if failure == "missing_qualifier":
        s.update(
            verdict="FULL",
            supporting_spans=[pair("send data", "send data")],
            unsupported_or_changed_details=[],
        )
    if failure == "full_changed":
        s["verdict"] = "FULL"
    if failure == "missing_details":
        s["unsupported_or_changed_details"] = []
    if failure == "source_absence":
        s["unsupported_or_changed_details"][0]["relationship"] = "CHANGED"
    if failure == "none_quotes":
        s["verdict"] = "NONE"
    with pytest.raises(ValidationError):
        DualAssessment.model_validate(output, context=dict(submission=sub, source=src))


def test_literal_quotes_cannot_prove_actor_attachment():
    sub, src = (
        "Alice sends the approved results to Bob during the meeting.",
        "Bob sends the approved results to Alice during the meeting.",
    )
    output = assessment(sub, src, support="FULL")
    c = dict(
        submission_text=sub,
        source_text=src,
        dual_assessment=output,
        verification_provenance={},
        final_match_status="POTENTIAL_REUSE",
    )
    # Deliberately wrong model judgment, with correct literal strings. Mechanical
    # checks cannot prove semantic attachment; no code may promote it to verified.
    annotate_evidence(c)
    assert c["dual_assessment_locations"]["semantic_attachment_validated"] is False
    assert c["evidence_assessment"]["semantic_support"]["basis"] == "MODEL_REPORTED"
    assert (
        c["evidence_assessment"]["semantic_support"]["independently_validated"] is False
    )
    assert c["evidence_assessment"]["coverage_basis"] == "EXCLUDED_FROM_LEGACY_METRICS"
