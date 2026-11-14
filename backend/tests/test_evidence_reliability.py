"""Production reliability fixes preserve decisions and strict literal grounding."""

import copy


import json


from unittest.mock import MagicMock


import pytest


from app.compat.evidence import annotate_evidence, literal_locations


from app.parsing.sentences import SentenceUnit


from app.retrieval.bm25 import LocalCorpus


SUB = "Alpha beta gamma delta epsilon zeta eta theta iota kappa ."


SRC = "Alpha beta gamma delta epsilon unrelated words about something else ."


def result(literal=True, **updates):
    value = dict(
        verdict="POSSIBLE_PARAPHRASE",
        confidence="HIGH",
        explanation="A model assessment for a copy diagnostic.",
        supporting_spans=[
            dict(
                submission_span=SUB,
                source_span=SRC if literal else SRC.replace(" .", "."),
            )
        ],
        contradictions=[],
        limitations=[],
    )
    value.update(updates)
    return value




def test_literal_locations_preserve_unicode_and_expose_repeated_quote_ambiguity():
    text = "α🙂 heat moves. α🙂 heat moves."
    quote = "α🙂 heat moves."
    found = literal_locations(text, quote)
    assert found["occurrence_count"] == 2
    assert all(text[start:end] == quote for start, end in found["ranges"])
    assert literal_locations(text, "Heat moves.")["occurrence_count"] == 0
    assert literal_locations("a" * 30, "a")["truncated"]


def test_assessment_is_additive_and_does_not_resolve_ambiguous_quote_locations():
    item = dict(
        submission_text=SUB + SUB,
        source_text=SRC,
        llm_verdict="PARTIAL_OVERLAP",
        llm_verification=result(verdict="PARTIAL_OVERLAP"),
        final_match_status="PARTIAL_MATCH",
    )
    before = copy.deepcopy(item)
    annotate_evidence(item)
    assert item["evidence_assessment"]["supporting_span_locations"][0]["ambiguous"]
    assert all(item[k] == v for k, v in before.items())
    item["llm_verdict"] = None
    annotate_evidence(item)
    assert item["evidence_assessment"]["semantic_status"] == "NOT_EVALUATED"


@pytest.mark.parametrize(
    "verdict,confidence,contradictions,status,expected",
    [
        ("INSUFFICIENT_CONTEXT", "HIGH", [], "MANUAL_REVIEW", "UNCERTAIN"),
        ("POSSIBLE_PARAPHRASE", "LOW", [], "MANUAL_REVIEW", "UNCERTAIN"),
        ("UNRELATED", "HIGH", ["Actor swapped"], "REJECTED", "CONTRADICTION_REPORTED"),
        (
            "PARTIAL_OVERLAP",
            "HIGH",
            ["Quantity differs"],
            "PARTIAL_MATCH",
            "PARTIAL_SUPPORT_WITH_CONTRADICTIONS",
        ),
        (
            "POSSIBLE_PARAPHRASE",
            "HIGH",
            [],
            "MANUAL_REVIEW",
            "CONFLICT_REQUIRES_REVIEW",
        ),
    ],
)
def test_uncertainty_and_contradiction_states_are_explicit(
    verdict, confidence, contradictions, status, expected
):
    item = dict(
        submission_text=SUB,
        source_text=SRC,
        llm_verdict=verdict,
        llm_verification=result(
            verdict=verdict, confidence=confidence, contradictions=contradictions
        ),
        final_match_status=status,
    )
    annotate_evidence(item)
    assert item["evidence_assessment"]["semantic_status"] == expected
