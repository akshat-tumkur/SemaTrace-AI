"""Deterministic alignment/accounting controls; not real-world plagiarism accuracy."""

import copy


import hashlib


import pytest


from app.similarity import build_similarity_report


WORDS = "crystal lantern meadow harbor silver planet forest copper velvet glacier ribbon orchard canyon ember pebble willow river autumn marble summit".split()


def unit(text, key="u1", start=0):
    return dict(
        id=key,
        text=text,
        start_char=start,
        end_char=start + len(text),
        unit_type="sentence",
    )


def candidate(
    submission,
    source,
    *,
    unit_id="u1",
    source_id="source-A-v1",
    source_start=100,
    **changes,
):
    value = dict(
        candidate_id="c-" + source_id + "-" + unit_id,
        unit_id=unit_id,
        submission_text=submission,
        source_text=source,
        source_title=source_id,
        source_url="https://example.org/" + source_id,
        source_id=source_id,
        source_document_id=source_id,
        source_document_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_start_char=source_start,
        source_end_char=source_start + len(source),
        passage_id=hashlib.sha256(source.encode()).hexdigest(),
        verification_method="deterministic",
        deterministic_outcome="ACCEPT",
        final_match_status="MATCH",
        match_type="EXACT_OR_LEXICAL_OVERLAP",
    )
    value.update(changes)
    return value


def report(text, candidates, units=None, options=None):
    return build_similarity_report(
        text, units or [unit(text)], candidates, "document-test", options
    )


def assert_counts(result, total, matched, percent):
    assert result["total_eligible_words"] == total
    assert result["unique_matched_words"] == matched
    assert result["textual_similarity_percent"] == pytest.approx(percent, abs=0.01)
    assert 0 <= result["textual_similarity_percent"] <= 100
    assert sum(s["attributed_matched_words"] for s in result["sources"]) == matched


def matched_token_ids(result):
    return {
        i
        for e in result["ledger"]
        if e["score_eligible"]
        for i in e["document_token_ids"]
    }


def test_twenty_copied_words_in_a_hundred_word_document():
    words = (
        [f"prefix{i}" for i in range(10)] + WORDS + [f"original{i}" for i in range(70)]
    )
    text = " ".join(words)
    result = report(text, [candidate(text, " ".join(WORDS))])
    assert_counts(result, 100, 20, 20)
    assert matched_token_ids(result) == set(range(10, 30))
    eligible = [e for e in result["ledger"] if e["score_eligible"]]
    assert eligible
    for entry in eligible:
        assert entry["mechanically_grounded"]
        assert text[entry["document_start"] : entry["document_end"]] == " ".join(WORDS)
        assert entry["source_start"] == 100
        assert entry["source_end"] == 100 + len(" ".join(WORDS))
        assert entry["source_document_id"] == "source-A-v1"
        assert len(entry["document_token_ids"]) == 20


def test_two_sources_for_same_span_do_not_double_count_and_primary_is_stable():
    text = " ".join(WORDS)
    a, b = candidate(text, text), candidate(text, text, source_id="source-B-v1")
    result = report(text, [a, b])
    assert_counts(result, 20, 20, 100)
    assert sorted(s["raw_matched_words"] for s in result["sources"]) == [20, 20]
    assert sorted(s["attributed_matched_words"] for s in result["sources"]) == [0, 20]
    assert result == report(text, [b, a])


def test_partially_overlapping_sources_are_unioned():
    text = " ".join(WORDS)
    result = report(
        text,
        [
            candidate(text, " ".join(WORDS[:12])),
            candidate(text, " ".join(WORDS[7:]), source_id="source-B-v1"),
        ],
    )
    assert_counts(result, 20, 20, 100)
    assert sorted(s["raw_matched_words"] for s in result["sources"]) == [12, 13]
    assert sum(s["raw_matched_words"] for s in result["sources"]) == 25


def test_source_display_shares_sum_exactly_to_rounded_similarity():
    text = " ".join(WORDS[:9])
    candidates = [
        candidate(text, " ".join(WORDS[:3])),
        candidate(text, " ".join(WORDS[3:6]), source_id="source-B-v1"),
    ]
    result = report(text, candidates, options=dict(minimum_match_words=3))
    assert_counts(result, 9, 6, 66.67)
    assert (
        sum(s["attributed_percent_hundredths"] for s in result["sources"])
        == result["textual_similarity_percent_hundredths"]
        == 6667
    )
    assert sorted(s["attributed_percent_hundredths"] for s in result["sources"]) == [
        3333,
        3334,
    ]


def test_repeated_submission_occurrences_are_independent_units():
    copied = " ".join(WORDS[:10])
    text = copied + ". " + copied
    units = [unit(copied, "u1"), unit(copied, "u2", len(copied) + 2)]
    candidates = [candidate(copied, copied), candidate(copied, copied, unit_id="u2")]
    assert_counts(report(text, candidates, units), 20, 20, 100)


def test_repeated_phrase_within_one_unit_keeps_both_document_occurrences():
    copied = " ".join(WORDS[:8])
    text = copied + " interlude " + copied
    assert_counts(report(text, [candidate(text, copied)]), 17, 16, 100 * 16 / 17)


def test_inserted_word_is_not_counted_as_reused():
    src = " ".join(WORDS)
    text = " ".join(WORDS[:10] + ["unmatchedinsertion"] + WORDS[10:])
    result = report(text, [candidate(text, src)])
    assert_counts(result, 21, 20, 100 * 20 / 21)
    assert matched_token_ids(result) == set(range(21)) - {10}


def test_substituted_word_is_not_counted_as_reused():
    src = " ".join(WORDS)
    words = WORDS.copy()
    words[10] = "novelsubstitution"
    text = " ".join(words)
    result = report(text, [candidate(text, src)])
    assert_counts(result, 20, 19, 95)
    assert matched_token_ids(result) == set(range(20)) - {10}


@pytest.mark.parametrize(
    "text",
    [
        "In conclusion",
        "further research is needed",
        "the results show",
        "the and of in to by with from",
    ],
)
def test_short_generic_and_stopword_runs_do_not_inflate_similarity(text):
    result = report(text, [candidate(text, text)])
    assert result["unique_matched_words"] == 0
    assert result["textual_similarity_percent"] == 0


def test_scattered_common_words_do_not_form_a_copied_span():
    text = (
        "The crystal study and lantern work concerns meadow findings in harbor reports."
    )
    source = "A crystal example with lantern illustrations records meadow observations beside harbor notes."
    assert report(text, [candidate(text, source)])["unique_matched_words"] == 0


def test_paraphrase_and_dual_findings_are_not_confirmed_textual_words():
    text = "Brief recordings allow synthetic reproduction of an individual voice."
    source = "Speech cloning recreates a speaker using seconds of captured audio."
    c = candidate(
        text,
        source,
        verification_method="llm",
        llm_verdict="POSSIBLE_PARAPHRASE",
        match_type="POSSIBLE_PARAPHRASE",
        final_match_status="POTENTIAL_MATCH",
    )
    r = report(text, [c])
    assert r["unique_matched_words"] == 0 and r["unresolved_findings"]
    copied = " ".join(WORDS)
    dual = candidate(
        copied,
        copied,
        verification_method="llm",
        final_match_status="POTENTIAL_REUSE",
        dual_assessment={"textual_reuse": {"verdict": "YES"}},
    )
    r = report(copied, [dual])
    assert r["unique_matched_words"] == 0 and r["unresolved_findings"]


def test_changed_date_counts_surrounding_words_without_claiming_semantic_support():
    src = " ".join(WORDS[:10] + ["2015"] + WORDS[10:])
    text = " ".join(WORDS[:10] + ["2016"] + WORDS[10:])
    c = candidate(text, src, deterministic_conflicts=["different_numbers"])
    c["effective_semantic_support"] = {
        "status": "UNKNOWN",
        "semantic_support_coverage_permitted": False,
    }
    result = report(text, [c])
    assert_counts(result, 21, 20, 100 * 20 / 21)
    assert matched_token_ids(result) == set(range(21)) - {10}


def test_unsupported_clause_is_not_counted_from_whole_sentence_retention():
    src = " ".join(WORDS[:10])
    text = src + "; " + " ".join(f"unrelated{i}" for i in range(10))
    result = report(text, [candidate(text, src)])
    assert_counts(result, 20, 10, 50)
    assert matched_token_ids(result) == set(range(10))


def test_explicit_quotation_exclusion_removes_document_words_from_both_counts():
    copied = " ".join(WORDS[:10])
    text = copied + " " + " ".join(f"original{i}" for i in range(10))
    options = dict(
        excluded_ranges=[dict(start=0, end=len(copied), reason="quoted_text")]
    )
    result = report(text, [candidate(text, copied)], options=options)
    assert_counts(result, 10, 0, 0)
    assert result["exclusions"]["excluded_document_words"] == 10


def test_selected_source_exclusion_changes_numerator_not_denominator():
    text = " ".join(WORDS)
    c = candidate(text, text)
    result = report(
        text, [c], options=dict(excluded_source_ids=[c["source_document_id"]])
    )
    assert_counts(result, 20, 0, 0)


def test_excluding_primary_source_preserves_a_qualified_alternative():
    text = " ".join(WORDS)
    a, b = candidate(text, text), candidate(text, text, source_id="source-B-v1")
    result = report(
        text, [a, b], options=dict(excluded_source_ids=[a["source_document_id"]])
    )
    assert_counts(result, 20, 20, 100)
    chosen = next(
        s for s in result["sources"] if s["source_id"] == b["source_document_id"]
    )
    assert chosen["attributed_matched_words"] == 20


def test_minimum_word_run_policy_is_configurable():
    copied = " ".join(WORDS[:5])
    assert (
        report(
            copied, [candidate(copied, copied)], options=dict(minimum_match_words=6)
        )["unique_matched_words"]
        == 0
    )
    assert (
        report(
            copied, [candidate(copied, copied)], options=dict(minimum_match_words=5)
        )["unique_matched_words"]
        == 5
    )


def test_unicode_casefold_and_repeated_positions_preserve_original_offsets():
    copied = "Café naïve résumé lantern glacier copper meadow summit"
    text = "🌍 " + copied + ". " + copied
    a, b = unit(copied, "u1", 2), unit(copied, "u2", 2 + len(copied) + 2)
    result = report(
        text,
        [candidate(copied, copied), candidate(copied, copied, unit_id="u2")],
        [a, b],
    )
    assert_counts(result, 16, 16, 100)
    eligible = [e for e in result["ledger"] if e["score_eligible"]]
    assert sorted({(e["document_start"], e["document_end"]) for e in eligible}) == [
        (a["start_char"], a["end_char"]),
        (b["start_char"], b["end_char"]),
    ]
    assert_counts(
        report(copied.upper(), [candidate(copied.upper(), copied)]), 8, 8, 100
    )


def test_repeated_original_source_occurrences_remain_distinct_without_extra_words():
    text = " ".join(WORDS[:10])
    c = candidate(text, text)
    c["source_occurrences"] = [
        dict(
            occurrence_id=f"o{i}",
            source_document_id="source-A-v1",
            source_start_char=n,
            source_end_char=n + len(text),
            source_document_sha256=c["source_document_sha256"],
            title="Original source",
            url="https://example.org/original",
        )
        for i, n in enumerate([100, 500])
    ]
    result = report(text, [c])
    assert_counts(result, 10, 10, 100)
    entries = [e for e in result["ledger"] if e["score_eligible"]]
    assert {e["source_start"] for e in entries} == {100, 500}
    assert len(result["sources"]) == 1
    assert result["sources"][0]["raw_matched_words"] == 10


def test_duplicate_candidate_observations_do_not_increase_score():
    text = " ".join(WORDS)
    c = candidate(text, text)
    duplicate = copy.deepcopy(c)
    duplicate["candidate_id"] = "duplicate-observation"
    result = report(text, [c, duplicate])
    assert_counts(result, 20, 20, 100)
    assert len(result["sources"]) == 1


def test_overlapping_explicit_exclusions_remove_each_denominator_word_once():
    text = " ".join(WORDS)
    first_end = len(" ".join(WORDS[:10]))
    second_start = len(" ".join(WORDS[:5])) + 1
    second_end = len(" ".join(WORDS[:15]))
    result = report(
        text,
        [candidate(text, text)],
        options=dict(
            excluded_ranges=[
                dict(start=0, end=first_end, reason="quoted_text"),
                dict(
                    start=second_start, end=second_end, reason="bibliography_explicit"
                ),
            ]
        ),
    )
    assert_counts(result, 5, 5, 100)
    assert result["exclusions"]["excluded_document_words"] == 15


def test_ambiguous_source_range_is_unresolved_instead_of_fabricated():
    text = " ".join(WORDS)
    c = candidate(text, text, source_end_char=999999)
    result = report(text, [c])
    assert result["unique_matched_words"] == 0 and result["unresolved_findings"]


def test_missing_source_provenance_is_unresolved_not_invented():
    text = " ".join(WORDS)
    c = candidate(text, text)
    for k in ("source_document_id", "source_start_char", "source_end_char"):
        c.pop(k)
    result = report(text, [c])
    assert result["unique_matched_words"] == 0
    assert result["unresolved_findings"]


def test_existing_false_positive_filter_remains_an_exclusion():
    text = " ".join(WORDS)
    result = report(
        text, [candidate(text, text, final_match_status="FILTERED_FALSE_POSITIVE")]
    )
    assert result["unique_matched_words"] == 0


def test_semantic_rejection_does_not_disprove_literal_word_reuse():
    source = " ".join(WORDS[:10] + ["2015"] + WORDS[10:])
    text = " ".join(WORDS[:10] + ["2016"] + WORDS[10:])
    c = candidate(
        text,
        source,
        verification_method="llm",
        llm_verdict="UNRELATED",
        final_match_status="REJECTED",
    )
    assert_counts(report(text, [c]), 21, 20, 100 * 20 / 21)


@pytest.mark.parametrize("text", ["", "!!! 🌍 ..."])
def test_empty_or_wordless_documents_have_safe_zero_denominator(text):
    assert_counts(report(text, []), 0, 0, 0)


def test_all_explicitly_excluded_document_has_zero_score():
    text = " ".join(WORDS)
    result = report(
        text,
        [candidate(text, text)],
        options=dict(excluded_ranges=[dict(start=0, end=len(text), reason="explicit")]),
    )
    assert_counts(result, 0, 0, 0)


def test_no_match_and_input_candidate_mutation_are_safe():
    text = " ".join(WORDS)
    c = candidate(
        text,
        "Different research describes entirely unrelated engineering developments.",
    )
    prior = copy.deepcopy(c)
    assert_counts(report(text, [c]), 20, 0, 0)
    assert c == prior
