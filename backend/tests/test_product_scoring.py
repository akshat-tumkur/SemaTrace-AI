import hashlib


import pytest


from app.alignment.exact import exact_runs


from app.scoring import build_report


TEXT = "Coastal researchers measured winter temperatures across northern monitoring stations"


def source(text):
    return {
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "revision": "one",
        "title": "Source",
        "url": "https://example.org",
        "attribution": {},
    }


def test_saved_word_grounding_rejects_partial_source_words_and_changed_text():
    finding = exact_runs(TEXT, TEXT)[0]
    finding["source_document_id"] = "source"
    data = {"source": source(TEXT)}
    assert build_report(TEXT, [finding], data)["unique_matched_words"] == 9
    data["source"]["text"] = TEXT + " modified"
    with pytest.raises(ValueError, match="integrity"):
        build_report(TEXT, [finding], data)
    # Same source substring inside a longer word must not become whole-word evidence.
    prefixed = "X" + TEXT
    data = {"source": source(prefixed)}
    finding = dict(
        finding,
        source_start=1,
        source_end=len(prefixed),
        matched_runs=[
            dict(finding["matched_runs"][0], source_start=1, source_end=len(prefixed))
        ],
    )
    with pytest.raises(ValueError, match="source token"):
        build_report(TEXT, [finding], data)


def test_repeated_submitted_passages_count_separately_and_source_aliases_once():
    target = TEXT + ". " + TEXT
    findings = []
    for sid in ("one", "two"):
        findings.extend(
            dict(f, source_document_id=sid) for f in exact_runs(target, TEXT)
        )
    sources = {sid: source(TEXT) for sid in ("one", "two")}
    report = build_report(target, findings, sources)
    assert report["unique_matched_words"] == 18
    assert sum(s["attributed_matched_words"] for s in report["sources"]) == 18
    assert (
        build_report(target, findings, sources, {"excluded_source_ids": ["one"]})[
            "unique_matched_words"
        ]
        == 18
    )
    excluded = build_report(
        target,
        findings,
        sources,
        {"excluded_ranges": [{"start": 0, "end": len(TEXT), "reason": "quote"}]},
    )
    assert (
        excluded["unique_matched_words"] == 9 and excluded["total_eligible_words"] == 9
    )
