"""Saved-value PDF integration, Unicode, exclusions and injection regressions."""

from test_similarity_api import api, saved_report


import pymupdf


def text_of(raw):
    with pymupdf.open(stream=raw, filetype="pdf") as document:
        return "\n".join(page.get_text() for page in document)


def test_pdf_uses_saved_values_without_detection_and_tracks_exclusions(api):
    client, store, forbidden = api
    record = saved_report()
    record["filename"] = "Résumé <script>.txt"
    record["created_at"] = "2026-11-20T12:00:00+00:00"
    store.save("pdf", record)
    assert client.get("/api/analysis/pdf/report.pdf").status_code == 409
    report = client.post("/api/analysis/pdf/similarity", json={}).json()
    before = store.get("pdf")
    response = client.get("/api/analysis/pdf/report.pdf")
    assert (
        response.status_code == 200
        and response.headers["content-type"] == "application/pdf"
    )
    assert response.headers["cache-control"] == "no-store"
    text = text_of(response.content)
    assert "100.0%" in text and "Résumé <script>.txt" in text and "2026-11-20" in text
    assert store.get("pdf") == before
    client.post(
        "/api/analysis/pdf/similarity",
        json={"excluded_source_ids": [report["sources"][0]["source_id"]]},
    )
    excluded = text_of(client.get("/api/analysis/pdf/report.pdf").content)
    assert "0.0%" in excluded and "SOURCE_EXCLUDED" in excluded
    forbidden.assert_not_called()
    assert client.get("/api/analysis/missing/report.pdf").status_code == 404


def test_long_evidence_unicode_and_empty_pdf():
    from app.reports.pdf import render_pdf
    from app.similarity import build_similarity_report

    result = saved_report()
    result["analysis_id"] = "long"
    result["similarity_report"] = build_similarity_report(
        result["document"]["text"], result["units"], result["matches"], "submission"
    )
    report = result["similarity_report"]
    row = report["ledger"][0]
    report["ledger"] = [
        dict(
            row,
            submission_excerpt="Greek αβγ; Cyrillic Пример; emoji 🙂. "
            + row["submission_excerpt"] * 15,
        )
        for _ in range(24)
    ]
    report["sources"][0]["url"] = "https://example.org/" + "long-url-part" * 100
    raw = render_pdf(result)
    with pymupdf.open(stream=raw, filetype="pdf") as document:
        assert len(document) > 3
        text = "".join(page.get_text() for page in document)
        assert "αβγ" in text and "Пример" in text and "[U+1F642]" in text
        assert "24." in text and "Font coverage:" in text
    report["ledger"] = []
    report["sources"] = []
    assert "No grounded textual passages" in text_of(render_pdf(result))


def test_pdf_export_concurrency_is_bounded(api):
    from app import main

    client, store, _ = api
    store.save("bounded", saved_report())
    main.pdf_export_slot.acquire()
    try:
        response = client.get("/api/analysis/bounded/report.pdf")
        assert response.status_code == 429 and response.headers["retry-after"] == "5"
    finally:
        main.pdf_export_slot.release()


def test_zero_word_coverage_still_shows_saved_semantic_sources_and_literal_context():
    import copy
    from app.reports.pdf import render_pdf
    from app.similarity import build_similarity_report

    result = saved_report()
    result["analysis_id"] = "semantic-zero"
    result["similarity_report"] = build_similarity_report(
        result["document"]["text"], result["units"], [], "submission"
    )
    result["related_sources"] = [
        {
            "candidate_id": "first",
            "submission_start": 0,
            "submission_end": 23,
            "submission_excerpt": "Moisture returns to air.",
            "source_document_id": "water-fixed",
            "source_title": "Water cycle",
            "source_url": "https://example.org/water",
            "source_revision": "frozen",
            "source_document_sha256": "0" * 64,
            "source_start": 0,
            "source_end": 21,
            "source_excerpt": "Water becomes vapor.",
            "minilm_rank": 1,
            "score": 0.8,
        }
    ]
    before = copy.deepcopy(result)
    text = text_of(render_pdf(result))
    assert "Water cycle: first-ranked for 1 of 1" in text
    assert "Moisture returns to air." in text and "Water becomes vapor." in text
    assert "MATCHING-WORD COVERAGE" in text and "0.0%" in text
    assert result == before
