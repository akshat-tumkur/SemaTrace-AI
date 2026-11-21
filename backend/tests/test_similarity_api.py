"""Local saved-report integration. No retrieval/provider calls during recalculation."""

from __future__ import annotations


import importlib


import json


import sqlite3


from unittest.mock import MagicMock


import pytest


from fastapi.testclient import TestClient


from app.storage.database import AnalysisStore


TEXT = "Alpha beta gamma delta epsilon zeta eta theta iota kappa."


def saved_report(*, document=True):
    candidate = dict(
        candidate_id="c1",
        unit_id="u1",
        source_id="source-a",
        source_title="Source A",
        source_url="https://example.org/a",
        submission_text=TEXT,
        source_text=TEXT,
        source_document_id="version-a",
        source_document_sha256="a" * 64,
        source_start_char=100,
        source_end_char=100 + len(TEXT),
        passage_id="passage-a",
        deterministic_outcome="ACCEPT",
        verification_method="deterministic",
        final_match_status="MATCH",
        match_type="EXACT_OR_LEXICAL_OVERLAP",
        verification_score=0.9,
        token_overlap=1,
        source_occurrences=[
            dict(
                occurrence_id="occurrence-a",
                source_document_id="version-a",
                source_document_sha256="a" * 64,
                source_start_char=100,
                source_end_char=100 + len(TEXT),
                source_id="source-a",
                title="Source A",
                url="https://example.org/a",
                revision="revision-a",
            )
        ],
    )
    result = dict(
        filename="sample.txt",
        file_type="txt",
        status="completed",
        mode="offline",
        risk_level="HIGH",
        risk_score=0.5,
        coverage_percent=50,
        legacy_coverage_percent=50,
        verified_coverage_percent=50,
        strong_matches=1,
        possible_paraphrases=0,
        false_positives=0,
        candidate_count=1,
        summary="Preserved legacy summary",
        recommendations=["Review source attribution."],
        matches=[candidate],
        verification=dict(candidates=[candidate], metrics=dict(llm_calls=0)),
        audit_log=[dict(agent="offline", event="test", details="No provider calls.")],
        units=[
            dict(
                id="u1",
                text=TEXT,
                start_char=0,
                end_char=len(TEXT),
                unit_type="sentence",
            )
        ],
    )
    if document:
        result["document"] = dict(
            text=TEXT, offset_unit="unicode_code_points", end_exclusive=True
        )
    return result


@pytest.fixture
def api(monkeypatch, tmp_path):
    main = importlib.import_module("app.main")
    store = AnalysisStore(tmp_path / "analyses.sqlite3")
    monkeypatch.setattr(main, "analysis_store", store)
    forbidden = MagicMock(
        side_effect=AssertionError("Recalculation must not investigate or call models")
    )
    monkeypatch.setattr(main, "run_detection", forbidden)
    monkeypatch.setattr("app.product.providers.OpenAIReviewer", forbidden)
    with TestClient(main.app) as client:
        yield client, store, forbidden


def test_recalculation_persists_only_new_section_and_survives_restart(api):
    client, store, forbidden = api
    store.save("saved", saved_report())
    before = store.get("saved")
    response = client.post("/api/analysis/saved/similarity", json={})
    assert response.status_code == 200, response.text
    similarity = response.json()
    assert (
        similarity["total_eligible_words"] == similarity["unique_matched_words"] == 10
    )
    assert similarity["textual_similarity_percent"] == 100
    after = store.get("saved")
    assert {k: v for k, v in after.items() if k != "similarity_report"} == before
    assert after["similarity_report"] == similarity
    restarted = AnalysisStore(store.database_path)
    assert restarted.get("saved")["similarity_report"] == similarity
    assert (
        client.get("/api/analysis/saved/report").json()["similarity_report"]
        == similarity
    )
    forbidden.assert_not_called()


def test_explicit_exclusions_and_minimum_recalculate_saved_metric(api):
    client, store, forbidden = api
    store.save("saved", saved_report())
    report = client.post("/api/analysis/saved/similarity", json={}).json()
    source_id = report["sources"][0]["source_id"]
    excluded = client.post(
        "/api/analysis/saved/similarity", json={"excluded_source_ids": [source_id]}
    ).json()
    assert (
        excluded["unique_matched_words"] == 0 and excluded["total_eligible_words"] == 10
    )
    assert excluded["sources"][0]["excluded"]
    minimum = client.post(
        "/api/analysis/saved/similarity", json={"minimum_match_words": 100}
    ).json()
    assert (
        minimum["unique_matched_words"] == 0 and minimum["total_eligible_words"] == 10
    )
    quotation = client.post(
        "/api/analysis/saved/similarity",
        json={
            "excluded_ranges": [
                dict(start=0, end=len(TEXT), reason="EXPLICIT_QUOTATION")
            ]
        },
    ).json()
    assert quotation["total_eligible_words"] == quotation["unique_matched_words"] == 0
    assert quotation["textual_similarity_percent"] == 0
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"minimum_match_words": 2},
        {"minimum_match_words": 101},
        {"minimum_match_words": True},
        {"minimum_match_words": 5.5},
        {"minimum_match_words": "5"},
        {"unexpected": True},
        {"excluded_source_ids": [" "]},
        {"excluded_source_ids": ["x"] * 101},
        {"excluded_ranges": [dict(start=-1, end=4)]},
        {"excluded_ranges": [dict(start=5, end=4)]},
        {"excluded_ranges": [dict(start=0, end=len(TEXT) + 1)]},
        {"excluded_ranges": [dict(start=True, end=4)]},
        {"excluded_ranges": [dict(start=0, end=4, reason=" ")]},
        {"excluded_ranges": [dict(start=0, end=4)] * 101},
    ],
)
def test_invalid_exclusions_are_rejected_without_overwriting_report(api, options):
    client, store, forbidden = api
    store.save("saved", saved_report())
    before = store.get("saved")
    response = client.post("/api/analysis/saved/similarity", json=options)
    assert response.status_code == 422, response.text
    assert store.get("saved") == before
    forbidden.assert_not_called()


def test_older_missing_text_records_are_not_silently_reconstructed(api):
    client, store, forbidden = api
    store.save("old", saved_report(document=False))
    assert "similarity_report" not in client.get("/api/analysis/old").json()
    assert client.post("/api/analysis/old/similarity", json={}).status_code == 409
    assert client.post("/api/analysis/missing/similarity", json={}).status_code == 404
    assert "similarity_report" not in store.get("old")
    forbidden.assert_not_called()


def test_store_missing_update_does_not_insert_analysis(tmp_path):
    store = AnalysisStore(tmp_path / "analyses.sqlite3")
    assert not store.update_similarity_report(
        "missing", {"textual_similarity_percent": 0}
    )
    assert store.get("missing") is None


def test_store_atomic_update_preserves_non_similarity_metadata_and_rows(tmp_path):
    store = AnalysisStore(tmp_path / "analyses.sqlite3")
    store.save("saved", saved_report())
    with sqlite3.connect(store.database_path) as connection:
        before = connection.execute(
            "SELECT risk_score, coverage_percent, summary FROM analyses WHERE id='saved'"
        ).fetchone()
        matches = connection.execute("SELECT payload FROM matches").fetchall()
        audit = connection.execute("SELECT payload FROM audit_events").fetchall()
    assert store.update_similarity_report(
        "saved", {"method": "test-only", "textual_similarity_percent": 20}
    )
    with sqlite3.connect(store.database_path) as connection:
        assert (
            connection.execute(
                "SELECT risk_score, coverage_percent, summary FROM analyses WHERE id='saved'"
            ).fetchone()
            == before
        )
        assert connection.execute("SELECT payload FROM matches").fetchall() == matches
        assert (
            connection.execute("SELECT payload FROM audit_events").fetchall() == audit
        )
    assert store.get("saved")["similarity_report"]["textual_similarity_percent"] == 20


def test_similarity_write_uses_existing_rate_limiter():
    from fastapi import FastAPI
    from app.limits import RequestLimits

    local_app = FastAPI()
    local_app.add_middleware(RequestLimits, requests_per_minute=1)

    @local_app.post("/api/analysis/{analysis_id}/similarity")
    def local_write(analysis_id: str):
        return {"analysis_id": analysis_id}

    with TestClient(local_app) as client:
        assert client.post("/api/analysis/test/similarity", json={}).status_code == 200
        limited = client.post("/api/analysis/test/similarity", json={})
        assert limited.status_code == 429 and limited.headers["retry-after"] == "60"
