"""Real local-model upload → alignment → accounting → saved/PDF user journey."""

import importlib


from pathlib import Path


import pytest


from fastapi.testclient import TestClient


from app.storage.source_corpus import SourceCorpusStore


from app.storage.database import AnalysisStore


from app.corpus.models import SourceMetadata


from app.retrieval.generations import GenerationStore, IndexSpec


from app.retrieval.local_model import default_directory


from app.similarity import words


A = "The coastal laboratory collected samples from the northern monitoring station during winter"


B = "Researchers compared the measurements with previously recorded observations from nearby coastal stations"


SOURCE = A + " 2020. " + B + "."


TARGET = "🙂 " + A + " 2024. " + B + "."


@pytest.fixture
def local_api(monkeypatch, tmp_path):
    if not default_directory().is_dir():
        pytest.skip(
            "Provision the pinned local model for real inference integration tests"
        )
    main = importlib.import_module("app.main")
    store = SourceCorpusStore(tmp_path / "sources.sqlite3")
    for name in ("source-one", "source-alias"):
        store.ingest(
            SourceMetadata(
                collection="authorized-test",
                external_id=name,
                title=name,
                url="https://example.org/" + name,
                revision="revision-1",
                extractor_version="text-v1",
                rights_status="PRIVATE_UPLOAD",
            ),
            SOURCE,
            SOURCE.encode(),
        )
    corpus = store.snapshot()
    GenerationStore(corpus.retrieval_index_path).build(
        corpus, IndexSpec.local(), api_key=""
    )
    main.app.dependency_overrides[main.get_source_store] = lambda: store
    monkeypatch.setattr(
        main, "analysis_store", AnalysisStore(tmp_path / "analyses.sqlite3")
    )
    with TestClient(main.app) as client:
        yield client, main, store
    main.app.dependency_overrides.clear()


def test_actual_upload_uses_new_aligner_and_saved_exclusions(local_api, monkeypatch):
    client, main, store = local_api
    result = client.post(
        "/api/analyze", files={"file": ("unicode.txt", TARGET.encode(), "text/plain")}
    )
    assert result.status_code == 202, result.text
    saved = result.json()
    assert (
        saved["report_version"] == "3.0"
        and saved["mode"] == "Deterministic local document alignment"
    )
    assert (
        saved["findings"]
        and len({f["source_document_id"] for f in saved["findings"]}) == 2
    )
    assert all(
        f["submission_document_id"] == saved["document"]["document_id"]
        for f in saved["findings"]
    )
    report = saved["similarity_report"]
    expected = len(words(A)) + len(words(B))
    assert report["unique_matched_words"] == expected
    assert report["total_eligible_words"] == expected + 1
    assert sum(s["attributed_percent"] for s in report["sources"]) == pytest.approx(
        report["textual_similarity_percent"]
    )
    assert all("2024" not in entry["submission_excerpt"] for entry in report["ledger"])
    assert all(f["category"] == "MODIFIED_REUSE" for f in saved["findings"])
    assert saved["related_sources"] and all(
        not r["score_eligible"] for r in saved["related_sources"]
    )
    identifier = saved["analysis_id"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved views/exclusions/PDF must not redetect")

    monkeypatch.setattr(main, "run_detection", forbidden)
    reloaded = client.get("/api/analysis/" + identifier).json()
    assert reloaded["findings"] == saved["findings"]
    assert "alignment_sources" not in reloaded
    assert (
        client.get("/api/analysis/" + identifier).headers["cache-control"] == "no-store"
    )
    assert (
        client.delete(
            "/api/analysis/" + identifier,
            headers={"Origin": "https://unrelated.example"},
        ).status_code
        == 403
    )
    assert (
        client.get("/api/health", headers={"Host": "unrelated.example"}).status_code
        == 400
    )
    ids = [s["source_id"] for s in report["sources"]]
    one = client.post(
        "/api/analysis/" + identifier + "/similarity",
        json={"excluded_source_ids": ids[:1]},
    ).json()
    assert one["unique_matched_words"] == expected
    both = client.post(
        "/api/analysis/" + identifier + "/similarity", json={"excluded_source_ids": ids}
    ).json()
    assert both["unique_matched_words"] == 0
    restored = client.post(
        "/api/analysis/" + identifier + "/similarity", json={}
    ).json()
    assert restored == report
    pdf = client.get("/api/analysis/" + identifier + "/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert (
        client.get("/api/analysis/" + identifier).json()["similarity_report"] == report
    )
    listing = client.get("/api/analyses").json()
    assert listing == [
        {
            "analysis_id": identifier,
            "filename": "unicode.txt",
            "status": "completed",
            "created_at": saved["created_at"],
        }
    ]
    assert client.delete("/api/analysis/" + identifier).status_code == 200
    assert client.get("/api/analysis/" + identifier).status_code == 404
    assert client.get("/api/analyses").json() == []
    with main.analysis_store._connect() as connection:
        for table in ("units", "matches", "audit_events"):
            assert (
                connection.execute(
                    "SELECT count(*) FROM " + table + " WHERE analysis_id=?",
                    (identifier,),
                ).fetchone()[0]
                == 0
            )


def test_missing_local_index_is_an_error_not_fake_success(local_api):
    client, main, store = local_api
    with GenerationStore(store.snapshot().retrieval_index_path).connect() as c:
        c.execute("DELETE FROM active_generations")
    response = client.post(
        "/api/analyze", files={"file": ("test.txt", TARGET.encode())}
    )
    assert response.status_code == 503
