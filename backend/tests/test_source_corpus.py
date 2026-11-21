import copy


import importlib


from pathlib import Path


import sqlite3


from fastapi.testclient import TestClient


import pytest


from app.corpus.models import CorpusUnavailable, SourceMetadata


from app.storage.source_corpus import SourceCorpusStore, digest


from app.retrieval.corpus import load_local_corpus


from app.retrieval.bm25 import LocalCorpus


from app.storage.database import AnalysisStore


TEXT = "The Cedar reservoir stores winter rainfall in carefully covered storage tanks throughout the extended dry season."


def meta(identifier="one", **changes):
    return SourceMetadata(
        **{
            **dict(
                collection="test",
                external_id=identifier,
                title="Cedar notes",
                url="https://example.org/" + identifier,
                revision="1",
                extractor_version="plain-v1",
            ),
            **changes,
        }
    )


def test_restart_idempotence_and_stable_versions(tmp_path):
    path = tmp_path / "corpus.db"
    store = SourceCorpusStore(path)
    first = store.ingest(meta(), TEXT, TEXT.encode())
    snap = store.snapshot()
    second = SourceCorpusStore(path).ingest(meta(), TEXT, TEXT.encode())
    assert second["duplicate"] and second["units_indexed"] == 0
    assert (
        first["source_id"] == second["source_id"]
        and first["version_id"] == second["version_id"]
    )
    assert SourceCorpusStore(path).snapshot().corpus_info == snap.corpus_info
    assert snap.search("covered winter rainfall")[0].source_text == TEXT


def test_occurrence_and_cross_source_dedup_preserve_all_offsets(tmp_path):
    store = SourceCorpusStore(tmp_path / "corpus.db")
    original = "  " + TEXT + "\n\n" + TEXT
    store.ingest(meta(), original, original.encode())
    store.ingest(meta("two"), original, original.encode())
    snapshot = store.snapshot()
    assert len(snapshot.documents) == 1
    d = snapshot.documents[0]
    assert len(d.source_occurrences) == 4
    assert len({o.occurrence_id for o in d.source_occurrences}) == 4
    for o in d.source_occurrences:
        assert (
            snapshot.source_documents[o.source_document_id][
                o.source_start_char : o.source_end_char
            ]
            == d.text
        )
    assert snapshot.corpus_info["unique_text_documents"] == 1
    assert snapshot.corpus_info["source_origins"] == 2
    assert snapshot.corpus_info["passage_occurrences"] == 4


def test_refresh_does_not_mutate_old_snapshot_or_version(tmp_path):
    store = SourceCorpusStore(tmp_path / "corpus.db")
    old = store.ingest(meta(), TEXT, TEXT.encode())
    snapshot = store.snapshot()
    newer = TEXT.replace("winter", "summer")
    new = store.ingest(meta(revision="2"), newer, newer.encode())
    assert new["version_id"] != old["version_id"]
    assert (
        snapshot.documents[0].text == TEXT
        and store.snapshot().documents[0].text == newer
    )
    assert store.get_version(old["version_id"])["text"] == TEXT
    assert store.stats()["source_origins"] == 1


def test_failed_index_build_rolls_back_active_revision(tmp_path, monkeypatch):
    store = SourceCorpusStore(tmp_path / "corpus.db")
    old = store.ingest(meta(), TEXT, TEXT.encode())
    before = store.stats()

    def fail(*args):
        raise CorpusUnavailable("test_index_build_failed")

    with monkeypatch.context() as m:
        m.setattr(store, "_snapshot", fail)
        with pytest.raises(CorpusUnavailable):
            store.ingest(meta(revision="2"), TEXT + " A later claim.", b"new raw")
    assert store.stats() == before
    assert store.list_sources()[0]["version_id"] == old["version_id"]


@pytest.mark.parametrize("corruption", ["missing_blob", "changed_blob", "bad_offset"])
def test_integrity_failures_are_not_empty_or_stale_fallbacks(tmp_path, corruption):
    store = SourceCorpusStore(tmp_path / "corpus.db")
    r = store.ingest(meta(), TEXT, TEXT.encode())
    if corruption == "missing_blob":
        (store.blobs / digest(TEXT)).unlink()
    elif corruption == "changed_blob":
        (store.blobs / digest(TEXT)).write_bytes(b"wrong content")
    else:
        with store.connect() as c:
            c.execute("UPDATE occurrences SET start_char=3")
    with pytest.raises(CorpusUnavailable):
        store.snapshot()


def test_metadata_only_and_collection_document_filters(tmp_path):
    store = SourceCorpusStore(tmp_path / "corpus.db")
    identifier = store.record_metadata(meta())
    assert (
        store.stats()["metadata_only_sources"] == 1
        and store.stats()["unique_passages"] == 0
    )
    assert store.list_sources()[0]["content_status"] == "METADATA_ONLY"
    first = store.ingest(meta(), TEXT, TEXT.encode())
    second = store.ingest(meta("two", collection="other"), TEXT, TEXT.encode())
    assert store.snapshot(collections=("test",)).corpus_info["source_origins"] == 1
    assert (
        store.snapshot(source_ids=(second["source_id"],)).corpus_info["source_origins"]
        == 1
    )
    assert store.snapshot(collections=("absent",)).documents == []


def test_legacy_ids_decimals_and_unicode_survive_import(tmp_path):
    folder = tmp_path / "legacy"
    folder.mkdir()
    original = "  Dr. Iyer measured 12.5 volts in the calibrated laboratory experiment.\nUnicode π remains in the original recorded source snapshot."
    (folder / "lab.txt").write_text(original, encoding="utf-8")
    corpus = load_local_corpus(folder)
    store = SourceCorpusStore(tmp_path / "corpus.db")
    store.import_legacy(corpus)
    snapshot = store.snapshot()
    assert [d.source_id for d in snapshot.documents] == [
        d.source_id for d in corpus.documents
    ]
    assert [d.source_start_char for d in snapshot.documents] == [
        d.source_start_char for d in corpus.documents
    ]
    assert all(
        o.rights_status == "UNVERIFIED"
        for d in snapshot.documents
        for o in d.source_occurrences
    )


def test_runs_track_failures_and_resume_records_without_inventing_success(tmp_path):
    store = SourceCorpusStore(tmp_path / "corpus.db")
    run = store.start_run({"max_documents": 12})
    store.record_item(run, "missing", "FAILED", {"error": "HTTP_404"})
    store.finish_run(run, "PARTIAL")
    assert (
        store.get_run(run)["status"] == "PARTIAL"
        and store.stats()["failed_ingestion_items"] == 1
    )
    store.record_item(run, "missing", "INDEXED", {"units_indexed": 1})
    assert store.get_run(run)["items"][0]["status"] == "INDEXED"
    assert store.stats()["failed_ingestion_items"] == 0


def test_api_upload_survives_new_client_and_reports_versioned_provenance(
    tmp_path, monkeypatch
):
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "local_corpus", LocalCorpus([]))
    monkeypatch.setattr(main, "analysis_store", AnalysisStore(tmp_path / "analysis.db"))
    with TestClient(main.app) as client:
        source = client.post(
            "/api/sources",
            data={"title": "Cedar notes", "url": "https://example.org/cedar"},
            files={"file": ("cedar.txt", TEXT.encode(), "text/plain")},
        )
        assert source.status_code == 201 and source.json()["persisted"]
        duplicate = client.post(
            "/api/sources",
            data={"title": "Cedar notes", "url": "https://example.org/cedar"},
            files={"file": ("cedar.txt", TEXT.encode(), "text/plain")},
        )
        assert duplicate.json()["duplicate"] and duplicate.json()["units_indexed"] == 0
    with TestClient(main.app) as client:
        assert client.get("/api/corpus").json()["unique_passages"] == 1
        assert client.get("/api/corpus/sources?limit=0").status_code == 422
        assert (
            client.get("/api/corpus/sources").json()[0]["rights_status"]
            == "PRIVATE_UPLOAD"
        )
        result = client.post(
            "/api/analyze", files={"file": ("copy.txt", TEXT.encode(), "text/plain")}
        ).json()
        assert len(result["matches"]) == 1
        match = result["matches"][0]
        o = match["source_occurrences"][0]
        original = client.get("/api/corpus/versions/" + o["source_document_id"]).json()[
            "text"
        ]
        assert (
            original[o["source_start_char"] : o["source_end_char"]]
            == match["source_text"]
        )
        assert result["corpus"]["dense_status"] == "LOCAL_LEARNED_PERSISTENT_GENERATION"
        assert result["verification"]["metrics"]["llm_calls"] == 0
        saved = client.get("/api/analysis/" + result["analysis_id"] + "/report").json()
        assert saved["corpus"] == result["corpus"]


def test_store_rejects_unknown_schema_and_invalid_blob_paths(tmp_path):
    path = tmp_path / "db"
    with sqlite3.connect(path) as c:
        c.execute("PRAGMA user_version=99")
    with pytest.raises(CorpusUnavailable):
        SourceCorpusStore(path)
    store = SourceCorpusStore(tmp_path / "other.db")
    with pytest.raises(CorpusUnavailable):
        store.read_blob("../arbitrary-file")


def test_corrupt_api_corpus_returns_explicit_503(tmp_path, monkeypatch):
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "local_corpus", LocalCorpus([]))
    store = main.get_source_store()
    store.ingest(meta(), TEXT, TEXT.encode())
    (store.blobs / digest(TEXT)).write_bytes(b"corrupted")
    with TestClient(main.app) as client:
        assert client.get("/api/corpus").status_code == 503
        response = client.post(
            "/api/analyze", files={"file": ("copy.txt", TEXT.encode(), "text/plain")}
        )
        assert response.status_code == 503


def test_original_raw_and_metadata_are_retained_without_fabricated_dates(tmp_path):
    store = SourceCorpusStore(tmp_path / "db")
    raw = b"Original file bytes distinct from normalized source text"
    r = store.ingest(meta(), TEXT, raw)
    v = store.get_version(r["version_id"])
    assert store.read_blob(v["raw_sha256"]) == raw
    assert v["metadata"]["published_at"] is None and v["metadata"]["authors"] == []
    assert store.list_sources()[0]["ingested_at"]


def test_pinned_corpus_can_be_reconstructed_after_refresh_and_new_sources(tmp_path):
    store = SourceCorpusStore(tmp_path / "db")
    store.ingest(meta(), TEXT, TEXT.encode())
    original = store.snapshot()
    version = original.corpus_info["version"]
    store.ingest(meta(revision="2"), TEXT.replace("winter", "summer"), b"new raw")
    store.ingest(
        meta("another"),
        "A different source contains additional context for the new corpus version.",
        b"other raw",
    )
    recovered = SourceCorpusStore(store.path).snapshot(version_id=version)
    assert recovered.corpus_info == original.corpus_info
    assert recovered.documents == original.documents
    with pytest.raises(CorpusUnavailable):
        store.snapshot(version_id="unknown-version")


def test_analysis_can_pin_the_corpus_version_after_source_refresh(
    tmp_path, monkeypatch, provision_local_index
):
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "local_corpus", LocalCorpus([]))
    monkeypatch.setattr(main, "analysis_store", AnalysisStore(tmp_path / "analyses.db"))
    store = main.get_source_store()
    store.ingest(meta(), TEXT, TEXT.encode())
    old = store.snapshot().corpus_info["version"]
    provision_local_index(store.snapshot())
    store.ingest(meta(revision="2"), TEXT.replace("winter", "summer"), b"new raw")
    with TestClient(main.app) as client:
        result = client.post(
            "/api/analyze",
            data={"corpus_version": old},
            files={"file": ("old.txt", TEXT.encode(), "text/plain")},
        )
        assert result.status_code == 202
        assert result.json()["corpus"]["version"] == old
        assert result.json()["matches"][0]["source_text"] == TEXT.rstrip(".")
