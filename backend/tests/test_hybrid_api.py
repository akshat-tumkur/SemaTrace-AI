import importlib


import json


import sqlite3


from fastapi.testclient import TestClient


from app.retrieval.bm25 import CorpusDocument, LocalCorpus


from app.storage.database import AnalysisStore, VERIFICATION_FIELDS


def test_api_creation_and_persisted_report_preserve_hybrid_metadata(
    monkeypatch, tmp_path, provision_local_index
):
    monkeypatch.setenv("SEMATRACE_DB_PATH", str(tmp_path / "analysis.db"))
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "analysis_store", AnalysisStore(tmp_path / "analysis.db"))
    source = "Voice cloning systems recreate a speaker voice using several seconds of carefully recorded audio."
    monkeypatch.setattr(
        main,
        "local_corpus",
        LocalCorpus([CorpusDocument("s1", "Voice", "local://s1", source)]),
    )
    provision_local_index(main.get_source_store().snapshot())
    with TestClient(main.app) as client:
        response = client.post(
            "/api/analyze", files={"file": ("test.txt", source.encode(), "text/plain")}
        )
        assert response.status_code == 202
        created = response.json()
        loaded = client.get(f"/api/analysis/{created['analysis_id']}").json()
        report = client.get(f"/api/analysis/{created['analysis_id']}/report").json()
        for key in VERIFICATION_FIELDS:
            assert key in created
            assert created[key] == loaded[key] == report[key]
        assert (
            created["verification"]["candidates"][0]["verification_method"]
            == "deterministic"
        )
        assert created["verification"]["metrics"]["llm_calls"] == 0
        assert (
            client.get("/api/health").json()["evidence_verification"]
            == "deterministic-fallback"
        )


def test_existing_database_gets_additive_metadata_migration(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE analyses (
            id TEXT PRIMARY KEY, filename TEXT, file_type TEXT, status TEXT, mode TEXT,
            risk_level TEXT, risk_score REAL, coverage_percent REAL, strong_matches INTEGER,
            possible_paraphrases INTEGER, false_positives INTEGER, candidate_count INTEGER,
            summary TEXT DEFAULT '', recommendations TEXT DEFAULT '[]')""")
        connection.execute(
            "INSERT INTO analyses (id, filename) VALUES ('old', 'old.txt')"
        )
    store = AnalysisStore(path)
    with sqlite3.connect(path) as connection:
        assert (
            json.loads(
                connection.execute(
                    "SELECT verification_metadata FROM analyses WHERE id = 'old'"
                ).fetchone()[0]
            )
            == {}
        )
    assert store.get("old")["filename"] == "old.txt"
