from app.storage.database import AnalysisStore


def test_analysis_store_round_trips_workflow_result(tmp_path) -> None:
    store = AnalysisStore(tmp_path / "analysis.db")
    result = {
        "filename": "paper.txt",
        "file_type": "txt",
        "status": "completed",
        "mode": "agentic deterministic workflow",
        "risk_level": "HIGH",
        "risk_score": 0.8,
        "coverage_percent": 42.0,
        "strong_matches": 1,
        "possible_paraphrases": 2,
        "false_positives": 1,
        "candidate_count": 1,
        "units": [{"id": "u1", "text": "A claim.", "start_char": 0, "end_char": 8, "unit_type": "sentence"}],
        "matches": [{"unit_id": "u1", "source_url": "local://source"}],
        "audit_log": [{"agent": "Judge", "event": "issued_verdict", "details": "Risk HIGH"}],
    }

    store.save("a_test", result)
    loaded = store.get("a_test")

    assert loaded is not None
    assert loaded["risk_score"] == 0.8
    assert loaded["units"][0]["id"] == "u1"
    assert loaded["matches"][0]["source_url"] == "local://source"
    assert loaded["audit_log"][0]["agent"] == "Judge"
