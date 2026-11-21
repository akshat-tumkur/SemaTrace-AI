"""Grounded semantic journeys, score isolation and real optional CPU inference."""

import copy, os


from contextlib import closing


from pathlib import Path


import pytest


from test_detector_api import local_api, TARGET


from app.discovery.evidence import verify_row, selected_rows


@pytest.mark.parametrize("action", ["rerank", "investigate"])
def test_semantic_writes_are_rate_limited(action):
    from starlette.responses import JSONResponse
    from fastapi.testclient import TestClient
    from app.limits import RequestLimits

    async def inner(scope, receive, send):
        await JSONResponse({"ok": True})(scope, receive, send)

    with closing(TestClient(RequestLimits(inner, requests_per_minute=1))) as client:
        path = "/api/analysis/example/semantic/" + action
        assert client.post(path).status_code == 200
        assert client.post(path).status_code == 429


def upload(client):
    result = client.post(
        "/api/analyze",
        files={
            "file": ("semantic.txt", ("🙂  " + TARGET + "\r\n").encode(), "text/plain")
        },
    )
    assert result.status_code == 202, result.text
    return result.json()


def test_automatic_discovery_offsets_and_optional_missing_model(local_api, monkeypatch):
    client, main, store = local_api
    saved = upload(client)
    rows = saved["related_sources"]
    assert rows
    for row in rows:
        assert row["score_eligible"] is False and row["score_is_probability"] is False
        assert (
            saved["document"]["original_text"][
                row["original_submission_start"] : row["original_submission_end"]
            ]
            == row["original_submission_excerpt"]
        )
        verify_row(saved, row, store)
    identifier = saved["analysis_id"]
    candidate = rows[0]["candidate_id"]
    monkeypatch.delenv("SEMATRACE_REVIEW_MODEL_DIR", raising=False)
    response = client.post(
        f"/api/analysis/{identifier}/semantic/rerank", json={"candidate_id": candidate}
    )
    assert response.status_code == 503
    assert (
        client.get(f"/api/analysis/{identifier}").json()["similarity_report"]
        == saved["similarity_report"]
    )
    assert (
        client.post(
            f"/api/analysis/{identifier}/semantic/rerank",
            json={"candidate_id": "0" * 64},
        ).status_code
        == 422
    )
    tampered = copy.deepcopy(rows[0])
    tampered["source_excerpt"] += "invented"
    with pytest.raises(ValueError):
        verify_row(saved, tampered, store)


def test_real_optional_onnx_saved_reuse_and_score_isolation(local_api, monkeypatch):
    root = Path(__file__).resolve().parents[2] / ".local/models/roberta-review-v1"
    if not (root / "review-manifest.json").is_file():
        pytest.skip("Private trained review artifact intentionally not distributed")
    monkeypatch.setenv("SEMATRACE_REVIEW_MODEL_DIR", str(root.resolve()))
    client, main, store = local_api
    saved = upload(client)
    identifier = saved["analysis_id"]
    candidate = saved["related_sources"][0]["candidate_id"]
    path = f"/api/analysis/{identifier}/semantic/rerank"
    r = client.post(path, json={"candidate_id": candidate})
    assert r.status_code == 200, r.text
    assert r.json()["score_eligible"] is False
    reloaded = client.get(f"/api/analysis/{identifier}").json()
    assert reloaded["similarity_report"] == saved["similarity_report"]
    assert reloaded["semantic_review"]["rankings"]
    from app.discovery import evidence

    monkeypatch.setattr(
        evidence,
        "rerank",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Do not rerun saved ranking")
        ),
    )
    assert client.post(path, json={"candidate_id": candidate}).json() == r.json()


def test_real_graph_conditional_tools_and_ungrounded_quote_rejection(local_api):
    client, main, store = local_api
    saved = upload(client)
    candidate = saved["related_sources"][0]["candidate_id"]
    from app.discovery.investigation import investigate

    class Advisor:
        model = "branch-regression"

        def ask(self, instruction, payload, schema, tokens):
            if "action" in schema["properties"]:
                return {
                    "action": "context" if not payload["executed"] else "alternatives",
                    "reason": "Test both actual tools",
                }
            return {
                "candidate_id": candidate,
                "submission_quote": "fabricated quote",
                "source_quote": "fabricated quote",
                "similarities": "invented",
                "differences": "none",
                "uncertainty": "none",
            }

    original = copy.deepcopy(saved["similarity_report"])
    result = investigate(saved, candidate, store, Advisor())
    assert [t["tool"] for t in result["trace"] if "tool" in t] == [
        "context",
        "alternatives",
    ]
    assert result["explanation"]["status"] == "REJECTED_UNGROUNDED_OUTPUT"
    assert result["score_eligible"] is False and saved["similarity_report"] == original


def test_graph_grounded_contract_and_timeout_preserve_tools(local_api):
    client, main, store = local_api
    saved = upload(client)
    candidate = saved["related_sources"][0]["candidate_id"]
    from app.discovery.investigation import investigate, InvestigationUnavailable

    class Advisor:
        model = "contract-regression"
        fail = False

        def ask(self, instruction, payload, schema, tokens):
            if "action" in schema["properties"]:
                return {"action": "explain", "reason": "Ready to compare"}
            if self.fail:
                raise InvestigationUnavailable("timeout")
            return {
                "candidate_id": "0",
                "submission_quote": payload["submission"][:20],
                "source_quote": payload["evidence"][0]["source"][:20],
                "similarities": "Overlapping experimental setting.",
                "differences": "The measurement year differs.",
                "uncertainty": "Independent writing remains possible.",
            }

    advisor = Advisor()
    result = investigate(saved, candidate, store, advisor)
    assert result["explanation"]["status"] == "GROUNDED_QUOTES_UNVERIFIED_EXPLANATION"
    assert (
        result["explanation"]["source_document_id"]
        == saved["related_sources"][0]["source_document_id"]
    )
    assert any(t.get("tool") == "context" for t in result["trace"])
    advisor.fail = True
    partial = investigate(saved, candidate, store, advisor)
    assert partial["explanation"]["status"] == "LOCAL_EXPLANATION_UNAVAILABLE"
    assert partial["contexts"] and any(
        t.get("tool") == "context" for t in partial["trace"]
    )


def test_investigation_uses_saved_generation_after_library_growth(local_api):
    client, _, store = local_api
    saved = upload(client)
    candidate = saved["related_sources"][0]["candidate_id"]
    from app.corpus.models import SourceMetadata
    from app.retrieval.generations import GenerationStore, IndexSpec
    from app.discovery.investigation import investigate

    text = (
        "A distinct added reference describes a new thermal energy storage experiment."
    )
    store.ingest(
        SourceMetadata(
            collection="authorized-test",
            external_id="new-after-report",
            title="New reference",
            url="https://example.org/new",
            revision="1",
            extractor_version="text-v1",
            rights_status="PRIVATE_UPLOAD",
        ),
        text,
        text.encode(),
    )
    new = store.snapshot()
    generation, _ = GenerationStore(new.retrieval_index_path).build(
        new, IndexSpec.local(), api_key=""
    )
    assert generation.id != saved["retrieval"]["generation_id"]

    class Advisor:
        model = "historical-regression"

        def ask(self, instruction, payload, schema, tokens):
            if "action" in schema["properties"]:
                return {
                    "action": "alternatives" if not payload["executed"] else "explain",
                    "reason": "Inspect competitors",
                }
            return {
                "candidate_id": "0",
                "submission_quote": payload["submission"][:20],
                "source_quote": payload["evidence"][0]["source"][:20],
                "similarities": "Related evidence.",
                "differences": "Details differ.",
                "uncertainty": "Independent writing is possible.",
            }

    result = investigate(saved, candidate, store, Advisor())
    details = next(
        t["details"] for t in result["trace"] if t.get("tool") == "alternatives"
    )
    assert (
        details["retrieved"] > 0 and details["snapshot"] == saved["corpus"]["version"]
    )
    assert (
        client.get("/api/analysis/" + saved["analysis_id"]).json()["similarity_report"]
        == saved["similarity_report"]
    )
