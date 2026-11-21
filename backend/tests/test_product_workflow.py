"""Product engineering contracts; artificial provider replies are not accuracy evidence."""

import copy, json, time


import numpy as np


import pytest


from test_detector_api import local_api, TARGET


from app.product.review import ground, review_saved


from app.product.providers import ProviderUnavailable


from app.product.jobs import JobStore


from app.retrieval.bm25 import CorpusDocument, LocalCorpus


def wait_job(client, key):
    for _ in range(500):
        result = client.get("/api/analysis-jobs/" + key).json()
        if result["status"] != "running":
            return result
        time.sleep(0.02)
    raise AssertionError("Job did not finish")


def test_real_local_upload_job_persists_complete_report_and_can_reload(
    local_api, monkeypatch
):
    client, main, _ = local_api
    monkeypatch.setenv("OPENAI_API_KEY", "configured-but-never-call-without-consent")

    def forbidden(*args, **kwargs):
        raise AssertionError("Local-only mode must not invoke paid providers")

    monkeypatch.setattr("app.product.providers.OpenAIReviewer", forbidden)
    content = (
        "🙂  " + TARGET + "\nSource: https://example.org/reference.html?revision=2"
    ).encode()
    response = client.post(
        "/api/analysis-jobs",
        files={"file": ("product.txt", content, "text/plain")},
        data={"cloud_review": "false", "web_search": "false"},
    )
    assert response.status_code == 202, response.text
    job = wait_job(client, response.json()["job_id"])
    assert job["status"] == "completed", job
    saved = client.get("/api/analysis/" + job["analysis_id"]).json()
    assert (
        saved["semantic_assessment"]["status"] == "disabled"
        and saved["web_discovery"]["status"] == "disabled"
    )
    assert saved["findings"] and saved["document"]["original_text"] == content.decode()
    assert any(u["text"].endswith("reference.html?revision=2") for u in saved["units"])
    assert (
        client.get("/api/analysis-jobs/" + job["job_id"]).json()["analysis_id"]
        == saved["analysis_id"]
    )
    assert (
        client.get("/api/analysis/" + saved["analysis_id"] + "/report").json()[
            "processing"
        ]["job_id"]
        == job["job_id"]
    )
    assert client.delete("/api/analysis/" + saved["analysis_id"]).status_code == 200
    assert client.get("/api/analysis-jobs/" + job["job_id"]).status_code == 404


def test_bad_document_job_has_explicit_error_and_no_successful_report(local_api):
    client, main, _ = local_api
    response = client.post("/api/analysis-jobs", files={"file": ("empty.txt", b"")})
    job = wait_job(client, response.json()["job_id"])
    assert job["status"] == "failed" and "empty" in job["error"].lower()
    assert not main.analysis_store.list_reports()


def test_restart_recovery_marks_pending_jobs_failed_not_completed(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.create("job", "sample.txt", {"cloud_review": False})
    store.recover()
    job = store.get("job")
    assert (
        job["status"] == "failed"
        and job["analysis_id"] is None
        and "restarted" in job["error"]
    )


def test_open_initialized_corpus_does_not_take_write_lock_during_snapshot(tmp_path):
    from app.storage.source_corpus import SourceCorpusStore

    path = tmp_path / "corpus.sqlite3"
    store = SourceCorpusStore(path)
    with store.connect() as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM sources").fetchall()
        # A concurrent reader in SQLite's rollback mode prevents write commit.
        # Request-scoped initialization must not write schema/version again.
        another = SourceCorpusStore(path)
        assert another.list_sources() == []


def test_grounding_requires_exact_unambiguous_quotes_and_never_adds_coverage():
    unit = {
        "submission_start": 10,
        "submission_excerpt": "The backup replica takes over after the main server fails.",
    }
    candidate = {
        "candidate_id": "x",
        "source_document_id": "fixed",
        "source_context_start": 100,
        "source_context": "A secondary database replaces the primary when its server stops responding.",
    }
    item = {
        "relation": "PARAPHRASE",
        "submission_quote": "backup replica takes over",
        "source_quote": "secondary database replaces the primary",
        "explanation": "Both describe failover to a replica.",
        "differences": [],
    }
    finding = ground(item, unit, candidate)
    assert (
        finding["source_start"] == 102
        and finding["submission_start"] == 14
        and finding["score_eligible"] is False
    )
    assert not finding["independently_validated"]
    for bad in [
        dict(item, source_quote="a fabricated source sentence"),
        dict(item, submission_quote="backup"),
        dict(item, explanation="Confirmed plagiarism occurred."),
    ]:
        with pytest.raises(ValueError):
            ground(bad, unit, candidate)
    assert ground(dict(item, relation="RELATED_ONLY"), unit, candidate) is None


def test_ai_provider_failure_preserves_local_metric_and_surfaces_unavailable(
    local_api, monkeypatch
):
    client, _, store = local_api
    saved = client.post(
        "/api/analyze", files={"file": ("test.txt", TARGET.encode())}
    ).json()
    before = copy.deepcopy(saved["similarity_report"])

    class Failed:
        calls = 1
        usage = {}
        actual_model = "offline-test"

        def ask(self, *args):
            raise ProviderUnavailable("Test provider unavailable")

    result = review_saved(saved, store, provider=Failed())
    assert (
        result["status"] == "failed" and result["warnings"] and not result["findings"]
    )
    assert saved["similarity_report"] == before


def test_bm25_postings_preserve_reference_scores_duplicate_terms_and_tie_order():
    corpus = LocalCorpus(
        [
            CorpusDocument(str(i), str(i), "local://source", s)
            for i, s in enumerate(
                ["water water rivers", "water lakes", "mountain snow", "water lakes"]
            )
        ]
    )
    for query in ["water", "water water lakes", "unknown", "snow rivers", ""]:
        tokens = __import__("app.retrieval.bm25", fromlist=["_tokenize"])._tokenize(
            query
        )
        expected = sorted(
            [(corpus._score(tokens, t), i) for i, t in enumerate(corpus._tokens)],
            reverse=True,
        )
        actual = corpus.search(query, 100)
        assert [(m.source_id, m.score) for m in actual] == [
            (str(i), round(score, 4)) for score, i in expected if score > 0
        ]


def test_web_fetch_rejects_loopback_private_addresses_and_credentials(monkeypatch):
    from app.product.web import public_target, UnsafeSource
    import socket

    for ip in ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1", "::ffff:127.0.0.1"]:
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *args, **kwargs: [(0, 0, 0, "", (ip, 443))]
        )
        with pytest.raises(UnsafeSource):
            public_target("https://source.example/page")
    for url in [
        "file:///etc/passwd",
        "http://user:pass@example.org",
        "https://example.org:8080",
    ]:
        with pytest.raises(UnsafeSource):
            public_target(url)


def test_submission_url_segmentation_preserves_offsets_and_original_source_segmentation():
    from app.parsing.sentences import split_into_units

    text = "See https://example.org/file.html?revision=2. Next sentence.\nLicense: https://example.org/licenses/4.0/"
    units = split_into_units(text, preserve_urls=True)
    assert (
        len(units) == 3
        and units[0].text == "See https://example.org/file.html?revision=2."
    )
    assert all(text[u.start_char : u.end_char] == u.text for u in units)
    assert len(split_into_units(text)) > len(units)


def test_finished_ai_review_preserves_prior_provenance_without_another_call(local_api):
    client, _, store = local_api
    saved = client.post(
        "/api/analyze", files={"file": ("test.txt", TARGET.encode())}
    ).json()
    starts = sorted({r["submission_start"] for r in saved["related_sources"]})
    previous = {
        "reviewed_unit_starts": starts,
        "model": "actual-snapshot",
        "calls": 2,
        "usage": {"total_tokens": 500},
        "review_runs": [{"model": "actual-snapshot", "calls": 2}],
        "warnings": ["Earlier validation warning"],
    }

    class Forbidden:
        def ask(self, *args):
            raise AssertionError("No further call is needed")

    result = review_saved(saved, store, previous, Forbidden())
    assert result["remaining_units"] == 0 and result["model"] == "actual-snapshot"
    assert (
        result["usage"] == previous["usage"]
        and result["review_runs"] == previous["review_runs"]
    )
    assert result["warnings"] == previous["warnings"]


def test_unfittable_context_is_skipped_without_false_review_or_repeated_calls(
    local_api, monkeypatch
):
    client, _, store = local_api
    saved = client.post(
        "/api/analyze", files={"file": ("test.txt", TARGET.encode())}
    ).json()
    monkeypatch.setattr("app.product.review.context_for", lambda *args: None)

    class Forbidden:
        calls = 0
        usage = {}
        actual_model = "test"

        def ask(self, *args):
            raise AssertionError("No context should be sent")

    result = review_saved(saved, store, provider=Forbidden())
    assert (
        result["skipped_units"]
        and not result["reviewed_unit_starts"]
        and result["remaining_units"] == 0
    )
    assert result["calls"] == 0 and result["warnings"]
    assert (
        review_saved(saved, store, result, Forbidden())["skipped_units"]
        == result["skipped_units"]
    )


def test_advisory_ai_metadata_survives_json_pdf_without_changing_word_coverage(
    local_api,
):
    client, main, _ = local_api
    saved = client.post(
        "/api/analyze", files={"file": ("test.txt", TARGET.encode())}
    ).json()
    assessment = {
        "status": "complete",
        "findings": [],
        "method": "grounded-openai-passage-review-v1",
        "reviewed_unit_starts": [0],
        "eligible_units": 1,
        "remaining_units": 0,
        "model": "test-snapshot",
        "calls": 1,
        "usage": {"total_tokens": 123},
    }
    assert main.analysis_store.save_product_section(
        saved["analysis_id"], "semantic_assessment", assessment
    )
    exported = client.get("/api/analysis/" + saved["analysis_id"] + "/report").json()
    assert (
        exported["semantic_assessment"] == assessment
        and exported["similarity_report"] == saved["similarity_report"]
    )
    import pymupdf

    response = client.get("/api/analysis/" + saved["analysis_id"] + "/report.pdf")
    with pymupdf.open(stream=response.content, filetype="pdf") as pdf:
        text = "\n".join(page.get_text() for page in pdf)
        assert "AI meaning and paraphrase review" in text and "test-snapshot" in text
