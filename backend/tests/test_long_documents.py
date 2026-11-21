"""Whole long uploads retain final-sentence evidence and bounded retrieval batches."""

import numpy as np


import pytest


from test_detector_api import local_api, TARGET, A, B


from app.retrieval.hybrid import HybridRetriever, QUERY_BATCH_SIZE


from app.parsing.sentences import split_into_units


from app.similarity import words


@pytest.mark.parametrize("prefix_count", [128, 998])
def test_real_upload_over_100_units_is_complete_and_saved(local_api, prefix_count):
    client, main, store = local_api
    prefix = (
        "\n".join(
            f"Note {i}."
            if prefix_count == 998
            else f"Entry {i}: The ceramic exhibit uses glazed tiles in the eastern gallery."
            for i in range(prefix_count)
        )
        + "\n"
    )
    original = prefix + TARGET
    r = client.post(
        "/api/analyze",
        files={"file": ("long-report.txt", original.encode(), "text/plain")},
    )
    assert r.status_code == 202, r.text
    saved = r.json()
    identifier = saved["analysis_id"]
    text = saved["document"]["text"]
    count = len(split_into_units(text))
    assert (
        count == prefix_count + 2
        and saved["units_analyzed"] == count == saved["retrieval"]["query_count"]
    )
    assert saved["retrieval"]["query_batch_size"] == QUERY_BATCH_SIZE
    assert saved["document"]["original_text"] == original
    assert {row["unit_index"] for row in saved["related_sources"]} == set(range(count))
    assert saved["findings"] and all(
        f["submission_start"] >= len(prefix) for f in saved["findings"]
    )
    assert saved["similarity_report"]["unique_matched_words"] == len(words(A)) + len(
        words(B)
    )
    final_candidates = [
        r for r in saved["related_sources"] if r["unit_index"] == count - 1
    ]
    assert final_candidates
    for row in final_candidates:
        assert (
            original[row["original_submission_start"] : row["original_submission_end"]]
            == row["original_submission_excerpt"]
        )
    reloaded = client.get("/api/analysis/" + identifier).json()
    assert (
        reloaded["findings"] == saved["findings"]
        and reloaded["related_sources"] == saved["related_sources"]
    )
    assert reloaded["similarity_report"] == saved["similarity_report"]


def test_1000_queries_stream_in_bounded_batches_without_dropping_units(local_api):
    _, _, store = local_api
    corpus = store.snapshot()

    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(list(texts))
            return [
                vectors[int(text.rsplit(" ", 1)[1]) % len(vectors)] for text in texts
            ]

    vectors = np.eye(384, dtype=np.float32)[: len(corpus.documents)]
    encoder = Encoder()
    retriever = HybridRetriever(corpus, vectors=vectors, encoder=encoder)
    queries = [f"query {i}" for i in range(1000)]
    result = retriever.rank(queries)
    assert [text for batch in encoder.calls for text in batch] == queries
    assert len(encoder.calls) == 32 and max(map(len, encoder.calls)) == 32
    assert {r["unit_index"] for r in result["related"]} == set(range(1000))
    for row in result["related"]:
        if row["minilm_rank"] == 1:
            assert (
                row["source_excerpt"]
                == corpus.documents[row["unit_index"] % len(corpus.documents)].text
            )
            assert row["score"] == pytest.approx(1.0)
    assert result["query_count"] == 1000 and result["documents"]


def test_late_retrieval_batch_failure_does_not_save_a_partial_report(
    local_api, monkeypatch
):
    client, main, _ = local_api
    from app.retrieval.hybrid import local_encoder

    encoder = local_encoder()
    original = encoder.encode
    calls = []

    def failed(texts):
        calls.append(len(texts))
        if len(calls) == 2:
            return np.full((len(texts), 384), np.nan)
        return original(texts)

    monkeypatch.setattr(encoder, "encode", failed)
    text = "\n".join(f"Note {i}." for i in range(130))
    response = client.post(
        "/api/analyze", files={"file": ("long-report.txt", text.encode(), "text/plain")}
    )
    assert response.status_code == 503 and calls == [32, 32]
    assert main.analysis_store.list_reports() == []
