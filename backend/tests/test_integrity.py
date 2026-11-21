"""Offline regressions for the approved production integrity corrections."""

from itertools import permutations


import json


import math


from unittest.mock import MagicMock


import pytest


from app.parsing.sentences import split_into_units


from app.retrieval.bm25 import CorpusDocument, LocalCorpus


from app.retrieval.corpus import add_source_to_corpus, load_local_corpus


from app.retrieval.embeddings import EmbeddingIndex


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("No live API calls in Gate 0.1")

    monkeypatch.setattr("app.retrieval.embeddings.urlopen", blocked)
    monkeypatch.setattr("app.product.web.fetch_public", blocked)


def corpus():
    return LocalCorpus(
        [
            CorpusDocument("a", "A", "local://a", "alpha evidence"),
            CorpusDocument("b", "B", "local://b", "beta control"),
        ]
    )


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[1.0]],
        [[1.0], []],
        [[1.0], [1.0, 0.0]],
        [[math.nan], [1.0]],
        [[math.inf], [1.0]],
        [["bad"], [1.0]],
    ],
)
def test_malformed_corpus_vectors_fall_back_as_complete_index(monkeypatch, bad):
    calls = []

    def provider(self, texts):
        calls.append(texts)
        return bad

    monkeypatch.setattr(EmbeddingIndex, "_openai_embeddings", provider)
    index = EmbeddingIndex(corpus(), api_key="offline")
    assert index.provider == "deterministic-hash-fallback" and len(index._vectors) == 2
    assert {len(v) for v in index._vectors} == {256}
    assert index.search("alpha evidence")[0].source_id == "a"
    assert len(calls) == 1  # Sticky fallback makes no further provider requests.


@pytest.mark.parametrize(
    "bad",
    [[], [[]], [[1.0]], [[math.nan, 0.0]], [[math.inf, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
)
def test_malformed_query_triggers_whole_index_fallback(monkeypatch, bad):
    responses = iter([[[1.0, 0.0], [0.0, 1.0]], bad])
    monkeypatch.setattr(
        EmbeddingIndex, "_openai_embeddings", lambda self, texts: next(responses)
    )
    index = EmbeddingIndex(corpus(), api_key="offline")
    assert index.provider == "openai"
    assert index.search("alpha evidence")[0].source_id == "a"
    assert index.provider == "deterministic-hash-fallback"
    assert {len(v) for v in index._vectors} == {256}
    assert index.search("alpha evidence")[0].source_id == "a"


def test_new_index_is_the_only_recovery_path(monkeypatch):
    count = []

    def provider(self, texts):
        count.append(texts)
        if len(count) == 1:
            raise TimeoutError()
        return [
            [1.0, 0.0] if text == "alpha evidence" else [0.0, 1.0] for text in texts
        ]

    monkeypatch.setattr(EmbeddingIndex, "_openai_embeddings", provider)
    old = EmbeddingIndex(corpus(), api_key="offline")
    old.search("alpha evidence")
    old.search("alpha evidence")
    assert len(count) == 1 and old.provider == "deterministic-hash-fallback"
    new = EmbeddingIndex(corpus(), api_key="offline")
    assert new.provider == "openai" and new.search("alpha evidence")[0].source_id == "a"
    assert old.provider == "deterministic-hash-fallback"


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        ([1.0], []),
        ([1.0], [1.0, 0.0]),
        ([math.nan], [1.0]),
        ([1.0], [math.inf]),
    ],
)
def test_cosine_rejects_invalid_vectors(left, right):
    with pytest.raises(ValueError):
        EmbeddingIndex._cosine(left, right)


def test_similarity_formula_and_empty_corpus_are_preserved(monkeypatch):
    assert EmbeddingIndex._cosine([0.6, 0.8], [1.0, 0.0]) == 0.6
    assert EmbeddingIndex._cosine([-1.0, 0.0], [1.0, 0.0]) == 0
    monkeypatch.setattr(EmbeddingIndex, "_openai_embeddings", lambda self, texts: [])
    assert EmbeddingIndex(LocalCorpus([]), api_key="offline").search("query") == []


@pytest.mark.parametrize("dimensions", [0, -1, True])
def test_invalid_hash_dimensions_rejected_before_provider_request(dimensions):
    with pytest.raises(ValueError, match="hash_embedding_dimensions"):
        EmbeddingIndex(corpus(), dimensions=dimensions, api_key="offline")


def test_duplicate_or_missing_provider_indices_are_rejected(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]}
    ).encode()
    monkeypatch.setattr(
        "app.retrieval.embeddings.urlopen", MagicMock(return_value=response)
    )
    index = EmbeddingIndex(corpus(), api_key="offline")
    assert index.provider == "deterministic-hash-fallback"


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Dr. Iyer measured 12.5 volts. Next claim follows.",
            ["Dr. Iyer measured 12.5 volts.", "Next claim follows."],
        ),
        (
            "The U.S. laboratory measured 3.14 volts.",
            ["The U.S. laboratory measured 3.14 volts."],
        ),
        (
            "The team works in the U.S. Next we test it.",
            ["The team works in the U.S.", "Next we test it."],
        ),
        (
            "The U.S. Navy uses calibrated sensors.",
            ["The U.S. Navy uses calibrated sensors."],
        ),
        (
            "J. R. Rao measured 12.5 volts. Prof. García repeated it.",
            ["J. R. Rao measured 12.5 volts.", "Prof. García repeated it."],
        ),
        (
            "  α🙂 measured 12.5 volts!\nβ repeated the observation?",
            ["α🙂 measured 12.5 volts!", "β repeated the observation?"],
        ),
        (
            "First claim. Second claim! Third claim?",
            ["First claim.", "Second claim!", "Third claim?"],
        ),
        (
            "Data arrived on 17 November 2026. Values were 0.25 and 17.0.",
            ["Data arrived on 17 November 2026.", "Values were 0.25 and 17.0."],
        ),
    ],
)
def test_narrow_boundaries_preserve_original_slices(text, expected):
    units = split_into_units(text)
    assert [u.text for u in units] == expected
    assert all(text[u.start_char : u.end_char] == u.text for u in units)


def test_source_loader_retains_decoded_leading_whitespace_and_old_constructors(
    tmp_path,
):
    text = "  \nA complete source sentence preserves its original position.\n"
    (tmp_path / "source.txt").write_bytes(text.encode("utf-8"))
    corpus_index = load_local_corpus(tmp_path)
    d = corpus_index.documents[0]
    assert d.source_id == "source-1" and d.source_start_char == 3
    assert corpus_index.source_documents[d.source_document_id] == text
    assert (
        CorpusDocument(
            "legacy", "Title", "local://legacy", "Old text"
        ).source_start_char
        is None
    )


def test_source_api_count_and_stored_provenance_roundtrip(monkeypatch, tmp_path):
    import importlib
    from fastapi.testclient import TestClient
    from app.storage.database import AnalysisStore

    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "local_corpus", LocalCorpus([]))
    monkeypatch.setattr(main, "analysis_store", AnalysisStore(tmp_path / "gate01.db"))
    claim = "Dr. Iyer measured exactly 12.5 volts during the calibrated laboratory experiment."
    text = (
        claim
        + " The same laboratory then repeated its measurement for the control group."
    )
    with TestClient(main.app) as client:
        indexed = client.post(
            "/api/sources",
            data={"title": "Lab", "url": "https://example.invalid/lab"},
            files={"file": ("lab.txt", text.encode(), "text/plain")},
        )
        assert indexed.status_code == 201 and indexed.json()["units_indexed"] == 2
        identifier = indexed.json()["source_id"]
        assert sorted(
            d.source_id for d in main.get_source_store().snapshot().documents
        ) == [identifier + "-1", identifier + "-4"]
        response = client.post(
            "/api/analyze", files={"file": ("copy.txt", claim.encode(), "text/plain")}
        )
        assert response.status_code == 202
        record = response.json()
        stored = client.get("/api/analysis/" + record["analysis_id"]).json()
        match = stored["matches"][0]
        assert match["source_id"] == identifier + "-1"
        assert match["source_start_char"] == 0 and match["source_end_char"] == len(
            claim.rstrip(".")
        )
        original = client.get(
            "/api/corpus/versions/" + match["source_document_id"]
        ).json()["text"]
        assert (
            original[match["source_start_char"] : match["source_end_char"]]
            == match["source_text"]
        )
        assert stored["verification"] == record["verification"]
