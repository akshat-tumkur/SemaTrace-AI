"""Offline integrity/compute tests, never semantic-accuracy evidence."""

from concurrent.futures import ThreadPoolExecutor


from http.client import RemoteDisconnected


import importlib


import json


import math


import subprocess


import socket


import sys


import threading


from pathlib import Path


from unittest.mock import Mock


from fastapi.testclient import TestClient


import pytest


from app.corpus.models import SourceMetadata, CorpusUnavailable


from app.retrieval.bm25 import LocalCorpus, _cached_statistics


from app.retrieval.embeddings import EmbeddingIndex


from app.retrieval.generations import (
    GenerationStore,
    IndexSpec,
    IndexUnavailable,
    PersistentEmbeddingIndex,
    ProviderBuildFailure,
)


from app.storage.database import AnalysisStore


from app.storage.source_corpus import SourceCorpusStore


TEXT = "The Cedar reservoir stores winter rainfall in covered tanks for the dry season."


OTHER = (
    "Solar modules convert sunlight into electricity in the laboratory demonstration."
)


SPEC = IndexSpec.openai("offline-test-model", "fixture-v1")


def metadata(key="one", revision="1"):
    return SourceMetadata(
        collection="test",
        external_id=key,
        title=key,
        url="https://example.org/" + key,
        revision=revision,
        extractor_version="test-v1",
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    blocked = Mock(side_effect=AssertionError("Milestone 2 tests forbid live services"))
    monkeypatch.setattr("app.retrieval.embeddings.urlopen", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    original = socket.socket.connect

    def external_only(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise AssertionError("No external connections in offline tests")
        return original(sock, address)

    monkeypatch.setattr("socket.socket.connect", external_only)


@pytest.fixture
def sources(tmp_path):
    store = SourceCorpusStore(tmp_path / "sources.db")
    store.ingest(metadata(), TEXT + "\n" + TEXT + "\n" + OTHER, b"original")
    return store


def fake(texts):
    # Explicit artificial normalized vectors. Same shape as a provider response;
    # this function is NOT a learned semantic embedding.
    return [
        [1.0, 0.0, 0.0] if "reservoir" in text else [0.0, 1.0, 0.0] for text in texts
    ]


def handle(corpus, store, generation, **kwargs):
    return PersistentEmbeddingIndex(
        corpus, store, generation, api_key="offline", **kwargs
    )


def test_restart_reuses_generation_without_corpus_calls(sources, monkeypatch):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    provider = Mock(side_effect=fake)
    g, metrics = store.build(corpus, SPEC, embedder=provider)
    assert metrics["new_vectors"] == 2 and provider.call_count == 1
    monkeypatch.setattr(
        EmbeddingIndex,
        "_openai_embeddings",
        Mock(side_effect=AssertionError("No corpus recomputation")),
    )
    reopened = GenerationStore(store.path).acquire(corpus, api_key="offline", spec=SPEC)
    assert reopened.generation.id == g.id
    assert reopened.metrics["new_vectors"] == 0
    assert (
        reopened.search(TEXT)[0].source_text == TEXT
    )  # exact query uses corpus vector
    assert reopened.metrics.get("query_embedding_requests", 0) == 0


def test_real_new_process_loads_hash_generation(sources):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    g, _ = store.build(corpus, IndexSpec.hash())
    code = "from app.storage.source_corpus import SourceCorpusStore; from app.retrieval.generations import GenerationStore; import sys; c=SourceCorpusStore(sys.argv[1]).snapshot(); print(GenerationStore(sys.argv[2]).acquire(c,api_key='').generation.id)"
    result = subprocess.run(
        [sys.executable, "-c", code, str(sources.path), str(store.path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == g.id


def test_incremental_refresh_reuses_by_passage_not_occurrence(sources):
    old = sources.snapshot()
    store = GenerationStore(old.retrieval_index_path)
    g, _ = store.build(old, SPEC, embedder=fake)
    new_text = (
        TEXT
        + "\nA calibrated probe measures the lake temperature during each winter inspection."
    )
    sources.ingest(metadata(revision="2"), new_text, b"next original")
    sources.ingest(metadata("duplicate-origin"), TEXT, b"another source")
    new = sources.snapshot()
    provider = Mock(side_effect=fake)
    next_g, metrics = store.build(new, SPEC, embedder=provider)
    assert (
        next_g.id != g.id
        and metrics["new_vectors"] == 1
        and metrics["reused_vectors"] == 1
    )
    assert provider.call_args.args[0] == [
        d.text for d in new.documents if d.text != TEXT
    ]
    assert store.load(g.id, old).id == g.id
    with pytest.raises(IndexUnavailable):
        store.load(g.id, new)
    matches = handle(new, store, next_g).search(TEXT)
    match = next(m for m in matches if m.source_text == TEXT)
    assert len(match.source_occurrences) == 2
    for o in match.source_occurrences:
        assert (
            new.source_documents[o.source_document_id][
                o.source_start_char : o.source_end_char
            ]
            == match.source_text
        )


def test_model_revision_and_dimension_spaces_never_reuse(sources):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    first, _ = store.build(corpus, SPEC, embedder=fake)
    changed = SPEC.model_copy(update={"revision": "fixture-v2"})
    second, metrics = store.build(corpus, changed, embedder=fake)
    assert (
        first.manifest["space"] != second.manifest["space"]
        and metrics["new_vectors"] == 2
    )
    with pytest.raises(IndexUnavailable, match="space_mismatch"):
        store.load(first.id, corpus, expected_spec=changed)
    dimensions = SPEC.model_copy(update={"dimensions": 2})
    third, metrics = store.build(
        corpus, dimensions, embedder=lambda texts: [[1.0, 0.0] for t in texts]
    )
    assert (
        third.manifest["space"] != first.manifest["space"]
        and metrics["reused_vectors"] == 0
    )


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[]],
        [[1.0]],
        [[1.0, 0.0], []],
        [[math.nan], [1.0]],
        [[math.inf], [1.0]],
        [["1"], [1.0]],
        [[True], [1.0]],
    ],
)
def test_invalid_provider_never_publishes_partial_generation(sources, bad):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    with pytest.raises(ProviderBuildFailure):
        store.build(corpus, SPEC, embedder=lambda texts: bad)
    with store.connect() as c:
        for table in ("vectors", "generations", "active_generations"):
            assert c.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_interrupted_batched_build_commits_nothing(sources):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    provider = Mock(side_effect=[[[1.0, 0.0, 0.0]], TimeoutError()])
    with pytest.raises(ProviderBuildFailure):
        store.build(corpus, SPEC, embedder=provider, batch_size=1)
    assert store.active(corpus, SPEC) == (None, None)
    assert store.build(corpus, SPEC, embedder=fake)[1]["new_vectors"] == 2


def test_sql_publication_failure_rolls_back_all_rows(sources):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    with store.connect() as c:
        c.execute(
            "CREATE TRIGGER fail_publish BEFORE INSERT ON active_generations BEGIN SELECT RAISE(ABORT,'offline injected failure'); END"
        )
    with pytest.raises(IndexUnavailable):
        store.build(corpus, IndexSpec.hash())
    with store.connect() as c:
        assert c.execute("SELECT count(*) FROM vectors").fetchone()[0] == 0
        assert c.execute("SELECT count(*) FROM generations").fetchone()[0] == 0
        c.execute("DROP TRIGGER fail_publish")
    assert store.build(corpus, IndexSpec.hash())[0].id


@pytest.mark.parametrize(
    "corruption", ["manifest", "vector", "missing_vector", "vector_hash"]
)
def test_corruption_fails_explicitly_without_rebuilding(sources, corruption):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    g, _ = store.build(corpus, IndexSpec.hash())
    with store.connect() as c:
        if corruption == "manifest":
            c.execute("UPDATE generations SET manifest='{}'")
        elif corruption == "vector":
            c.execute("UPDATE vectors SET payload='[1]'")
        elif corruption == "vector_hash":
            c.execute("UPDATE vectors SET vector_hash='wrong'")
        else:
            c.execute("DELETE FROM vectors")
    with pytest.raises(IndexUnavailable):
        store.acquire(corpus, api_key="")


def test_query_failure_falls_back_atomically_and_stays_there(sources, monkeypatch):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    learned, _ = store.build(corpus, SPEC, embedder=fake)
    first = handle(corpus, store, learned)
    unaffected = handle(corpus, store, learned)
    provider = Mock(side_effect=TimeoutError())
    monkeypatch.setattr(EmbeddingIndex, "_openai_embeddings", provider)
    assert first.search("reservoir query not identical to corpus")
    assert first.provider == "deterministic-hash-fallback"
    assert first.search("reservoir query not identical to corpus")
    assert provider.call_count == 1
    assert unaffected.provider == "openai" and unaffected.generation.id == learned.id
    assert store.load(learned.id, corpus).manifest["spec"]["provider"] == "openai"


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[]],
        [[1.0, 0.0]],
        [[math.nan, 0.0, 0.0]],
        [[True, 0.0, 0.0]],
        [["bad", 0.0, 0.0]],
    ],
)
def test_invalid_query_vectors_never_compare_to_learned_corpus(
    sources, monkeypatch, bad
):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    g, _ = store.build(corpus, SPEC, embedder=fake)
    index = handle(corpus, store, g)
    monkeypatch.setattr(EmbeddingIndex, "_openai_embeddings", lambda self, texts: bad)
    index.search("different reservoir query")
    assert index.provider == "deterministic-hash-fallback"
    assert index.generation.manifest["spec"]["provider"] == index.provider


def test_provider_fallback_recovery_needs_explicit_complete_build(sources, monkeypatch):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    provider = Mock(side_effect=TimeoutError())
    monkeypatch.setattr(EmbeddingIndex, "_openai_embeddings", provider)
    first = store.acquire(corpus, api_key="offline", spec=SPEC)
    assert first.provider == "deterministic-hash-fallback"
    assert (
        first.metrics["physical_embedding_requests"] == 1
        and first.metrics["failed_build"]
    )
    monkeypatch.setattr(
        EmbeddingIndex,
        "_openai_embeddings",
        Mock(side_effect=AssertionError("sticky persisted fallback")),
    )
    assert (
        store.acquire(corpus, api_key="offline", spec=SPEC).generation.id
        == first.generation.id
    )
    restored, metrics = store.build(corpus, SPEC, embedder=fake)
    assert metrics["new_vectors"] == 2
    fresh = store.acquire(corpus, api_key="offline", spec=SPEC)
    assert fresh.provider == "openai" and fresh.generation.id == restored.id
    assert first.provider == "deterministic-hash-fallback"


def test_equal_dimensions_do_not_make_hash_and_learned_compatible(sources, monkeypatch):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    hashed, _ = store.build(corpus, IndexSpec.hash(3))
    learned, _ = store.build(corpus, SPEC, embedder=fake)
    assert hashed.manifest["space"] != learned.manifest["space"]
    index = handle(corpus, store, hashed)
    # Hash dimensions follow its published space, not the default fallback size.
    assert index.generation.manifest["spec"]["dimensions"] == 3
    assert index.search(TEXT) == index.search(TEXT)
    with pytest.raises(IndexUnavailable):
        store.load(hashed.id, corpus, expected_spec=SPEC)


def test_rollback_and_concurrent_reader_pinning(sources):
    old = sources.snapshot()
    store = GenerationStore(old.retrieval_index_path)
    g, _ = store.build(old, IndexSpec.hash())
    reader = store.acquire(old, api_key="")
    sources.ingest(
        metadata("new"),
        "A newly added source contains a sufficiently long independent observation.",
        b"new",
    )
    new = sources.snapshot()
    started = threading.Event()
    finish = threading.Event()

    def embed(texts):
        started.set()
        assert finish.wait(5)
        return fake(texts)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(store.build, new, SPEC, embedder=embed)
        assert started.wait(5)
        assert reader.search(TEXT)[0].source_text == TEXT
        assert store.active(new, SPEC) == (None, None)
        finish.set()
        new_g, _ = future.result()
    assert reader.generation.id == g.id and store.load(new_g.id, new)
    store.activate(g.id, old, IndexSpec.hash())
    assert store.active(old, IndexSpec.hash())[0].id == g.id


def test_query_cache_is_in_memory_and_space_specific(sources, monkeypatch):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    g, _ = store.build(corpus, SPEC, embedder=fake)
    index = handle(corpus, store, g)
    provider = Mock(side_effect=lambda self, texts: fake(texts))
    monkeypatch.setattr(
        EmbeddingIndex, "_openai_embeddings", lambda self, texts: provider(self, texts)
    )
    query = "reservoir query not stored as a source"
    assert index.search(query) == index.search(query)
    assert provider.call_count == 1
    reopened = handle(corpus, store, g)
    reopened.search(query)
    assert provider.call_count == 2
    with store.connect() as c:
        assert all(
            query not in r[0] for r in c.execute("SELECT manifest FROM generations")
        )


def test_bm25_reuses_statistics_but_keeps_integrity_checks(sources):
    _cached_statistics.cache_clear()
    first = sources.snapshot()
    second = sources.snapshot()
    assert first is not second and first._tokens is second._tokens
    assert first.search("winter rainfall") == second.search("winter rainfall")
    with sources.connect() as c:
        c.execute("UPDATE occurrences SET start_char=100")
    with pytest.raises(CorpusUnavailable):
        sources.snapshot()


def test_corrupt_api_generation_returns_503(
    sources, monkeypatch, tmp_path, provision_local_index
):
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "local_corpus", LocalCorpus([]))
    monkeypatch.setattr(main, "analysis_store", AnalysisStore(tmp_path / "analyses.db"))
    main.app.dependency_overrides[main.get_source_store] = lambda: sources
    provision_local_index(sources.snapshot())
    try:
        with TestClient(main.app) as client:
            assert (
                client.post(
                    "/api/analyze",
                    files={"file": ("copy.txt", TEXT.encode(), "text/plain")},
                ).status_code
                == 202
            )
            store = GenerationStore(sources.snapshot().retrieval_index_path)
            with store.connect() as c:
                c.execute("UPDATE vectors SET payload='bad'")
            assert (
                client.post(
                    "/api/analyze",
                    files={"file": ("copy.txt", TEXT.encode(), "text/plain")},
                ).status_code
                == 503
            )
    finally:
        main.app.dependency_overrides.clear()


def test_empty_corpus_hash_generation_is_reusable(tmp_path):
    corpus = SourceCorpusStore(tmp_path / "empty.db").snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    first = store.acquire(corpus, api_key="")
    second = store.acquire(corpus, api_key="")
    assert first.search("query") == [] and first.generation.id == second.generation.id


def test_passage_order_and_literal_identity_are_validated(sources):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    g, _ = store.build(corpus, IndexSpec.hash())
    corpus.documents.reverse()
    with pytest.raises(IndexUnavailable, match="alignment"):
        store.load(g.id, corpus)
    corpus.documents[0] = corpus.documents[0].__class__(
        "bad",
        "bad",
        "local://bad",
        "Changed target text",
        passage_id=corpus.documents[0].passage_id,
    )
    with pytest.raises(IndexUnavailable, match="passage_ids"):
        store.load(g.id, corpus)


def test_build_budget_has_no_partial_publication(sources):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    provider = Mock(side_effect=fake)
    with pytest.raises(ProviderBuildFailure):
        store.build(corpus, SPEC, embedder=provider, batch_size=1, max_requests=1)
    assert provider.call_count == 1 and store.active(corpus, SPEC) == (None, None)


@pytest.mark.parametrize("change", ["model", "coordinate", "index"])
def test_raw_provider_serialization_cannot_hide_space_or_numeric_errors(
    sources, monkeypatch, change
):
    payload = {
        "model": "offline-test-model",
        "data": [
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
        ],
    }
    if change == "model":
        payload["model"] = "different-model"
    elif change == "coordinate":
        payload["data"][0]["embedding"][0] = True
    else:
        payload["data"][0]["index"] = False
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(payload).encode()
    monkeypatch.setattr("app.retrieval.embeddings.urlopen", Mock(return_value=response))
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    with pytest.raises(ProviderBuildFailure):
        store.build(corpus, SPEC, api_key="offline")


def test_cli_paid_build_requires_explicit_permission(monkeypatch):
    from app.retrieval.index_cli import main

    with pytest.raises(SystemExit) as error:
        main(["build", "--provider", "openai"])
    assert error.value.code == 2


@pytest.mark.parametrize("failure", [ConnectionResetError(), RemoteDisconnected()])
def test_transport_disconnect_cannot_partially_convert_reader(
    sources, monkeypatch, failure
):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    g, _ = store.build(corpus, SPEC, embedder=fake)
    index = handle(corpus, store, g)
    monkeypatch.setattr(EmbeddingIndex, "_openai_embeddings", Mock(side_effect=failure))
    assert index.search("new reservoir query")
    assert index.provider == "deterministic-hash-fallback"


def test_dimensions_cannot_change_between_build_batches(sources):
    corpus = sources.snapshot()
    store = GenerationStore(corpus.retrieval_index_path)
    provider = Mock(side_effect=[[[1.0, 0.0, 0.0]], [[1.0, 0.0]]])
    with pytest.raises(ProviderBuildFailure):
        store.build(corpus, SPEC, embedder=provider, batch_size=1)
    assert store.active(corpus, SPEC) == (None, None)


def test_query_handle_rejects_another_corpus(sources):
    old = sources.snapshot()
    store = GenerationStore(old.retrieval_index_path)
    g, _ = store.build(old, IndexSpec.hash())
    sources.ingest(
        metadata("another"),
        "A new source has a separate passage with its own original occurrence.",
        b"another",
    )
    with pytest.raises(IndexUnavailable, match="query_handle_corpus_mismatch"):
        handle(sources.snapshot(), store, g)
