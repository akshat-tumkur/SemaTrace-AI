"""Local provider engineering tests. Artificial vectors are not accuracy labels."""

import json


import math


import socket


from unittest.mock import Mock


import pytest


from app.corpus.models import SourceMetadata


from app.retrieval.bm25 import LocalCorpus


from app.retrieval.generations import (
    GenerationStore,
    IndexSpec,
    IndexUnavailable,
    ProviderBuildFailure,
)


from app.retrieval.local_model import (
    local_encoder,
    LocalModelUnavailable,
    verify_artifact,
    model_lock,
    default_directory,
)


from app.storage.source_corpus import SourceCorpusStore


TEXT = "The reservoir collects winter rainfall in covered tanks for the dry season."


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.embeddings.urlopen",
        Mock(side_effect=AssertionError("No paid embeddings")),
    )
    monkeypatch.setattr(
        socket, "create_connection", Mock(side_effect=AssertionError("No network"))
    )
    original = socket.socket.connect

    def deny(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise AssertionError("No network")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", deny)


@pytest.fixture
def sources(tmp_path):
    s = SourceCorpusStore(tmp_path / "sources.db")
    meta = SourceMetadata(
        collection="test",
        external_id="one",
        title="Cedar",
        url="https://example.org/cedar",
        revision="1",
        extractor_version="v1",
    )
    s.ingest(meta, TEXT, b"original")
    return s, meta


def vector(texts):
    return [[1.0] + [0.0] * 383 for t in texts]


def test_explicit_build_only_then_restart_reuse(sources, monkeypatch):
    s, _ = sources
    c = s.snapshot()
    store = GenerationStore(c.retrieval_index_path)
    monkeypatch.setenv("SEMATRACE_EMBEDDING_PROVIDER", "local-onnx")
    with pytest.raises(IndexUnavailable, match="explicit_build_required"):
        store.acquire(c, api_key="")
    provider = Mock()
    provider.encode.side_effect = vector
    monkeypatch.setattr("app.retrieval.local_model.local_encoder", lambda: provider)
    first = store.acquire(c, api_key="", allow_local_build=True)
    assert first.provider == "local-onnx" and provider.encode.call_count == 1
    reopened = GenerationStore(store.path).acquire(s.snapshot(), api_key="")
    assert (
        reopened.generation.id == first.generation.id
        and provider.encode.call_count == 1
    )
    reopened.search("A differently worded reservoir query")
    assert provider.encode.call_count == 2
    reopened.search("A differently worded reservoir query")
    assert provider.encode.call_count == 2
    assert first.metrics["physical_embedding_requests"] == 0


def test_local_update_reuses_only_compatible_passages(sources):
    s, meta = sources
    c = s.snapshot()
    store = GenerationStore(c.retrieval_index_path)
    g, _ = store.build(c, IndexSpec.local(), embedder=vector)
    s.ingest(
        meta.model_copy(update={"revision": "2"}),
        TEXT
        + "\nA second observation describes the nearby solar array used for irrigation.",
        b"new",
    )
    new = s.snapshot()
    new_g, stats = store.build(new, IndexSpec.local(), embedder=vector)
    assert stats["new_vectors"] == 1 and stats["reused_vectors"] == 1
    with pytest.raises(IndexUnavailable):
        store.load(g.id, new)
    with pytest.raises(IndexUnavailable):
        store.load(new_g.id, new, expected_spec=IndexSpec.hash(384))
    with pytest.raises(IndexUnavailable):
        store.load(new_g.id, new, expected_spec=IndexSpec.openai(dimensions=384))


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "invented"),
        ("dimensions", 256),
        ("preprocessing", "literal-passage-and-query-v1"),
        ("model", "another-model"),
    ],
)
def test_unverified_local_specs_rejected_before_embedding(sources, field, value):
    s, _ = sources
    c = s.snapshot()
    store = GenerationStore(c.retrieval_index_path)
    with pytest.raises(IndexUnavailable, match="unverified_local_model_spec"):
        store.build(
            c, IndexSpec.local().model_copy(update={field: value}), embedder=vector
        )


@pytest.mark.parametrize(
    "bad", [[], [[]], [[1.0] * 256], [[math.nan] * 384], [[True] * 384]]
)
def test_bad_local_vectors_do_not_publish(sources, bad):
    s, _ = sources
    c = s.snapshot()
    store = GenerationStore(c.retrieval_index_path)
    with pytest.raises(ProviderBuildFailure):
        store.build(c, IndexSpec.local(), embedder=lambda t: bad)
    assert store.active(c, IndexSpec.local()) == (None, None)


def test_query_failure_never_silently_falls_back(sources, monkeypatch):
    s, _ = sources
    c = s.snapshot()
    store = GenerationStore(c.retrieval_index_path)
    g, _ = store.build(c, IndexSpec.local(), embedder=vector)
    index = store.acquire(c, api_key="", spec=IndexSpec.local())
    monkeypatch.setattr(
        "app.retrieval.local_model.local_encoder",
        Mock(side_effect=LocalModelUnavailable("missing")),
    )
    with pytest.raises(IndexUnavailable):
        index.search("New semantic wording")
    assert index.provider == "local-onnx" and index.generation.id == g.id


def test_corrupt_local_generation_is_not_rebuilt(sources):
    s, _ = sources
    c = s.snapshot()
    store = GenerationStore(c.retrieval_index_path)
    store.build(c, IndexSpec.local(), embedder=vector)
    with store.connect() as db:
        db.execute("UPDATE vectors SET payload='[]'")
    with pytest.raises(IndexUnavailable):
        store.acquire(c, api_key="", spec=IndexSpec.local())


def test_model_hash_failure_has_no_network_or_substitution():
    entry = model_lock()["files"][0]
    with pytest.raises(LocalModelUnavailable):
        verify_artifact(b"invalid", entry)
    with pytest.raises(LocalModelUnavailable, match="hash_mismatch"):
        verify_artifact(b"x" * entry["size"], entry)


@pytest.mark.parametrize("bad", [[], [[1.0] * 256], [[math.nan] * 384]])
def test_query_vector_mismatch_is_explicit_not_hash(sources, monkeypatch, bad):
    s, _ = sources
    c = s.snapshot()
    store = GenerationStore(c.retrieval_index_path)
    store.build(c, IndexSpec.local(), embedder=vector)
    index = store.acquire(c, api_key="", spec=IndexSpec.local())
    provider = Mock()
    provider.encode.return_value = bad
    monkeypatch.setattr("app.retrieval.local_model.local_encoder", lambda: provider)
    with pytest.raises(IndexUnavailable, match="local_query_failed"):
        index.search("new wording")
    assert index.provider == "local-onnx"


def test_download_failure_does_not_publish_model(tmp_path, monkeypatch):
    from app.retrieval.local_model_setup import download

    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b"wrong bytes"
    monkeypatch.setattr(
        "app.retrieval.local_model_setup.urlopen", lambda *a, **k: response
    )
    target = tmp_path / "model"
    with pytest.raises(LocalModelUnavailable):
        download(target)
    assert not target.exists()


def test_local_cli_rejects_invented_revision_before_publication(sources, monkeypatch):
    from app.retrieval.index_cli import main

    s, _ = sources
    with pytest.raises(SystemExit) as failure:
        main(
            [
                "build",
                "--provider",
                "local-onnx",
                "--revision",
                "invented",
                "--corpus-db",
                str(s.path),
            ]
        )
    assert failure.value.code == 2


@pytest.mark.skipif(
    not default_directory().exists(),
    reason="Optional pinned local model not downloaded",
)
def test_real_cpu_model_lazy_reuse_dimensions_and_normalization():
    pytest.importorskip("onnxruntime")
    m = local_encoder()
    assert local_encoder() is m
    assert m.session.get_providers() == ["CPUExecutionProvider"]
    rows = m.encode(["A cat rests on a mat.", "A dog runs through the field."])
    assert all(
        len(row) == 384 and abs(sum(v * v for v in row) - 1) < 1e-5 for row in rows
    )
    before = m.truncated_passages
    assert (
        len(m.encode(["example " * 300])[0]) == 384
        and m.truncated_passages == before + 1
    )
