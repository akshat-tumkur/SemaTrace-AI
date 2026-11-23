"""Transactional local vector generations. No provider work occurs at import time.

Source snapshots remain authoritative. This store contains derived vectors only;
neither submitted queries nor credentials are persisted. Readers own their query
handle, while published generations and their vector rows are immutable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Callable, Literal

import numpy as np
from urllib.error import HTTPError, URLError

from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.retrieval.embeddings import EmbeddingIndex, _safe_error

SCHEMA_VERSION = 1
PREPROCESSING = "literal-passage-and-query-v1"


def canonical(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class IndexUnavailable(RuntimeError):
    """A missing, incompatible or corrupt generation must be handled explicitly."""


class ProviderBuildFailure(IndexUnavailable):
    def __init__(self, message, *, metrics=None):
        super().__init__(message)
        self.metrics = metrics or {}


class IndexSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    provider: Literal["openai", "deterministic-hash-fallback", "local-onnx"]
    model: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=200)
    dimensions: int | None = Field(default=None, gt=0, le=65536)
    preprocessing: Literal[
        "literal-passage-and-query-v1", "minilm-onnx-mean-l2-256-v1"
    ] = PREPROCESSING
    schema_version: Literal[1] = SCHEMA_VERSION

    @classmethod
    def hash(cls, dimensions=256):
        return cls(
            provider="deterministic-hash-fallback",
            model="sha256-token-hash",
            revision="v1",
            dimensions=dimensions,
        )

    @classmethod
    def openai(cls, model=None, revision=None, dimensions=None):
        return cls(
            provider="openai",
            model=model or config.EMBEDDING_MODEL,
            revision=revision
            or os.getenv("SEMATRACE_EMBEDDING_REVISION", "provider-alias-unpinned"),
            dimensions=dimensions,
        )

    @classmethod
    def local(cls):
        from app.retrieval.local_model import (
            model_lock,
            PREPROCESSING as LOCAL_PREPROCESSING,
        )

        lock = model_lock()
        return cls(
            provider="local-onnx",
            model=lock["model"],
            revision=lock["revision"],
            dimensions=lock["dimensions"],
            preprocessing=LOCAL_PREPROCESSING,
        )


def validate_vectors(vectors, count, dimensions):
    if not isinstance(vectors, (list, tuple)) or len(vectors) != count:
        raise IndexUnavailable("vector_count_mismatch")
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions <= 0
    ):
        raise IndexUnavailable("invalid_vector_dimensions")
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or len(vector) != dimensions:
            raise IndexUnavailable("vector_shape_mismatch")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in vector
        ):
            raise IndexUnavailable("invalid_vector_coordinate")


def _decoded_vector(payload, dimensions):
    """Decode one verified row; never retain JSON payloads and boxed floats globally."""
    vector = json.loads(payload)
    validate_vectors([vector], 1, dimensions)
    return tuple(vector)


def corpus_identity(corpus):
    info = getattr(corpus, "corpus_info", {})
    version = info.get("version")
    members = info.get("source_version_ids")
    if (
        not version
        or not isinstance(members, list)
        or version != "corpus-v1-" + sha(canonical(sorted(members)))
    ):
        raise IndexUnavailable("managed_corpus_identity_required")
    ids = [d.passage_id for d in corpus.documents]
    if len(ids) != len(set(ids)) or any(
        p != sha(d.text) for p, d in zip(ids, corpus.documents)
    ):
        raise IndexUnavailable("invalid_passage_ids")
    # Binds ordering, occurrence aliases, exact offsets and canonical source ownership.
    fingerprint = sha(canonical([asdict(d) for d in corpus.documents]))
    return version, ids, fingerprint


@dataclass(frozen=True)
class Generation:
    id: str
    manifest: dict
    vectors: np.ndarray
    documents: tuple


class GenerationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.connect() as c:
                if c.execute("PRAGMA user_version").fetchone()[0] not in (
                    0,
                    SCHEMA_VERSION,
                ):
                    raise IndexUnavailable("unsupported_retrieval_schema")
                c.executescript("""
                    CREATE TABLE IF NOT EXISTS vectors(
                        space TEXT NOT NULL, passage_id TEXT NOT NULL, vector_hash TEXT NOT NULL,
                        payload TEXT NOT NULL, PRIMARY KEY(space, passage_id));
                    CREATE TABLE IF NOT EXISTS generations(
                        generation_id TEXT PRIMARY KEY, manifest TEXT NOT NULL,
                        request_key TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                    CREATE TABLE IF NOT EXISTS active_generations(
                        corpus_version TEXT NOT NULL, request_key TEXT NOT NULL,
                        generation_id TEXT NOT NULL REFERENCES generations(generation_id), error TEXT,
                        PRIMARY KEY(corpus_version, request_key));
                    PRAGMA user_version=1;
                """)
        except sqlite3.Error as e:
            raise IndexUnavailable("retrieval_database_unavailable") from e

    @contextmanager
    def connect(self):
        try:
            c = sqlite3.connect(self.path, timeout=20)
        except sqlite3.Error as e:
            raise IndexUnavailable("retrieval_database_unavailable") from e
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            with c:
                yield c
        except sqlite3.Error as e:
            raise IndexUnavailable("retrieval_database_unavailable") from e
        finally:
            c.close()

    @staticmethod
    def key(spec):
        return sha(canonical(spec.model_dump()))

    def _vector(self, c, space, passage, expected=None, dimensions=None):
        row = c.execute(
            "SELECT * FROM vectors WHERE space=? AND passage_id=?", (space, passage)
        ).fetchone()
        return self._checked_vector(row, expected, dimensions)

    @staticmethod
    def _checked_vector(row, expected=None, dimensions=None):
        if row is None:
            if expected is not None:
                raise IndexUnavailable("generation_vector_missing")
            return None
        if sha(row["payload"]) != row["vector_hash"] or (
            expected is not None and row["vector_hash"] != expected
        ):
            raise IndexUnavailable("generation_vector_hash_mismatch")
        if dimensions is None:
            raise IndexUnavailable("vector_space_dimensions_required")
        try:
            return _decoded_vector(row["payload"], dimensions)
        except (ValueError, TypeError) as e:
            raise IndexUnavailable("invalid_vector_serialization") from e

    def load(self, generation_id, corpus, *, expected_spec=None):
        try:
            with self.connect() as c:
                c.execute("BEGIN")
                row = c.execute(
                    "SELECT manifest FROM generations WHERE generation_id=?",
                    (generation_id,),
                ).fetchone()
                if row is None:
                    raise IndexUnavailable("generation_missing")
                raw = row[0]
                if generation_id != "retrieval-v1-" + sha(raw):
                    raise IndexUnavailable("generation_manifest_hash_mismatch")
                m = json.loads(raw)
                spec = IndexSpec.model_validate(m["spec"])
                version, ids, fingerprint = corpus_identity(corpus)
                if (
                    m["schema_version"] != SCHEMA_VERSION
                    or m["corpus_version"] != version
                    or m["passage_ids"] != ids
                    or m["source_fingerprint"] != fingerprint
                    or len(m["vector_hashes"]) != len(ids)
                    or spec.dimensions is None
                ):
                    raise IndexUnavailable("generation_corpus_alignment_mismatch")
                if expected_spec and not self.compatible(spec, expected_spec):
                    raise IndexUnavailable("generation_space_mismatch")
                if m["space"] != self.key(spec):
                    raise IndexUnavailable("generation_space_identity_mismatch")
                # Float64 preserves every persisted coordinate exactly. Streaming
                # avoids a corpus-sized Python-float graph and decoded-JSON cache.
                vectors = np.empty((len(ids), spec.dimensions), dtype=np.float64)
                positions = {p: i for i, p in enumerate(ids)}
                # One SELECT per row repeatedly crosses the Windows/WSL file
                # boundary for SQLite locking/metadata. Bounded IN queries keep
                # row streaming and all integrity checks without that N+1 cost.
                for start in range(0, len(ids), 512):
                    batch = ids[start : start + 512]
                    rows = c.execute(
                        "SELECT passage_id,vector_hash,payload FROM vectors WHERE space=? AND passage_id IN ("
                        + ",".join("?" for _ in batch) + ")",
                        (m["space"], *batch),
                    )
                    found = 0
                    for row in rows:
                        i = positions[row["passage_id"]]
                        vectors[i] = self._checked_vector(
                            row, m["vector_hashes"][i], spec.dimensions
                        )
                        found += 1
                    if found != len(batch):
                        raise IndexUnavailable("generation_vector_missing")
                vectors.flags.writeable = False
                return Generation(generation_id, m, vectors, tuple(corpus.documents))
        except (ValueError, KeyError, TypeError) as e:
            raise IndexUnavailable("invalid_generation_serialization") from e

    @staticmethod
    def compatible(actual, requested):
        return actual.model_dump(exclude={"dimensions"}) == requested.model_dump(
            exclude={"dimensions"}
        ) and (
            requested.dimensions is None or actual.dimensions == requested.dimensions
        )

    def active(self, corpus, spec):
        version, _, _ = corpus_identity(corpus)
        with self.connect() as c:
            row = c.execute(
                "SELECT generation_id,error FROM active_generations WHERE corpus_version=? AND request_key=?",
                (version, self.key(spec)),
            ).fetchone()
        if not row:
            return None, None
        # A fallback pointer may deliberately serve an unavailable learned space.
        generation = self.load(row[0], corpus)
        actual = IndexSpec.model_validate(generation.manifest["spec"])
        if not self.compatible(actual, spec) and not (
            actual == IndexSpec.hash() and row[1]
        ):
            raise IndexUnavailable("active_generation_space_mismatch")
        return generation, row[1]

    def activate(self, generation_id, corpus, spec, *, error=None):
        g = self.load(generation_id, corpus)
        actual = IndexSpec.model_validate(g.manifest["spec"])
        if not self.compatible(actual, spec) and not (
            actual == IndexSpec.hash() and error
        ):
            raise IndexUnavailable("activation_space_mismatch")
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO active_generations VALUES(?,?,?,?)",
                (g.manifest["corpus_version"], self.key(spec), g.id, error),
            )

    def build(
        self,
        corpus,
        spec,
        *,
        api_key="",
        embedder: Callable | None = None,
        batch_size=64,
        max_requests=64,
        activate=True,
    ):
        """Embed missing passages only, then commit cache + manifest + pointer together.

        Provider calls happen before the write transaction. Concurrent builders can
        duplicate provider work, but cannot expose incomplete generations. No retry
        is implicit; callers must explicitly budget/approve an additional build.
        """
        started = time.perf_counter()
        version, ids, fingerprint = corpus_identity(corpus)
        if not 1 <= batch_size <= 256 or max_requests < 0:
            raise ValueError("invalid_build_budget")
        if spec.provider == "deterministic-hash-fallback" and (
            spec.dimensions is None or spec != IndexSpec.hash(spec.dimensions)
        ):
            raise IndexUnavailable("invalid_hash_space")
        if spec.provider == "local-onnx" and spec != IndexSpec.local():
            raise IndexUnavailable("unverified_local_model_spec")
        if spec.provider == "local-onnx" and batch_size > 64:
            raise IndexUnavailable("local_batch_limit")
        actual = spec
        # Infer unspecified dimensions only from a previously validated compatible
        # generation or the first new provider batch, then pin them throughout.
        if actual.dimensions is None:
            with self.connect() as c:
                rows = c.execute(
                    "SELECT generation_id,manifest FROM generations WHERE request_key=? ORDER BY rowid DESC",
                    (self.key(spec),),
                ).fetchall()
            for row in rows:
                if row[0] != "retrieval-v1-" + sha(row[1]):
                    raise IndexUnavailable("reuse_manifest_hash_mismatch")
                try:
                    prior = IndexSpec.model_validate(json.loads(row[1])["spec"])
                except (ValueError, KeyError, TypeError) as e:
                    raise IndexUnavailable("invalid_reuse_manifest") from e
                if self.compatible(prior, spec):
                    actual = prior
                    break
        vectors = (
            np.empty((len(ids), actual.dimensions), dtype=np.float64)
            if actual.dimensions is not None else None
        )
        present=bytearray(len(ids))
        if actual.dimensions is not None:
            with self.connect() as c:
                for i, passage in enumerate(ids):
                    vector = self._vector(
                        c, self.key(actual), passage, dimensions=actual.dimensions
                    )
                    if vector is not None:
                        vectors[i]=vector
                        present[i]=1
        reused = sum(present)
        missing = [i for i, exists in enumerate(present) if not exists]
        client = EmbeddingIndex.__new__(EmbeddingIndex)
        client.api_key, client.model, client.dimensions = (
            api_key,
            spec.model,
            spec.dimensions or 256,
        )
        calls = 0
        tokens = 0
        usage_available = False
        local_batches = 0
        for start in range(0, len(missing), batch_size):
            indexes = missing[start : start + batch_size]
            texts = [corpus.documents[i].text for i in indexes]
            try:
                if spec.provider == "deterministic-hash-fallback":
                    batch = [client._hash_embedding(t) for t in texts]
                elif spec.provider == "local-onnx":
                    from app.retrieval.local_model import (
                        local_encoder,
                        LocalModelUnavailable,
                    )

                    try:
                        batch = (embedder or local_encoder().encode)(texts)
                    except LocalModelUnavailable as e:
                        raise ProviderBuildFailure(
                            "local_encoder_unavailable:" + type(e).__name__
                        ) from e
                    local_batches += 1
                else:
                    if calls >= max_requests:
                        raise ProviderBuildFailure("embedding_build_request_budget")
                    if not api_key and embedder is None:
                        raise ProviderBuildFailure("embedding_build_missing_key")
                    calls += 1
                    client.last_response_usage = None
                    try:
                        batch = (embedder or client._openai_embeddings)(texts)
                    finally:
                        usage = getattr(client, "last_response_usage", {}) or {}
                        if isinstance(usage.get("total_tokens"), int):
                            tokens += usage["total_tokens"]
                            usage_available = True
                dimensions = actual.dimensions or (len(batch[0]) if batch else 0)
                validate_vectors(batch, len(indexes), dimensions)
                actual = actual.model_copy(update={"dimensions": dimensions})
                if vectors is None:
                    vectors=np.empty((len(ids),dimensions),dtype=np.float64)
                for i, vector in zip(indexes, batch):
                    vectors[i] = vector
            except (
                HTTPError,
                URLError,
                OSError,
                HTTPException,
                TimeoutError,
                ValueError,
                TypeError,
                IndexError,
                KeyError,
                IndexUnavailable,
            ) as e:
                raise ProviderBuildFailure(
                    "embedding_build_failed:" + _safe_error(e),
                    metrics={
                        "physical_embedding_requests": calls,
                        "total_tokens": tokens if usage_available else None,
                        "failed_build_seconds": time.perf_counter() - started,
                    },
                ) from e
        if actual.dimensions is None:
            raise IndexUnavailable("cannot_infer_empty_index_dimensions")
        if vectors is None:
            vectors=np.empty((0,actual.dimensions),dtype=np.float64)
        if vectors.shape!=(len(ids),actual.dimensions) or not np.isfinite(vectors).all():
            raise IndexUnavailable('invalid_generation_matrix')
        space = self.key(actual)
        manifest = dict(
            schema_version=SCHEMA_VERSION,
            corpus_version=version,
            passage_ids=ids,
            source_fingerprint=fingerprint,
            spec=actual.model_dump(),
            space=space,
            vector_hashes=[sha(canonical(v.tolist())) for v in vectors],
        )
        raw = canonical(manifest)
        generation_id = "retrieval-v1-" + sha(raw)
        # No persisted source content, submitted text, provider response bodies or keys.
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            for passage, vector in zip(ids, vectors):
                payload=canonical(vector.tolist())
                existing = self._vector(c, space, passage, dimensions=actual.dimensions)
                if existing is not None and canonical(existing) != payload:
                    raise IndexUnavailable("concurrent_vector_conflict")
                c.execute(
                    "INSERT OR IGNORE INTO vectors VALUES(?,?,?,?)",
                    (space, passage, sha(payload), payload),
                )
            c.execute(
                "INSERT OR IGNORE INTO generations(generation_id,manifest,request_key) VALUES(?,?,?)",
                (generation_id, raw, self.key(spec)),
            )
            if activate:
                c.execute(
                    "INSERT OR REPLACE INTO active_generations VALUES(?,?,?,NULL)",
                    (version, self.key(spec), generation_id),
                )
        # The published loader owns its matrix. Release the builder's before
        # loading, so ingestion never retains two corpus-wide representations.
        if ids:
            del vector
        del vectors
        g = self.load(generation_id, corpus, expected_spec=spec)
        metrics = dict(
            new_vectors=len(missing),
            reused_vectors=reused,
            physical_embedding_requests=calls,
            total_tokens=tokens if usage_available else None,
            elapsed_seconds=time.perf_counter() - started,
        )
        if spec.provider == "local-onnx":
            metrics["local_encode_batches"] = local_batches
        return g, metrics

    def acquire(
        self,
        corpus,
        *,
        api_key=None,
        model=None,
        revision=None,
        spec=None,
        allow_local_build=False,
    ):
        key = config.OPENAI_API_KEY if api_key is None else api_key
        selected = os.getenv("SEMATRACE_EMBEDDING_PROVIDER", "auto")
        if spec is None and selected == "local-onnx":
            spec = IndexSpec.local()
        elif spec is None and selected == "hash":
            spec = IndexSpec.hash()
        elif spec is None and selected not in {"auto", "openai"}:
            raise IndexUnavailable("unknown_embedding_provider")
        spec = spec or (
            IndexSpec.openai(model, revision)
            if key and corpus.documents
            else IndexSpec.hash()
        )
        generation, error = self.active(corpus, spec)
        if spec.provider == "local-onnx" and error:
            raise IndexUnavailable("local_request_cannot_use_fallback_generation")
        metrics = dict(
            new_vectors=0,
            reused_vectors=len(corpus.documents),
            physical_embedding_requests=0,
        )
        if generation is None:
            if spec.provider == "local-onnx" and not allow_local_build:
                raise IndexUnavailable(
                    "local_generation_missing_explicit_build_required"
                )
            try:
                generation, metrics = self.build(corpus, spec, api_key=key)
            except ProviderBuildFailure as e:
                if spec.provider == "local-onnx":
                    raise  # Local requests fail explicitly, never become silent hash results.
                error = str(e)
                generation, metrics = self.build(corpus, IndexSpec.hash())
                metrics.update(e.metrics)
                metrics["failed_build"] = True
                self.activate(generation.id, corpus, spec, error=error)
        return PersistentEmbeddingIndex(
            corpus, self, generation, api_key=key, error=error, metrics=metrics
        )


class PersistentEmbeddingIndex(EmbeddingIndex):
    """A per-investigation query handle over immutable corpus vectors."""

    def __init__(
        self, corpus, store, generation, *, api_key="", error=None, metrics=None
    ):
        # Never call EmbeddingIndex.__init__: it embeds the entire corpus.
        if (
            getattr(corpus, "corpus_info", {}).get("version")
            != generation.manifest["corpus_version"]
            or tuple(corpus.documents) != generation.documents
        ):
            raise IndexUnavailable("query_handle_corpus_mismatch")
        self.corpus, self.store = corpus, store
        self.api_key = api_key
        self.metrics = metrics or {}
        self.generation = generation
        self.model = generation.manifest["spec"]["model"]
        self.provider = generation.manifest["spec"]["provider"]
        self.dimensions = (
            generation.manifest["spec"]["dimensions"]
            if self.provider == "deterministic-hash-fallback"
            else 256
        )
        self._vectors = generation.vectors
        self.last_error = error
        self._query_cache = {}
        self._corpus_vectors = {
            d.text: v for d, v in zip(corpus.documents, generation.vectors)
        }

    def _query_embeddings(self, texts):
        if len(texts) != 1:
            raise ValueError("query_handle_expects_one_passage")
        query = texts[0]
        if query in self._corpus_vectors:
            return [self._corpus_vectors[query].tolist()]
        if query in self._query_cache:
            return [self._query_cache[query]]
        if self.provider == "local-onnx":
            from app.retrieval.local_model import local_encoder, LocalModelUnavailable

            try:
                result = local_encoder().encode(texts)
                validate_vectors(result, 1, len(self._vectors[0]))
            except (LocalModelUnavailable, IndexUnavailable) as e:
                raise IndexUnavailable("local_query_failed:" + type(e).__name__) from e
            self.metrics["local_query_batches"] = (
                self.metrics.get("local_query_batches", 0) + 1
            )
            if len(self._query_cache) < 128:
                self._query_cache[query] = tuple(result[0])
            return result
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        self.metrics["query_embedding_requests"] = (
            self.metrics.get("query_embedding_requests", 0) + 1
        )
        result = super()._query_embeddings(texts)
        try:
            validate_vectors(result, 1, len(self._vectors[0]))
        except IndexUnavailable as e:
            raise ValueError("query_vector_invalid") from e
        if len(self._query_cache) < 128:
            self._query_cache[query] = tuple(result[0])
        return result

    def _use_hash_fallback(self, error):
        # Query failure never modifies/pollutes a published learned generation or
        # another reader. Switch this entire handle to a validated hash generation.
        generation, _ = self.store.active(self.corpus, IndexSpec.hash())
        if generation is None:
            generation, _ = self.store.build(
                self.corpus, IndexSpec.hash(), activate=False
            )
        self._vectors = generation.vectors
        self.provider = "deterministic-hash-fallback"
        self.dimensions = 256
        self.generation = generation
        self.last_error = _safe_error(error)
        self._query_cache.clear()
        self._corpus_vectors = {}

    def search(self, query, limit=3):
        results = super().search(query, limit)
        return [
            replace(
                m,
                generation_id=self.generation.id,
                vector_space_id=self.generation.manifest["space"],
            )
            for m in results
        ]
