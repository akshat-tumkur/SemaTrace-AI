"""Cloud transport, ownership and publication contracts; no paid provider calls.

Postgres integration runs only against an explicitly supplied localhost test DB.
Supabase/Pinecone transports below are protocol doubles, not live certification.
"""

import base64
import hashlib
import json
import os
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
import pytest

from app.cloud.client import CloudUnavailable


def test_report_size_limit_counts_unicode_and_stops_before_oversized_tail(monkeypatch):
    from app.cloud import records

    payload = {"text": "\u6c34\u5faa\u73af \U0001f30d", "values": [1, None, True]}
    exact = len(json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr(records, "REPORT_MAX_BYTES", exact)
    records.enforce_report_limit(payload)
    monkeypatch.setattr(records, "REPORT_MAX_BYTES", exact - 1)
    with pytest.raises(ValueError, match="Report exceeds cloud storage limit"):
        records.enforce_report_limit(payload)
    # An oversized prefix must be rejected before visiting the invalid tail.
    # This also prevents eager serialization of all rejected report evidence.
    with pytest.raises(ValueError, match="Report exceeds cloud storage limit"):
        records.enforce_report_limit({"large": ["x" * 100] * 100, "tail": object()})


class Services:
    def __init__(self):
        self.blobs = {}
        self.vectors = {}
        self.tokens = {}
        self.fail_upsert = False
        self.fail_delete = False
        self.corrupt_query = False
        self.index_ready = True
        self.corrupt_global_fetch = False

    def token(self, owner, expired=False):
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {"sub": owner, "exp": time.time() + (-60 if expired else 3600)}
                ).encode()
            )
            .decode()
            .rstrip("=")
        )
        token = "eyJhbGciOiJIUzI1NiJ9." + payload + ".fixture-signature"
        if not expired:
            self.tokens[token] = owner
        return token

    def handle(self, request):
        path = request.url.path
        data = (
            json.loads(request.content)
            if request.content
            and request.headers.get("content-type", "").startswith("application/json")
            else {}
        )

        def response(status, value):
            return httpx.Response(status, json=value)

        if path == "/auth/v1/user":
            owner = self.tokens.get(
                request.headers.get("authorization", "").removeprefix("Bearer ")
            )
            return (
                response(200, {"id": owner, "is_anonymous": False})
                if owner
                else response(401, {"error": "invalid token"})
            )
        if path.startswith("/storage/v1/object/"):
            if request.method == "POST":
                if path in self.blobs:
                    return response(409, {"error": "Duplicate"})
                self.blobs[path] = request.content
                return response(200, {"Key": path})
            if path in self.blobs:
                return httpx.Response(200, content=self.blobs[path])
            return response(404, {})
        if request.url.host == "api.pinecone.io":
            return response(
                200,
                dict(
                    dimension=384,
                    metric="cosine",
                    host="127.0.0.1:8787",
                    status={"ready": self.index_ready},
                ),
            )
        ns = data.get("namespace") or request.url.params.get("namespace")
        if path == "/vectors/upsert":
            if self.fail_upsert:
                return response(503, {})
            for row in data["vectors"]:
                self.vectors[(ns, row["id"])] = row
            return response(200, {"upsertedCount": len(data["vectors"])})
        if path == "/vectors/fetch":
            vectors = {
                i: self.vectors[(ns, i)]
                for i in request.url.params.get_list("ids")
                if (ns, i) in self.vectors
            }
            if self.corrupt_global_fetch and ns == "global-v1":
                vectors = {
                    i: dict(row, values=[0.5] + [0.0] * 383)
                    for i, row in vectors.items()
                }
            return response(
                200,
                {"vectors": vectors},
            )
        if path == "/vectors/delete":
            if self.fail_delete:
                return response(503, {})
            conditions = data["filter"].get("$and", [data["filter"]])
            fields = {
                k: v["$eq"] for condition in conditions for k, v in condition.items()
            }
            source = fields["source_id"]
            for key, row in list(self.vectors.items()):
                if (
                    key[0] == ns
                    and row["metadata"]["source_id"] == source
                    and (
                        "version_id" not in fields
                        or row["metadata"]["version_id"] == fields["version_id"]
                    )
                ):
                    self.vectors.pop(key)
            return response(200, {})
        if path == "/query":
            versions = data["filter"]["$and"][1]["version_id"]["$in"]
            matches = []
            for (namespace, identifier), row in self.vectors.items():
                if namespace != ns or row["metadata"]["version_id"] not in versions:
                    continue
                meta = dict(row["metadata"])
                if self.corrupt_query:
                    meta["passage_id"] = "0" * 64
                matches.append(
                    dict(
                        id=identifier,
                        metadata=meta,
                        score=sum(a * b for a, b in zip(data["vector"], row["values"])),
                    )
                )
            return response(
                200,
                {
                    "matches": sorted(matches, key=lambda m: (-m["score"], m["id"]))[
                        : data["topK"]
                    ]
                },
            )
        raise AssertionError("Unexpected external request: " + str(request.url))


class Encoder:
    def encode(self, texts):
        # Artificial normalized vectors exercise transport/identity only.
        return [[1.0] + [0.0] * 383 for _ in texts]


@pytest.fixture
def cloud(monkeypatch):
    url = os.getenv("SEMATRACE_TEST_POSTGRES_URL", "")
    if not url:
        pytest.skip("Set SEMATRACE_TEST_POSTGRES_URL to a dedicated localhost database")
    assert urlsplit(url).hostname in {"localhost", "127.0.0.1"}
    monkeypatch.setenv("SEMATRACE_STORAGE_PROVIDER", "supabase")
    monkeypatch.setenv("SEMATRACE_ALLOW_LOCAL_CLOUD_ENDPOINTS", "true")
    monkeypatch.setenv("SUPABASE_DB_URL", url)
    for name, value in dict(
        SUPABASE_URL="http://127.0.0.1:8787",
        SUPABASE_ANON_KEY="fixture-anon",
        SUPABASE_SERVICE_ROLE_KEY="fixture-service",
        PINECONE_INDEX_HOST="http://127.0.0.1:8787",
        PINECONE_INDEX_NAME="fixture",
        PINECONE_API_KEY="fixture-key",
    ).items():
        monkeypatch.setenv(name, value)
    from app.cloud import client
    from app.cloud.auth import _verified

    services = Services()
    transport = httpx.Client(transport=httpx.MockTransport(services.handle))
    monkeypatch.setattr(client, "http_client", lambda: transport)
    _verified.clear()
    from app.cloud.client import connect
    from app.cloud.sources import CloudSourceStore
    from app.cloud.records import CloudAnalysisStore, CloudJobStore
    from app.retrieval import local_model

    monkeypatch.setattr(local_model, "local_encoder", lambda: Encoder())
    # Cloud modules import the same provider explicitly; keep every test offline.
    monkeypatch.setattr("app.cloud.sources.local_encoder", lambda: Encoder())
    monkeypatch.setattr("app.cloud.retrieval.local_encoder", lambda: Encoder())
    owners = [str(uuid4()), str(uuid4())]
    with connect() as c:
        from app.cloud.libraries import GLOBAL_LIBRARY

        c.execute("DELETE FROM st_sources WHERE owner_id=%s", (GLOBAL_LIBRARY,))
        c.execute("DELETE FROM st_passages WHERE owner_id=%s", (GLOBAL_LIBRARY,))
        c.execute("DELETE FROM st_index_state WHERE owner_id=%s", (GLOBAL_LIBRARY,))
        for owner in owners:
            c.execute("INSERT INTO auth.users(id) VALUES(%s)", (owner,))
        c.execute("DELETE FROM st_usage")  # only this dedicated localhost test DB
    yield dict(
        owners=owners,
        services=services,
        sources=[CloudSourceStore(o) for o in owners],
        reports=[CloudAnalysisStore(o) for o in owners],
        jobs=[CloudJobStore(o) for o in owners],
    )
    transport.close()


def meta(revision="1", external="sensor"):
    from app.corpus.models import SourceMetadata

    return SourceMetadata(
        collection="test",
        external_id=external,
        title="CC0 sensor control",
        url="https://example.org/sensor",
        revision=revision,
        extractor_version="test-v1",
        rights_status="PRIVATE_UPLOAD",
    )


TEXT = "The calibrated laboratory sensor recorded seventeen volts throughout the controlled experiment."


def test_modern_supabase_secret_is_not_used_as_a_bearer_jwt(monkeypatch):
    from app.cloud.client import Blobs

    monkeypatch.setenv("SUPABASE_URL", "https://fixture.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_fixture")
    blob = Blobs(str(uuid4()))
    assert blob.headers() == {"apikey": "sb_secret_fixture"}


def test_streamed_model_verification_preserves_pinned_hash_checks(tmp_path):
    from app.retrieval.local_model import verify_file, LocalModelUnavailable

    raw = b"known model bytes" * (100000)
    path = tmp_path / "model.onnx"
    path.write_bytes(raw)
    entry = {"size": len(raw), "lfs": {"sha256": hashlib.sha256(raw).hexdigest()}}
    verify_file(path, entry)
    path.write_bytes(b"x" + raw[1:])
    with pytest.raises(LocalModelUnavailable, match="hash_mismatch"):
        verify_file(path, entry)


def test_compact_vectors_preserve_exact_coordinates_and_serialized_hashes(tmp_path):
    from app.storage.source_corpus import SourceCorpusStore
    from app.retrieval.generations import GenerationStore, IndexSpec, canonical, sha
    import math

    source = SourceCorpusStore(tmp_path / "source.sqlite3")
    source.ingest(meta(), TEXT, TEXT.encode())
    corpus = source.snapshot()
    value = 0.1234567890123456
    vector = [value, math.sqrt(1 - value * value)]
    spec = IndexSpec.openai("fixture", "pinned", 2)
    generation, _ = GenerationStore(corpus.retrieval_index_path).build(
        corpus, spec, embedder=lambda texts: [vector[:] for _ in texts]
    )
    assert generation.vectors[0, 0] == value
    assert generation.manifest["vector_hashes"] == [sha(canonical(vector))]
    with pytest.raises(ValueError):
        generation.vectors[0, 0] = 0


def test_generation_loading_bounds_sql_round_trips_and_preserves_order(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from app.storage.source_corpus import SourceCorpusStore
    from app.retrieval.generations import GenerationStore, IndexSpec
    import numpy as np

    source = SourceCorpusStore(tmp_path / "sources.sqlite3")
    text = "\n".join(
        f"Calibration sample {i} recorded a stable voltage during the controlled laboratory experiment."
        for i in range(600)
    )
    source.ingest(meta(), text, text.encode())
    corpus = source.snapshot()
    assert len(corpus.documents) > 512
    spec = IndexSpec.openai("fixture", "pinned", 2)
    store = GenerationStore(corpus.retrieval_index_path)
    generation, _ = store.build(
        corpus, spec, embedder=lambda texts: [[1.0, 0.0] for _ in texts]
    )
    statements = []
    original = store.connect

    @contextmanager
    def traced():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            yield connection

    monkeypatch.setattr(store, "connect", traced)
    loaded = store.load(generation.id, corpus, expected_spec=spec)
    vector_reads = [
        s for s in statements if s.startswith("SELECT") and "FROM vectors" in s
    ]
    assert len(vector_reads) <= (len(corpus.documents) + 511) // 512
    assert loaded.manifest == generation.manifest
    assert np.array_equal(loaded.vectors, generation.vectors)


def test_cloud_report_and_job_ownership_and_restart(cloud):
    a, b = cloud["reports"]
    ja, jb = cloud["jobs"]
    a.save(
        "report",
        {
            "analysis_id": "report",
            "filename": "private.txt",
            "status": "completed",
            "document": {"text": TEXT},
        },
    )
    assert b.get("report") is None and not b.delete("report") and not b.list_reports()
    assert not b.update_similarity_report("report", {"forged": True})
    a.update_similarity_report("report", {"unique_matched_words": 12})
    assert a.get("report")["similarity_report"]["unique_matched_words"] == 12
    ja.create("completed", "private.txt", {})
    ja.persist_upload("completed", TEXT.encode())
    ja.update("completed", status="completed", analysis_id="report")
    ja.create("interrupted", "pending.txt", {})
    assert jb.get("completed") is None
    from app.cloud.records import CloudJobStore, CloudAnalysisStore

    CloudJobStore.recover()
    assert CloudJobStore(a.owner).get("completed")["status"] == "completed"
    assert CloudJobStore(a.owner).get("interrupted")["status"] == "failed"
    assert CloudAnalysisStore(a.owner).get("report")["document"]["text"] == TEXT


def test_configured_pinecone_host_is_validated(cloud):
    from app.cloud.pinecone import Pinecone

    assert Pinecone(cloud["owners"][0]).validate()["dimension"] == 384
    cloud["services"].index_ready = False
    with pytest.raises(CloudUnavailable):
        Pinecone(cloud["owners"][0]).validate()


def test_report_and_job_completion_commit_together(cloud):
    reports = cloud["reports"][0]
    jobs = cloud["jobs"][0]
    jobs.create("atomic-job", "private.txt", {})
    reports.save(
        "atomic-report",
        {"filename": "private.txt", "processing": {"job_id": "atomic-job"}},
    )
    assert jobs.get("atomic-job")["analysis_id"] == "atomic-report"
    assert jobs.get("atomic-job")["status"] == "completed"
    jobs.update("atomic-job", status="failed", error="late transport error")
    assert jobs.get("atomic-job")["status"] == "completed"


def test_explicit_local_report_migration_preserves_files_and_exclusions(
    cloud, tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path
    import sys
    from app.storage.database import AnalysisStore
    from app.product.jobs import JobStore
    from app.detector import run_detection

    source = cloud["sources"][0]
    source.ingest(meta(), TEXT, TEXT.encode())
    report = run_detection("private.txt", "txt", TEXT, source.snapshot())
    report["analysis_id"] = "migration-report"
    report["similarity_report"]["options"] = {
        "excluded_source_ids": ["retained-explicit-choice"]
    }
    db = tmp_path / "old.sqlite3"
    AnalysisStore(db).save(report["analysis_id"], report)
    jobs = JobStore(db.with_suffix(".jobs.sqlite3"))
    jobs.create("migration-job", "private.txt", {})
    jobs.update("migration-job", status="completed", analysis_id=report["analysis_id"])
    files = [db, jobs.path]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    module_spec = importlib.util.spec_from_file_location(
        "cloud_admin", Path(__file__).resolve().parents[2] / "scripts/cloud_admin.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    env = tmp_path / "private.env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cloud_admin",
            "--env-file",
            str(env),
            "import-reports",
            "--owner",
            source.owner,
            "--database",
            str(db),
            "--include-private",
        ],
    )
    module.main()
    module.main()  # same identities are idempotent, never overwritten
    assert cloud["reports"][0].get("migration-report") == AnalysisStore(db).get(
        "migration-report"
    )
    assert cloud["jobs"][0].get("migration-job")["status"] == "completed"
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in files] == before


def test_cloud_sources_match_local_identities_and_exact_alignment(cloud, tmp_path):
    from app.storage.source_corpus import SourceCorpusStore
    from app.detector import run_detection
    from app.cloud.retrieval import CloudRetriever

    a, b = cloud["sources"]
    m = meta()
    result = a.ingest(m, TEXT, TEXT.encode())
    local = SourceCorpusStore(tmp_path / "sources.sqlite3").ingest(
        m, TEXT, TEXT.encode()
    )
    assert (
        result["source_id"] == local["source_id"]
        and result["version_id"] == local["version_id"]
    )
    assert a.ingest(m, TEXT, b"different serialization")["duplicate"]
    assert len(cloud["services"].vectors) == 1
    assert not b.snapshot().documents and b.get_version(result["version_id"]) is None
    corpus = a.snapshot()
    ranking = CloudRetriever(corpus).rank([TEXT])
    assert ranking["documents"] == [result["version_id"]]
    report = run_detection("test.txt", "txt", TEXT, corpus)
    assert report["similarity_report"]["textual_similarity_percent"] == 100
    assert report["findings"][0]["source_document_id"] == result["version_id"]
    assert (
        a.snapshot(version_id=corpus.corpus_info["version"]).versions == corpus.versions
    )


def test_bm25_same_scores_as_local(cloud, tmp_path):
    from app.storage.source_corpus import SourceCorpusStore
    from app.cloud.retrieval import CloudRetriever

    a = cloud["sources"][0]
    local = SourceCorpusStore(tmp_path / "sources.sqlite3")
    for index, text in enumerate(
        [
            TEXT,
            "The sensor measured winter rainfall throughout the experiment.",
            "Ocean tides vary with lunar gravity.",
        ]
    ):
        m = meta(external=str(index))
        a.ingest(m, text, text.encode())
        local.ingest(m, text, text.encode())
    query = "sensor sensor experiment winter"
    expected = {}
    corpus = local.snapshot()
    for match in corpus.search(query, limit=100):
        for owner in match.source_occurrences:
            expected[owner.source_document_id] = max(
                expected.get(owner.source_document_id, 0), match.score
            )
    assert CloudRetriever(a.snapshot()).bm25(query) == expected


def test_upsert_failure_does_not_publish_source_version(cloud):
    a = cloud["sources"][0]
    first = a.ingest(meta(), TEXT, TEXT.encode())
    cloud["services"].fail_upsert = True
    with pytest.raises(CloudUnavailable):
        a.ingest(meta("2"), TEXT + " A new reading was captured.", b"new file")
    assert a.snapshot().versions == (first["version_id"],)
    cloud["services"].fail_upsert = False
    second = a.ingest(meta("2"), TEXT + " A new reading was captured.", b"new file")
    assert second["version_id"] != first["version_id"] and a.snapshot().versions == (
        second["version_id"],
    )


def test_deletion_filters_stale_vectors_and_retries_cleanup(cloud):
    a = cloud["sources"][0]
    result = a.ingest(meta(), TEXT, TEXT.encode())
    cloud["services"].fail_delete = True
    assert a.delete(result["source_id"])["vector_cleanup"] == "pending"
    assert not a.snapshot().documents and not a.list_sources()
    assert cloud[
        "services"
    ].vectors  # provider still contains inaccessible stale entries
    cloud["services"].fail_delete = False
    assert a.flush_deletions() and not cloud["services"].vectors


def test_corrupt_vector_provenance_is_rejected(cloud):
    a = cloud["sources"][0]
    a.ingest(meta(), TEXT, TEXT.encode())
    cloud["services"].corrupt_query = True
    with pytest.raises(CloudUnavailable, match="provenance"):
        a.snapshot().create_retriever().rank([TEXT])


def test_postgres_rls_blocks_other_accounts_and_browser_writes(cloud):
    from app.cloud.client import connect

    a, b = cloud["reports"]
    a.save("own", {"filename": "secret.txt"})
    with connect() as c:
        c.execute("SET LOCAL ROLE authenticated")
        c.execute("SELECT set_config('request.jwt.claim.sub',%s,true)", (b.owner,))
        assert c.execute("SELECT * FROM st_reports").fetchall() == []
    with pytest.raises(CloudUnavailable):
        with connect() as c:
            c.execute("SET LOCAL ROLE authenticated")
            c.execute("SELECT set_config('request.jwt.claim.sub',%s,true)", (a.owner,))
            c.execute(
                "INSERT INTO st_reports(owner_id,analysis_id,payload) VALUES(%s,'bypass','{}')",
                (a.owner,),
            )


def test_api_authentication_and_cross_user_access(cloud):
    from fastapi.testclient import TestClient
    from app.main import app

    a, b = cloud["owners"]
    s = cloud["services"]
    a_token = s.token(a)
    b_token = s.token(b)
    with TestClient(app) as client:
        assert client.get("/api/analyses").status_code == 401
        assert (
            client.get(
                "/api/analyses",
                headers={"Authorization": "Bearer " + s.token(a, expired=True)},
            ).status_code
            == 401
        )
        assert client.get("/api/analyses", headers={"X-User-ID": a}).status_code == 401
        ah = {"Authorization": "Bearer " + a_token}
        bh = {"Authorization": "Bearer " + b_token}
        reference = client.post(
            "/api/sources",
            headers=ah,
            data={"title": "CC0 control", "url": "https://example.org/control"},
            files={"file": ("control.txt", TEXT.encode())},
        )
        assert reference.status_code == 201, reference.text
        job = client.post(
            "/api/analysis-jobs",
            headers=ah,
            data={"cloud_review": "false", "web_search": "false"},
            files={"file": ("submission.txt", TEXT.encode())},
        )
        assert job.status_code == 202, job.text
        key = job.json()["job_id"]
        deadline = time.monotonic() + 15
        while True:
            state = client.get("/api/analysis-jobs/" + key, headers=ah).json()
            if state["status"] != "running" or time.monotonic() > deadline:
                break
            time.sleep(0.03)
        assert state["status"] == "completed", state
        identifier = state["analysis_id"]
        assert client.get("/api/analysis-jobs/" + key, headers=bh).status_code == 404
        assert client.get("/api/analysis/" + identifier, headers=bh).status_code == 404
        assert (
            client.get(
                "/api/analysis/" + identifier + "/report.pdf", headers=bh
            ).status_code
            == 404
        )
        assert (
            client.delete("/api/analysis/" + identifier, headers=bh).status_code == 404
        )
        own = client.get("/api/analysis/" + identifier, headers=ah).json()
        assert own["similarity_report"]["textual_similarity_percent"] == 100
        assert client.get(
            "/api/analysis/" + identifier + "/report.pdf", headers=ah
        ).content.startswith(b"%PDF")
        assert (
            client.post(
                "/api/analysis/" + identifier + "/ai-review",
                headers=ah,
                json={"cloud_review": True},
            ).status_code
            == 403
        )
        assert (
            client.delete(
                "/api/sources/" + reference.json()["source_id"], headers=bh
            ).status_code
            == 404
        )


def test_daily_reservations_are_persistent_and_atomic(cloud):
    from app.cloud.records import reserve
    from fastapi import HTTPException
    from app.cloud.client import connect

    owner = cloud["owners"][0]
    reserve(owner, "bounded", 1, 10)
    with pytest.raises(HTTPException) as e:
        reserve(owner, "bounded", 1, 10)
    assert e.value.status_code == 429
    with connect() as c:
        assert (
            c.execute(
                "SELECT used FROM st_usage WHERE subject='__global__' AND operation='bounded'"
            ).fetchone()["used"]
            == 1
        )
