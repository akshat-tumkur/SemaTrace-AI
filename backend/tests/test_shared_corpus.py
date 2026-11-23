"""Three-account shared/private security, merge and resumable promotion checks."""

import math
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from test_cloud import cloud, meta, TEXT  # noqa: F401
from app.cloud.client import CloudUnavailable, connect
from app.cloud.libraries import GLOBAL_LIBRARY, is_admin
from app.cloud.sources import CloudSourceStore
from app.cloud.records import CloudAnalysisStore
from app.cloud.promotion import promote_wikipedia
from app.detector import run_detection


@pytest.fixture
def shared(cloud):  # noqa: F811 - imported pytest fixture injection
    admin, first = cloud["owners"]
    second = str(uuid4())
    with connect() as c:
        c.execute("INSERT INTO auth.users(id) VALUES(%s)", (second,))
        c.execute("INSERT INTO st_admins(user_id) VALUES(%s)", (admin,))
    return dict(
        cloud=cloud,
        admin=admin,
        users=[admin, first, second],
        global_store=CloudSourceStore(admin, scope="global"),
        stores=[CloudSourceStore(x) for x in [admin, first, second]],
    )


def wiki():
    return meta(external="wikipedia-control").model_copy(
        update=dict(
            collection="wikipedia_en",
            url="https://en.wikipedia.org/wiki/Water_cycle",
            version_url="https://en.wikipedia.org/w/index.php?oldid=1",
            rights_status="WIKIMEDIA_DEFAULT_LICENSE",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            authors=("Wikipedia contributors",),
        )
    )


def test_new_accounts_analyze_global_without_own_references(shared):
    result = shared["global_store"].ingest(meta(), TEXT, TEXT.encode())
    for store in shared["stores"]:
        assert store.list_sources() == []
        corpus = store.snapshot()
        assert corpus.version_libraries == {result["version_id"]: GLOBAL_LIBRARY}
        report = run_detection("new-user.txt", "txt", TEXT, corpus)
        assert report["similarity_report"]["textual_similarity_percent"] == 100
        assert len(report["findings"]) == 1
        assert report["findings"][0]["source_document_id"] == result["version_id"]


def test_private_sources_and_reports_remain_isolated_even_from_admin(shared):
    shared["global_store"].ingest(meta(), TEXT, TEXT.encode())
    a, b, c = shared["stores"]
    private = b.ingest(
        meta(external="private"),
        TEXT + " Confidential calibration record.",
        b"private raw",
    )
    assert a.get_version(private["version_id"]) is None
    assert c.get_version(private["version_id"]) is None
    assert private["version_id"] in b.snapshot().versions
    assert private["version_id"] not in c.snapshot().versions
    report = run_detection("private.txt", "txt", TEXT, b.snapshot())
    CloudAnalysisStore(b.owner).save("private-result", report)
    assert CloudAnalysisStore(a.owner).get("private-result") is None
    assert CloudAnalysisStore(c.owner).get("private-result") is None


def test_backend_rejects_forged_global_management_and_revoked_admin(shared):
    from app.main import app

    service = shared["cloud"]["services"]
    admin, user, other = shared["users"]
    headers = [{"Authorization": "Bearer " + service.token(x)} for x in shared["users"]]
    with TestClient(app) as client:
        for h in headers[1:]:
            forged = {**h, "X-Role": "admin", "X-Is-Admin": "true"}
            denied = client.post(
                "/api/sources",
                headers=forged,
                data={
                    "scope": "global",
                    "title": "forged",
                    "url": "https://example.org/global",
                },
                files={"file": ("test.txt", TEXT.encode())},
            )
            assert denied.status_code == 403, denied.text
            assert (
                client.get("/api/account", headers=forged).json()["is_admin"] is False
            )
        created = client.post(
            "/api/sources",
            headers=headers[0],
            data={
                "scope": "global",
                "title": "Global control",
                "url": "https://example.org/global",
            },
            files={"file": ("test.txt", TEXT.encode())},
        )
        assert created.status_code == 201, created.text
        sid = created.json()["source_id"]
        version = created.json()["version_id"]
        for h in headers:
            assert (
                client.get("/api/corpus/sources?scope=global", headers=h).json()[0][
                    "source_id"
                ]
                == sid
            )
        for h in headers[1:]:
            assert (
                client.delete(
                    "/api/sources/" + sid + "?scope=global", headers=h
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/sources",
                    headers=h,
                    data={
                        "scope": "global",
                        "replace_source_id": sid,
                        "title": "forged update",
                        "url": "https://example.org/global",
                    },
                    files={"file": ("new.txt", b"changed")},
                ).status_code
                == 403
            )
        private = client.post(
            "/api/sources",
            headers=headers[1],
            data={
                "scope": "private",
                "title": "Private control",
                "url": "https://example.org/private",
            },
            files={"file": ("private.txt", TEXT.encode())},
        )
        assert private.status_code == 201, private.text
        psid = private.json()["source_id"]
        assert (
            client.get("/api/corpus/sources?scope=private", headers=headers[2]).json()
            == []
        )
        assert (
            client.delete("/api/sources/" + psid, headers=headers[2]).status_code == 404
        )
        changed = client.post(
            "/api/sources",
            headers=headers[0],
            data={
                "scope": "global",
                "replace_source_id": sid,
                "title": "Updated global control",
                "url": "https://example.org/global",
            },
            files={
                "file": (
                    "new.txt",
                    (
                        TEXT + " A second calibration verified the recorded voltage."
                    ).encode(),
                )
            },
        )
        assert changed.status_code == 201, changed.text
        assert (
            changed.json()["source_id"] == sid
            and changed.json()["version_id"] != version
        )
        assert (
            client.get(
                "/api/corpus/versions/" + version, headers=headers[2]
            ).status_code
            == 200
        )
        with connect() as db:
            db.execute("DELETE FROM st_admins WHERE user_id=%s", (admin,))
        assert (
            client.delete(
                "/api/sources/" + sid + "?scope=global", headers=headers[0]
            ).status_code
            == 403
        )


def test_combined_bm25_matches_local_unique_passage_statistics(shared, tmp_path):
    from app.storage.source_corpus import SourceCorpusStore

    local = SourceCorpusStore(tmp_path / "local.sqlite3")
    global_meta = meta(external="global")
    private_meta = meta(external="private")
    private_text = (
        TEXT
        + "\nThe laboratory experiment measured winter rainfall with a second sensor."
    )
    shared["global_store"].ingest(global_meta, TEXT, TEXT.encode())
    local.ingest(global_meta, TEXT, TEXT.encode())
    store = shared["stores"][1]
    store.ingest(private_meta, private_text, private_text.encode())
    local.ingest(private_meta, private_text, private_text.encode())
    query = "sensor sensor laboratory experiment rainfall"
    expected = {}
    corpus = local.snapshot()
    for match in corpus.search(query, limit=100):
        for o in match.source_occurrences:
            expected[o.source_document_id] = max(
                expected.get(o.source_document_id, 0), match.score
            )
    combined = store.snapshot()
    assert combined.corpus_info["unique_passages"] == len(corpus.documents)
    assert combined.create_retriever().bm25(query) == expected


def test_dense_candidates_merge_before_limit_and_private_never_crosses_users(
    shared, monkeypatch
):
    monkeypatch.setenv("SEMATRACE_PINECONE_TOP_K", "8")
    text = "\n".join(
        f"The sensor calibration sample {i} recorded seventeen volts in the controlled laboratory experiment."
        for i in range(12)
    )
    shared["global_store"].ingest(meta(external="global-many"), text, text.encode())
    private = shared["stores"][1].ingest(
        meta(external="private-best"), TEXT, TEXT.encode()
    )
    service = shared["cloud"]["services"]
    for (namespace, _), row in service.vectors.items():
        similarity = 0.8 if namespace == "global-v1" else 0.99
        row["values"] = [similarity, math.sqrt(1 - similarity**2)] + [0.0] * 382
    ranking = shared["stores"][1].snapshot().create_retriever().rank([TEXT])
    assert len(ranking["related"]) == 8
    assert ranking["related"][0]["source_document_id"] == private["version_id"]
    assert ranking["related"][0]["source_library_scope"] == "private"
    other = shared["stores"][2].snapshot().create_retriever().rank([TEXT])
    assert all(r["source_library_scope"] == "global" for r in other["related"])


def test_promotion_resumes_without_reencoding_and_preserves_history(
    shared, monkeypatch
):
    private = CloudSourceStore(shared["admin"], include_global=False)
    original = private.ingest(wiki(), TEXT, TEXT.encode())
    snapshot = private.snapshot()
    report = run_detection("historical.txt", "txt", TEXT, snapshot)
    saved = CloudAnalysisStore(shared["admin"])
    saved.save("historical", report)
    baseline = saved.get("historical")

    def forbidden():
        raise AssertionError("Promotion must not initialize an encoder")

    monkeypatch.setattr("app.cloud.sources.local_encoder", forbidden)
    service = shared["cloud"]["services"]
    service.fail_upsert = True
    with pytest.raises(CloudUnavailable):
        list(promote_wikipedia(shared["admin"], shared["admin"], retire_private=True))
    assert private.list_sources()[0]["version_id"] == original["version_id"]
    assert shared["stores"][2].snapshot().versions == ()
    service.fail_upsert = False
    result = list(
        promote_wikipedia(shared["admin"], shared["admin"], retire_private=True)
    )
    assert result[-1]["complete"] and result[0]["embedding_calls"] == 0
    assert private.list_sources() == []
    assert private.get_version(original["version_id"])["text"] == TEXT
    assert saved.get("historical") == baseline
    restored = shared["stores"][0].snapshot(version_id=snapshot.corpus_info["version"])
    assert restored.versions == snapshot.versions
    assert restored.version_libraries[original["version_id"]] == GLOBAL_LIBRARY
    assert list(
        promote_wikipedia(shared["admin"], shared["admin"], retire_private=True)
    )[-1]["complete"]


def test_duplicate_migrated_version_is_one_source_and_one_finding(shared):
    private = shared["stores"][0]
    original = private.ingest(wiki(), TEXT, TEXT.encode())
    list(promote_wikipedia(shared["admin"], shared["admin"]))
    corpus = private.snapshot()
    assert corpus.versions == (original["version_id"],)
    assert len(run_detection("same.txt", "txt", TEXT, corpus)["findings"]) == 1


def test_corrupt_transfer_is_not_published_or_retired(shared):
    original = shared["stores"][0].ingest(wiki(), TEXT, TEXT.encode())
    shared["cloud"]["services"].corrupt_global_fetch = True
    with pytest.raises(CloudUnavailable):
        list(promote_wikipedia(shared["admin"], shared["admin"], retire_private=True))
    assert shared["stores"][2].snapshot().versions == ()
    assert original["version_id"] in shared["stores"][0].snapshot().versions


def test_remote_cleanup_failure_is_resumable_and_does_not_remove_evidence(shared):
    original = shared["stores"][0].ingest(wiki(), TEXT, TEXT.encode())
    service = shared["cloud"]["services"]
    service.fail_delete = True
    with pytest.raises(CloudUnavailable):
        list(promote_wikipedia(shared["admin"], shared["admin"], retire_private=True))
    assert original["version_id"] in shared["stores"][2].snapshot().versions
    service.fail_delete = False
    assert list(
        promote_wikipedia(shared["admin"], shared["admin"], retire_private=True)
    )[-1]["complete"]
    assert (
        CloudSourceStore(shared["admin"], include_global=False).get_version(
            original["version_id"]
        )["text"]
        == TEXT
    )


def test_rls_blocks_self_granted_admin_and_other_private_sources(shared):
    global_source = shared["global_store"].ingest(meta(), TEXT, TEXT.encode())
    private = shared["stores"][1].ingest(
        meta(external="secret"), TEXT + " A secret note.", b"secret"
    )
    with connect() as c:
        c.execute("SET LOCAL ROLE authenticated")
        c.execute(
            "SELECT set_config('request.jwt.claim.sub',%s,true)", (shared["users"][2],)
        )
        versions = {
            r["version_id"] for r in c.execute("SELECT version_id FROM st_versions")
        }
        assert (
            global_source["version_id"] in versions
            and private["version_id"] not in versions
        )
    with pytest.raises(CloudUnavailable):
        with connect() as c:
            c.execute("SET LOCAL ROLE authenticated")
            c.execute(
                "INSERT INTO st_admins(user_id) VALUES(%s)", (shared["users"][2],)
            )
    assert not is_admin(shared["users"][2])


def test_readiness_requires_actual_global_vectors(shared, monkeypatch):
    from app.cloud import readiness

    shared["global_store"].ingest(meta(), TEXT, TEXT.encode())
    monkeypatch.setattr(readiness, "_last", 0)
    monkeypatch.setattr(readiness, "verify_directory", lambda directory: None)
    assert readiness.ready()["status"] == "ready"
    shared["cloud"]["services"].vectors.clear()
    monkeypatch.setattr(readiness, "_last", 0)
    with pytest.raises(CloudUnavailable):
        readiness.ready()


def test_promotion_does_not_claim_complete_for_unindexed_original(shared):
    original = shared["stores"][0].ingest(wiki(), TEXT, TEXT.encode())
    with connect() as c:
        c.execute(
            "UPDATE st_occurrences SET indexed=false WHERE owner_id=%s AND version_id=%s",
            (shared["admin"], original["version_id"]),
        )
    progress = list(
        promote_wikipedia(shared["admin"], shared["admin"], retire_private=True)
    )
    assert progress[0]["status"] == "awaiting_original_index"
    assert progress[-1]["complete"] is False
    assert progress[-1]["pending_original_sources"] == 1
    assert original["version_id"] in shared["stores"][0].snapshot().versions


def test_global_import_reuses_complete_private_wikipedia(shared, tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from app.storage.source_corpus import SourceCorpusStore

    local = SourceCorpusStore(tmp_path / "native.sqlite3")
    local.ingest(wiki(), TEXT, TEXT.encode())
    shared["stores"][0].ingest(wiki(), TEXT, TEXT.encode())

    def forbidden():
        raise AssertionError("Already verified vectors must be reused")

    monkeypatch.setattr("app.cloud.sources.local_encoder", forbidden)
    spec = importlib.util.spec_from_file_location(
        "shared_import", Path(__file__).resolve().parents[2] / "scripts/cloud_admin.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = tmp_path / "private.env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cloud_admin",
            "--env-file",
            str(env),
            "import-local",
            "--owner",
            shared["admin"],
            "--scope",
            "global",
            "--database",
            str(local.path),
        ],
    )
    module.main()
    assert len(shared["stores"][2].snapshot().versions) == 1


def test_collection_comparison_reuses_local_vectors_without_publishing(
    shared, tmp_path, monkeypatch
):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from test_cloud import Encoder
    from app.storage.source_corpus import SourceCorpusStore

    local = SourceCorpusStore(tmp_path / "native.sqlite3")
    published = local.ingest(wiki(), TEXT, TEXT.encode())
    local.ingest(
        meta(external="only-local"),
        "Unrelated local private instrument records.",
        b"private",
    )
    shared["global_store"].ingest(wiki(), TEXT, TEXT.encode())
    full = local.snapshot()
    before = local.path.read_bytes()
    generation = SimpleNamespace(
        vectors=np.asarray(Encoder().encode([d.text for d in full.documents]))
    )
    monkeypatch.setattr(
        "app.retrieval.generations.GenerationStore.active",
        lambda self, corpus, spec: (generation, None),
    )
    monkeypatch.setattr("app.retrieval.hybrid.local_encoder", lambda: Encoder())
    spec = importlib.util.spec_from_file_location(
        "collection_comparison",
        Path(__file__).resolve().parents[2] / "scripts/cloud_admin.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    env = tmp_path / "private.env"
    env.write_text("", encoding="utf-8")
    output = tmp_path / ".local" / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cloud_admin",
            "--env-file",
            str(env),
            "compare-retrieval",
            "--owner",
            shared["users"][2],
            "--database",
            str(local.path),
            "--output",
            str(output),
        ],
    )
    module.main()
    result = json.loads(output.read_text())
    assert result["passed"] and len(result["rows"]) == 1
    assert result["rows"][0]["version_id"] == published["version_id"]
    assert local.path.read_bytes() == before
