"""Resumable promotion of licensed Wikipedia sources; never re-embeds text."""

from urllib.parse import urlsplit
from app.cloud.client import Blobs, CloudUnavailable, connect, owner_id
from app.cloud.libraries import GLOBAL_LIBRARY, require_admin
from app.cloud.pinecone import Pinecone
from app.cloud.sources import CloudSourceStore
from app.corpus.models import SourceMetadata


def finish_retirement(from_owner, source, version):
    # Only the transferred version is removed remotely, never a whole private
    # namespace/source family or a concurrently updated private reference.
    Pinecone(from_owner).delete_version(source, version)
    with connect() as c:
        c.execute(
            "UPDATE st_occurrences SET indexed=false WHERE owner_id=%s AND version_id=%s",
            (from_owner, version),
        )
        c.execute(
            "UPDATE st_global_promotions SET state='retired',updated_at=now() WHERE from_owner=%s AND source_id=%s AND version_id=%s AND state='retire_pending'",
            (from_owner, source, version),
        )


def promote_wikipedia(actor, from_owner, *, retire_private=False):
    require_admin(actor)
    from_owner = owner_id(from_owner)
    private = CloudSourceStore(from_owner, include_global=False)
    global_store = CloudSourceStore(actor, scope="global")
    with connect() as c:
        cleanup = c.execute(
            "SELECT source_id,version_id FROM st_global_promotions WHERE from_owner=%s AND state='retire_pending' ORDER BY source_id",
            (from_owner,),
        ).fetchall()
    if retire_private:
        for row in cleanup:
            finish_retirement(from_owner, row["source_id"], row["version_id"])
    with connect() as c:
        sources = c.execute(
            "SELECT source_id,active_version FROM st_sources WHERE owner_id=%s AND metadata->>'collection'='wikipedia_en' AND active_version IS NOT NULL ORDER BY source_id",
            (from_owner,),
        ).fetchall()
    for row in sources:
        source, version = row["source_id"], row["active_version"]
        original = private.get_version(version)
        if not original:
            raise CloudUnavailable(
                "Original source evidence is unavailable; no private data was removed"
            )
        metadata = SourceMetadata.model_validate(original["metadata"])
        if (
            metadata.collection != "wikipedia_en"
            or metadata.rights_status != "WIKIMEDIA_DEFAULT_LICENSE"
            or urlsplit(metadata.version_url or metadata.url).hostname
            != "en.wikipedia.org"
            or not (metadata.license_url or "").startswith(
                "https://creativecommons.org/licenses/by-sa/"
            )
        ):
            raise CloudUnavailable(
                "Promotion is restricted to the licensed Wikipedia collection"
            )
        with connect() as c:
            counts = c.execute(
                "SELECT count(*) AS n,count(*) FILTER(WHERE indexed) AS indexed FROM st_occurrences WHERE owner_id=%s AND version_id=%s",
                (from_owner, version),
            ).fetchone()
            if not counts["n"] or counts["n"] != counts["indexed"]:
                yield dict(
                    source=source, status="awaiting_original_index", version=version
                )
                continue
            c.execute(
                "INSERT INTO st_global_promotions(from_owner,source_id,version_id) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                (from_owner, source, version),
            )
        result = global_store.ingest(
            metadata,
            original["text"],
            Blobs(from_owner).get(original["raw_sha256"]),
            source_id=source,
            reuse_from=from_owner,
        )
        if (
            result["version_id"] != version
            or result["reused_verified_vectors"] != counts["n"]
        ):
            raise CloudUnavailable(
                "Promotion identity or vector count changed; private source retained"
            )
        with connect() as c:
            c.execute(
                "UPDATE st_global_promotions SET state='verified',verified_vectors=%s,vector_digest=%s,updated_at=now() WHERE from_owner=%s AND source_id=%s AND version_id=%s",
                (counts["n"], result["vector_digest"], from_owner, source, version),
            )
            if retire_private:
                changed = c.execute(
                    "UPDATE st_sources SET active_version=NULL WHERE owner_id=%s AND source_id=%s AND active_version=%s AND metadata->>'collection'='wikipedia_en' RETURNING source_id",
                    (from_owner, source, version),
                ).fetchone()
                if not changed:
                    raise CloudUnavailable(
                        "Private source changed during promotion; redundant vectors were retained"
                    )
                c.execute(
                    "UPDATE st_global_promotions SET state='retire_pending',updated_at=now() WHERE from_owner=%s AND source_id=%s AND version_id=%s",
                    (from_owner, source, version),
                )
        if retire_private:
            finish_retirement(from_owner, source, version)
        yield dict(
            source=source,
            version=version,
            status="retired_private_copy" if retire_private else "global_verified",
            verified_vectors=counts["n"],
            embedding_calls=0,
        )
    with connect() as c:
        pending = c.execute(
            """SELECT count(*) AS n FROM st_sources s WHERE owner_id=%s AND metadata->>'collection'='wikipedia_en'
            AND NOT EXISTS(SELECT 1 FROM st_global_promotions p WHERE p.from_owner=s.owner_id AND p.source_id=s.source_id
              AND ((s.active_version IS NULL AND p.state='retired')
                OR (s.active_version=p.version_id AND p.state='verified')))""",
            (from_owner,),
        ).fetchone()["n"]
    yield dict(
        complete=pending == 0,
        pending_original_sources=pending,
        unpublished_original_sources=pending,
        global_library=GLOBAL_LIBRARY,
    )
