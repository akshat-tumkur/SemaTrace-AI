"""Immutable sources in a shared global library and isolated private libraries."""

from collections import Counter, OrderedDict
from collections.abc import Mapping
import os
import sys
import hashlib
import numpy as np

from app.cloud.client import Blobs, CloudUnavailable, connect, owner_id
from app.cloud.pinecone import Pinecone
from app.cloud.records import jsonb
from app.corpus.models import SourceMetadata, SourceOccurrence
from app.retrieval.bm25 import _tokenize
from app.retrieval.corpus import _source_documents
from app.retrieval.local_model import local_encoder
from app.storage.source_corpus import SourceCorpusStore, canonical, digest
from app.cloud.libraries import (
    GLOBAL_LIBRARY,
    ensure_private_library,
    require_admin,
    library_scope,
)


def checked_copy_vector(record, expected):
    if not record or record.get("metadata") != expected:
        raise CloudUnavailable(
            "Reusable vector provenance does not match the immutable source"
        )
    values = record.get("values", [])
    if len(values) != 384 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) for x in values
    ):
        raise CloudUnavailable("Invalid reusable MiniLM vector")
    array = np.asarray(values, dtype="<f4")
    if not np.isfinite(array).all() or abs(float(np.linalg.norm(array)) - 1) > 0.001:
        raise CloudUnavailable("Invalid reusable MiniLM vector normalization")
    return hashlib.sha256(array.tobytes()).hexdigest()


def occurrence(row):
    m = SourceMetadata.model_validate(row["metadata"])
    return SourceOccurrence(
        row["occurrence_id"],
        row["source_id"],
        row["version_id"],
        row["text_sha256"],
        row["start_char"],
        row["end_char"],
        m.title,
        row["url"],
        m.collection,
        m.revision,
        m.authors,
        m.license_url,
        m.license_name,
        m.rights_status,
        m.attribution_url,
        m.attribution_text,
    )


class SourceTexts(Mapping):
    def __init__(self, store, versions):
        self.store, self.versions, self.cache = store, versions, OrderedDict()

    def __getitem__(self, key):
        if key not in self.versions:
            raise KeyError(key)
        if key not in self.cache:
            row = self.store.get_version(key)
            if row is None:
                raise CloudUnavailable("Source version no longer available")
            self.cache[key] = row["text"]
        self.cache.move_to_end(key)
        while (
            len(self.cache) > 64
            or sum(sys.getsizeof(v) for v in self.cache.values()) > 8 * 1024 * 1024
        ):
            if len(self.cache) == 1:
                break
            self.cache.popitem(last=False)
        return self.cache[key]

    def __iter__(self):
        return iter(self.versions)

    def __len__(self):
        return len(self.versions)


class CloudCorpus:
    def __init__(self, store, versions, count, version_libraries=None):
        self.store, self.versions = store, tuple(sorted(versions))
        self.version_libraries = version_libraries or {
            v: store._owner() for v in versions
        }
        self.documents = range(count)  # presence/count only; passages stay in Postgres
        self.source_documents = SourceTexts(store, self.versions)
        self.corpus_info = dict(
            version="corpus-v1-" + digest(canonical(list(self.versions))),
            schema_version=1,
            source_version_ids=list(self.versions),
            source_origins=len(versions),
            unique_passages=count,
            lexical_status="POSTGRES_BM25",
            dense_status="PINECONE_MINILM",
            offset_unit="unicode_code_points",
            end_exclusive=True,
            library_scopes=sorted(
                {library_scope(k) for k in self.version_libraries.values()}
            ),
        )

    def create_retriever(self):
        from app.cloud.retrieval import CloudRetriever

        return CloudRetriever(self)


class CloudSourceStore:
    cloud = True

    def __init__(self, owner, scope="private", include_global=True):
        # A public readiness dependency is allowed to be constructed without an
        # owner. Every data method still requires a validated account UUID.
        self.owner = owner_id(owner) if owner else None
        if scope not in {"private", "global"}:
            raise ValueError("Unknown library scope")
        self.scope, self.include_global = scope, include_global

    def _owner(self):
        actor = owner_id(self.owner)
        return GLOBAL_LIBRARY if self.scope == "global" else actor

    def read_libraries(self):
        key = self._owner()
        return (
            [GLOBAL_LIBRARY, key]
            if self.scope == "private" and self.include_global
            else [key]
        )

    def require_write(self):
        if self.scope == "global":
            require_admin(self.owner)
        else:
            ensure_private_library(self.owner)

    def has_source(self, source_id):
        with connect() as c:
            return bool(
                c.execute(
                    "SELECT 1 FROM st_sources WHERE owner_id=%s AND source_id=%s",
                    (self._owner(), source_id),
                ).fetchone()
            )

    def get_version(self, version_id):
        with connect() as c:
            row = c.execute(
                "SELECT * FROM st_versions WHERE owner_id=ANY(%s::uuid[]) AND version_id=%s AND (owner_id<>%s OR published_at IS NOT NULL) ORDER BY CASE WHEN owner_id=%s THEN 0 ELSE 1 END LIMIT 1",
                (self.read_libraries(), version_id, GLOBAL_LIBRARY, GLOBAL_LIBRARY),
            ).fetchone()
        if not row:
            return None
        text = Blobs(str(row["owner_id"])).get(row["text_sha256"]).decode("utf-8")
        return dict(
            row,
            text=text,
            ingested_at=row["ingested_at"].isoformat(),
            library_scope=library_scope(row["owner_id"]),
        )

    def find_version(self, source_id, revision, extractor_version):
        with connect() as c:
            row = c.execute(
                "SELECT active_version,metadata FROM st_sources WHERE owner_id=%s AND source_id=%s",
                (self._owner(), source_id),
            ).fetchone()
        if (
            row
            and row["active_version"]
            and row["metadata"]["revision"] == revision
            and row["metadata"]["extractor_version"] == extractor_version
        ):
            return self.get_version(row["active_version"])
        return None

    def snapshot(self, *, collections=(), source_ids=(), version_id=None):
        owner = owner_id(self.owner)
        with connect() as c:
            if version_id:
                saved = c.execute(
                    "SELECT source_versions FROM st_snapshots WHERE owner_id=%s AND version_id=%s",
                    (owner, version_id),
                ).fetchone()
                if not saved:
                    raise CloudUnavailable("Unknown source collection snapshot")
                versions = saved["source_versions"]
            else:
                rows = c.execute(
                    """SELECT active_version FROM st_sources WHERE owner_id=ANY(%s::uuid[]) AND active_version IS NOT NULL
                  AND (%s OR source_id=ANY(%s)) AND (%s OR metadata->>'collection'=ANY(%s)) ORDER BY active_version""",
                    (
                        self.read_libraries(),
                        not source_ids,
                        list(source_ids),
                        not collections,
                        list(collections),
                    ),
                ).fetchall()
                versions = sorted({r["active_version"] for r in rows})
            # Prefer an identical global version when a historical private copy
            # exists. Its canonical ID, text fingerprint and offsets are unchanged.
            available = c.execute(
                """SELECT DISTINCT ON(v.version_id) v.version_id,v.owner_id FROM st_versions v
                  WHERE v.owner_id=ANY(%s::uuid[]) AND v.version_id=ANY(%s)
                    AND (v.owner_id<>%s OR v.published_at IS NOT NULL)
                    AND (%s OR EXISTS(SELECT 1 FROM st_sources a WHERE a.owner_id=v.owner_id AND a.active_version=v.version_id))
                  ORDER BY v.version_id,
                    CASE WHEN EXISTS(SELECT 1 FROM st_occurrences o WHERE o.owner_id=v.owner_id AND o.version_id=v.version_id AND o.indexed) THEN 0 ELSE 1 END,
                    CASE WHEN v.owner_id=%s THEN 0 ELSE 1 END""",
                (
                    self.read_libraries(),
                    versions,
                    GLOBAL_LIBRARY,
                    bool(version_id),
                    GLOBAL_LIBRARY,
                ),
            ).fetchall()
            mapping = {r["version_id"]: str(r["owner_id"]) for r in available}
            if len(mapping) != len(versions):
                raise CloudUnavailable(
                    "Source snapshot ownership or evidence is unavailable"
                )
            count = c.execute(
                "SELECT count(DISTINCT o.passage_id) AS n FROM st_occurrences o JOIN jsonb_each_text(%s) s ON o.version_id=s.key AND o.owner_id=s.value::uuid",
                (jsonb(mapping),),
            ).fetchone()["n"]
            corpus = CloudCorpus(self, versions, count, mapping)
            c.execute(
                "INSERT INTO st_snapshots(owner_id,version_id,source_versions,source_libraries) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (owner, corpus.corpus_info["version"], jsonb(versions), jsonb(mapping)),
            )
        return corpus

    def stats(self):
        return self.snapshot().corpus_info

    def list_sources(self, offset=0, limit=20, collection=None):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Invalid pagination")
        with connect() as c:
            rows = c.execute(
                """SELECT s.*,v.ingested_at,v.text_sha256 FROM st_sources s LEFT JOIN st_versions v
              ON v.owner_id=s.owner_id AND v.version_id=s.active_version WHERE s.owner_id=%s AND s.active_version IS NOT NULL AND (%s::text IS NULL OR s.metadata->>'collection'=%s)
              ORDER BY s.source_id LIMIT %s OFFSET %s""",
                (self._owner(), collection, collection, limit, offset),
            ).fetchall()
        return [
            dict(
                r["metadata"],
                source_id=r["source_id"],
                version_id=r["active_version"],
                ingested_at=r["ingested_at"].isoformat() if r["ingested_at"] else None,
                text_sha256=r["text_sha256"],
                content_status="INDEXED_PROSE" if r["active_version"] else "NOT_ACTIVE",
                library_scope=self.scope,
            )
            for r in rows
        ]

    def ingest(self, meta, text, raw, *, source_id=None, reuse_from=None):
        self.require_write()
        if reuse_from and self.scope != "global":
            raise ValueError(
                "Cross-library vector reuse is restricted to administrator promotion"
            )
        owner = self._owner()
        identifier = source_id or SourceCorpusStore.source_key(meta)
        if not self.flush_deletions():
            raise CloudUnavailable("Pending source removal must finish before indexing")
        if not text.strip() or len(text) > 200000:
            raise ValueError("Empty or oversized source text")
        metadata = meta.model_dump(mode="json")
        text_hash = digest(text)
        version = "v1-" + digest(
            canonical([identifier, canonical(metadata), text_hash])
        )
        parts, _ = _source_documents(
            text, identifier, meta.title, meta.version_url or meta.url
        )
        if not parts:
            raise ValueError("No indexable source passages")
        with connect() as c:
            if (
                self.scope == "private"
                and c.execute(
                    "SELECT 1 FROM st_sources WHERE owner_id=%s AND source_id=%s AND active_version=%s",
                    (GLOBAL_LIBRARY, identifier, version),
                ).fetchone()
            ):
                return dict(
                    source_id=identifier,
                    version_id=version,
                    duplicate=True,
                    persisted=True,
                    units_indexed=0,
                    occurrences_added=0,
                    library_scope="global",
                )
            existing = c.execute(
                "SELECT active_version FROM st_sources WHERE owner_id=%s AND source_id=%s",
                (owner, identifier),
            ).fetchone()
            if existing and existing["active_version"] == version and not reuse_from:
                return dict(
                    source_id=identifier,
                    version_id=version,
                    duplicate=True,
                    persisted=True,
                    units_indexed=0,
                    occurrences_added=0,
                )
        blobs = Blobs(owner)
        raw_hash = blobs.put(raw)
        assert blobs.put(text.encode()) == text_hash
        pine = Pinecone(owner)
        host_hash = digest(pine.host)
        with connect() as c:
            c.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended('sematrace-source-publication',0))"
            )
            state = c.execute(
                "SELECT * FROM st_index_state WHERE owner_id=%s", (owner,)
            ).fetchone()
            if state and (
                state["space"] != pine.space or state["host_hash"] != host_hash
            ):
                raise CloudUnavailable(
                    "Vector space or index host changed; explicit migration required"
                )
            c.execute(
                "INSERT INTO st_index_state VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                (owner, pine.space, host_hash),
            )
            total = c.execute(
                "SELECT count(DISTINCT passage_id) AS n FROM st_passages"
            ).fetchone()["n"]
            unique = list({digest(p.text) for p in parts})
            existing_count = c.execute(
                "SELECT count(DISTINCT passage_id) AS n FROM st_passages WHERE passage_id=ANY(%s)",
                (unique,),
            ).fetchone()["n"]
            if total + len(unique) - existing_count > int(
                os.getenv("SEMATRACE_CLOUD_MAX_PASSAGES", "20000")
            ):
                raise ValueError("Shared free-tier passage budget reached")
            c.execute(
                "INSERT INTO st_sources VALUES(%s,%s,%s,NULL) ON CONFLICT DO NOTHING",
                (owner, identifier, jsonb(metadata)),
            )
            c.execute(
                "INSERT INTO st_versions(owner_id,version_id,source_id,metadata,raw_sha256,text_sha256) VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (owner, version, identifier, jsonb(metadata), raw_hash, text_hash),
            )
            passage_rows = {}
            occurrence_rows = []
            for part in parts:
                if text[part.source_start_char : part.source_end_char] != part.text:
                    raise ValueError("Invalid source offsets")
                pid = digest(part.text)
                tokens = Counter(_tokenize(part.text))
                oid = digest(
                    canonical([version, part.source_start_char, part.source_end_char])
                )
                passage_rows[pid] = dict(
                    passage_id=pid,
                    text=part.text,
                    token_count=sum(tokens.values()),
                    terms=dict(tokens),
                )
                occurrence_rows.append(
                    dict(
                        occurrence_id=oid,
                        passage_id=pid,
                        source_id=part.source_id,
                        url=meta.version_url or meta.url,
                        start_char=part.source_start_char,
                        end_char=part.source_end_char,
                    )
                )
            payload = jsonb(list(passage_rows.values()))
            # A few bulk statements avoid thousands of pooler round trips and
            # psycopg executemany's implicit pipeline (unsupported by Supavisor).
            c.execute(
                """INSERT INTO st_passages SELECT %s,r.passage_id,r.text,r.token_count
              FROM jsonb_to_recordset(%s) AS r(passage_id text,text text,token_count integer)
              ON CONFLICT DO NOTHING""",
                (owner, payload),
            )
            c.execute(
                """INSERT INTO st_terms SELECT %s,r.passage_id,t.key,t.value::integer
              FROM jsonb_to_recordset(%s) AS r(passage_id text,terms jsonb)
              CROSS JOIN LATERAL jsonb_each_text(r.terms) t ON CONFLICT DO NOTHING""",
                (owner, payload),
            )
            c.execute(
                """INSERT INTO st_occurrences(owner_id,occurrence_id,version_id,passage_id,source_id,url,start_char,end_char)
              SELECT %s,r.occurrence_id,%s,r.passage_id,r.source_id,r.url,r.start_char,r.end_char
              FROM jsonb_to_recordset(%s) AS r(occurrence_id text,passage_id text,source_id text,url text,start_char integer,end_char integer)
              ON CONFLICT DO NOTHING""",
                (owner, version, jsonb(occurrence_rows)),
            )
            if (
                c.execute(
                    "SELECT pg_database_size(current_database()) AS n"
                ).fetchone()["n"]
                > 450 * 1024 * 1024
            ):
                raise ValueError(
                    "Free-tier database safety budget reached; publication was rolled back"
                )
        with connect() as c:
            pending = c.execute(
                "SELECT o.*,p.text FROM st_occurrences o JOIN st_passages p USING(owner_id,passage_id) WHERE o.owner_id=%s AND o.version_id=%s AND (%s OR NOT o.indexed) ORDER BY start_char,occurrence_id",
                (owner, version, bool(reuse_from)),
            ).fetchall()
        copy_digest = hashlib.sha256()
        # Provider batches are independent of ONNX microbatches (still 8).
        # At most 64 compact vectors are retained between network round trips.
        batch_size = 64
        for start in range(0, len(pending), batch_size):
            rows = pending[start : start + batch_size]
            if reuse_from:
                ids = [r["occurrence_id"] for r in rows]
                original = Pinecone(reuse_from).fetch(ids)
                current = pine.fetch(ids)
                records = []
                expected_hashes = {}
                expected_metadata = {}
                for r in rows:
                    oid = r["occurrence_id"]
                    expected = dict(
                        space=pine.space,
                        version_id=version,
                        passage_id=r["passage_id"],
                        source_id=identifier,
                    )
                    fingerprint = checked_copy_vector(original.get(oid), expected)
                    expected_hashes[oid] = fingerprint
                    expected_metadata[oid] = expected
                    copy_digest.update((oid + ":" + fingerprint + "\n").encode())
                    try:
                        same = (
                            checked_copy_vector(current.get(oid), expected)
                            == fingerprint
                        )
                    except CloudUnavailable:
                        same = False
                    if not same:
                        records.append(
                            dict(
                                id=oid,
                                values=original[oid]["values"],
                                metadata=expected,
                            )
                        )
                if records:
                    pine.upsert(records)
                verified = pine.fetch(ids) if records else current
                if any(
                    checked_copy_vector(verified.get(oid), expected_metadata[oid])
                    != expected_hashes[oid]
                    for oid in ids
                ):
                    raise CloudUnavailable(
                        "Global vector copy verification failed; private source was retained"
                    )
                with connect() as c:
                    c.execute(
                        "UPDATE st_occurrences SET indexed=true WHERE owner_id=%s AND occurrence_id=ANY(%s)",
                        (owner, ids),
                    )
                continue
            pids = [r["passage_id"] for r in rows]
            with connect() as c:
                reusable = c.execute(
                    "SELECT DISTINCT ON(passage_id) passage_id,occurrence_id,version_id FROM st_occurrences WHERE owner_id=%s AND passage_id=ANY(%s) AND indexed ORDER BY passage_id,occurrence_id",
                    (owner, pids),
                ).fetchall()
            fetched = pine.fetch([r["occurrence_id"] for r in reusable])
            by_pid = {}
            for r in reusable:
                stored = fetched.get(r["occurrence_id"])
                if stored is None:
                    continue
                sm = stored.get("metadata", {})
                if (
                    sm.get("space") != pine.space
                    or sm.get("passage_id") != r["passage_id"]
                    or sm.get("version_id") != r["version_id"]
                ):
                    raise CloudUnavailable("Cached vector provenance mismatch")
                by_pid[r["passage_id"]] = stored["values"]
            missing = [r for r in rows if r["passage_id"] not in by_pid]
            if missing:
                by_pid.update(
                    {
                        r["passage_id"]: v
                        for r, v in zip(
                            missing,
                            local_encoder().encode([r["text"] for r in missing]),
                        )
                    }
                )
            records = [
                dict(
                    id=r["occurrence_id"],
                    values=by_pid[r["passage_id"]],
                    metadata=dict(
                        space=pine.space,
                        version_id=version,
                        passage_id=r["passage_id"],
                        source_id=identifier,
                    ),
                )
                for r in rows
            ]
            pine.upsert(records)
            with connect() as c:
                c.execute(
                    "UPDATE st_occurrences SET indexed=true WHERE owner_id=%s AND occurrence_id=ANY(%s)",
                    (owner, [r["occurrence_id"] for r in rows]),
                )
        # The active pointer is the atomic publication point. Pending vectors are
        # excluded by every query's frozen version filter if anything above fails.
        with connect() as c:
            c.execute(
                "UPDATE st_versions SET published_at=coalesce(published_at,now()) WHERE owner_id=%s AND version_id=%s",
                (owner, version),
            )
            c.execute(
                "UPDATE st_sources SET active_version=%s,metadata=%s WHERE owner_id=%s AND source_id=%s",
                (version, jsonb(metadata), owner, identifier),
            )
        return dict(
            source_id=identifier,
            version_id=version,
            duplicate=False,
            persisted=True,
            units_indexed=len(parts),
            occurrences_added=len(parts),
            generation_id="pinecone-" + pine.space,
            reused_verified_vectors=len(pending) if reuse_from else None,
            vector_digest=copy_digest.hexdigest() if reuse_from else None,
        )

    def delete(self, source_id):
        self.require_write()
        with connect() as c:
            row = c.execute(
                "UPDATE st_sources SET active_version=NULL WHERE owner_id=%s AND source_id=%s RETURNING source_id",
                (self._owner(), source_id),
            ).fetchone()
            if not row:
                return None
            c.execute(
                "INSERT INTO st_vector_deletions(owner_id,source_id) VALUES(%s,%s) ON CONFLICT DO NOTHING",
                (self._owner(), source_id),
            )
        # Authoritative removal occurs first. Even an unavailable vector service
        # cannot make deleted sources eligible for subsequent analyses.
        pending = not self.flush_deletions()
        return dict(
            removed_from_library=True,
            vector_cleanup="pending" if pending else "complete",
            saved_report_evidence_retained=True,
        )

    def flush_deletions(self):
        self.require_write()
        with connect() as c:
            rows = c.execute(
                "SELECT source_id FROM st_vector_deletions WHERE owner_id=%s LIMIT 16",
                (self._owner(),),
            ).fetchall()
        for row in rows:
            try:
                Pinecone(self._owner()).delete_source(row["source_id"])
            except CloudUnavailable:
                return False
            with connect() as c:
                c.execute(
                    "UPDATE st_occurrences SET indexed=false WHERE owner_id=%s AND version_id IN(SELECT version_id FROM st_versions WHERE owner_id=%s AND source_id=%s)",
                    (self._owner(), self._owner(), row["source_id"]),
                )
                c.execute(
                    "DELETE FROM st_vector_deletions WHERE owner_id=%s AND source_id=%s",
                    (self._owner(), row["source_id"]),
                )
        return True

    def get_run(self, run_id):
        return None
