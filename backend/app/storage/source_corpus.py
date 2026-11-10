"""SQLite source repository with immutable blobs and transactional publication."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from app.corpus.models import CorpusUnavailable, SourceMetadata, SourceOccurrence
from app.retrieval.bm25 import CorpusDocument, LocalCorpus
from app.retrieval.corpus import _source_documents

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data/source-corpus/corpus.sqlite3"


def digest(value: bytes | str) -> str:
    return hashlib.sha256(
        value.encode("utf-8") if isinstance(value, str) else value
    ).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceCorpusStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or os.getenv("SEMATRACE_CORPUS_DB_PATH") or DEFAULT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = self.path.parent / "blobs"
        self.blobs.mkdir(exist_ok=True)
        with self.connect() as c:
            version = c.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise CorpusUnavailable("unsupported_corpus_schema")
            # Opening a repository on every HTTP request must remain read-only
            # once initialized. Rewriting user_version acquires a SQLite write
            # lock and can block behind a concurrent full-corpus snapshot.
            if version == 1 and "metrics" in {
                r[1] for r in c.execute("PRAGMA table_info(ingestion_runs)")
            }:
                return
            c.executescript("""
              CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY, collection TEXT NOT NULL,
                external_id TEXT NOT NULL, metadata TEXT NOT NULL, active_version TEXT);
              CREATE TABLE IF NOT EXISTS versions(version_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
                metadata TEXT NOT NULL, raw_sha256 TEXT NOT NULL, text_sha256 TEXT NOT NULL, ingested_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS passages(passage_id TEXT PRIMARY KEY,text TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS occurrences(occurrence_id TEXT PRIMARY KEY, version_id TEXT NOT NULL REFERENCES versions(version_id),
                passage_id TEXT NOT NULL REFERENCES passages(passage_id),source_id TEXT NOT NULL,url TEXT NOT NULL,
                start_char INTEGER NOT NULL,end_char INTEGER NOT NULL, UNIQUE(version_id,start_char,end_char));
              CREATE INDEX IF NOT EXISTS occurrences_version ON occurrences(version_id);
              CREATE TABLE IF NOT EXISTS corpus_snapshots(version_id TEXT PRIMARY KEY, source_version_ids TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS ingestion_runs(run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
                status TEXT NOT NULL, configuration TEXT NOT NULL, metrics TEXT NOT NULL DEFAULT '{}');
              CREATE TABLE IF NOT EXISTS ingestion_items(run_id TEXT NOT NULL REFERENCES ingestion_runs(run_id),item_key TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(run_id,item_key));
              PRAGMA user_version=1;
            """)
            if "metrics" not in {
                r[1] for r in c.execute("PRAGMA table_info(ingestion_runs)")
            }:
                c.execute(
                    "ALTER TABLE ingestion_runs ADD COLUMN metrics TEXT NOT NULL DEFAULT '{}'"
                )

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            with c:
                yield c
        finally:
            c.close()

    def put_blob(self, raw: bytes) -> str:
        key = digest(raw)
        path = self.blobs / key
        if path.exists():
            if digest(path.read_bytes()) != key:
                raise CorpusUnavailable("corrupt_existing_blob")
            return key
        temporary = self.blobs / (key + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return key

    def read_blob(self, key: str) -> bytes:
        if len(key) != 64 or any(ch not in "0123456789abcdef" for ch in key):
            raise CorpusUnavailable("invalid_blob_key")
        try:
            raw = (self.blobs / key).read_bytes()
        except OSError as e:
            raise CorpusUnavailable("missing_blob") from e
        if digest(raw) != key:
            raise CorpusUnavailable("blob_hash_mismatch")
        return raw

    @staticmethod
    def source_key(meta: SourceMetadata) -> str:
        return meta.collection + "-" + digest(meta.external_id)

    def record_metadata(
        self, meta: SourceMetadata, source_id: str | None = None
    ) -> str:
        identifier = source_id or self.source_key(meta)
        with self.connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO sources VALUES(?,?,?,?,NULL)",
                (
                    identifier,
                    meta.collection,
                    meta.external_id,
                    canonical(meta.model_dump(mode="json")),
                ),
            )
        return identifier

    def ingest(
        self,
        meta: SourceMetadata,
        text: str,
        raw: bytes,
        *,
        source_id: str | None = None,
    ) -> dict:
        if not text.strip():
            raise ValueError("empty_source_text")
        identifier = source_id or self.source_key(meta)
        metadata = canonical(meta.model_dump(mode="json"))
        raw_hash = self.put_blob(raw)
        text_hash = self.put_blob(text.encode("utf-8"))
        # Transport serialization can change without changing extracted content.
        # Keep the first raw-response provenance; semantic snapshot IDs use text.
        version = "v1-" + digest(canonical([identifier, metadata, text_hash]))
        parts, _ = _source_documents(
            text, identifier, meta.title, meta.version_url or meta.url
        )
        if not parts:
            raise ValueError("no_source_passages")
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute(
                "SELECT * FROM sources WHERE source_id=?", (identifier,)
            ).fetchone()
            if existing and (
                existing["collection"] != meta.collection
                or existing["external_id"] != meta.external_id
            ):
                raise ValueError("source_identity_collision")
            if existing and existing["active_version"] == version:
                return dict(
                    source_id=identifier,
                    version_id=version,
                    units_indexed=0,
                    occurrences_added=0,
                    duplicate=True,
                    persisted=True,
                )
            before = {
                r[0]
                for r in c.execute(
                    "SELECT DISTINCT passage_id FROM occurrences o JOIN sources s ON s.active_version=o.version_id"
                )
            }
            c.execute(
                "INSERT OR IGNORE INTO sources VALUES(?,?,?,?,NULL)",
                (identifier, meta.collection, meta.external_id, metadata),
            )
            c.execute(
                "INSERT OR IGNORE INTO versions VALUES(?,?,?,?,?,?)",
                (version, identifier, metadata, raw_hash, text_hash, now()),
            )
            for part in parts:
                if text[part.source_start_char : part.source_end_char] != part.text:
                    raise ValueError("invalid_source_offset")
                passage = digest(part.text)
                c.execute(
                    "INSERT OR IGNORE INTO passages VALUES(?,?)", (passage, part.text)
                )
                if (
                    c.execute(
                        "SELECT text FROM passages WHERE passage_id=?", (passage,)
                    ).fetchone()[0]
                    != part.text
                ):
                    raise CorpusUnavailable("passage_hash_collision")
                occurrence = digest(
                    canonical([version, part.source_start_char, part.source_end_char])
                )
                url = (
                    part.url
                    if meta.collection == "legacy_local"
                    else meta.version_url or meta.url
                )
                c.execute(
                    "INSERT OR IGNORE INTO occurrences VALUES(?,?,?,?,?,?,?)",
                    (
                        occurrence,
                        version,
                        passage,
                        part.source_id,
                        url,
                        part.source_start_char,
                        part.source_end_char,
                    ),
                )
            c.execute(
                "UPDATE sources SET active_version=?,metadata=? WHERE source_id=?",
                (version, metadata, identifier),
            )
            # Build and validate before committing the new active version. A failed
            # index/offset/blob check rolls back all rows; previous snapshots survive.
            snapshot = self._snapshot(c)
            self._remember_snapshot(c, snapshot)
            after = {d.passage_id for d in snapshot.documents}
        return dict(
            source_id=identifier,
            version_id=version,
            units_indexed=len(after - before),
            occurrences_added=len(parts),
            duplicate=False,
            persisted=True,
            corpus_version=snapshot.corpus_info["version"],
        )

    def import_legacy(self, corpus: LocalCorpus) -> None:
        covered = set()
        for parent, text in corpus.source_documents.items():
            documents = [d for d in corpus.documents if d.source_document_id == parent]
            if not documents:
                continue
            covered.update(id(d) for d in documents)
            first = documents[0]
            identifier = parent.split("@sha256:", 1)[0]
            meta = SourceMetadata(
                collection="legacy_local",
                external_id=identifier,
                title=first.title,
                url=first.url.rsplit("#u", 1)[0],
                revision=digest(text),
                extractor_version="legacy-decoded-v1",
            )
            self.ingest(meta, text, text.encode("utf-8"), source_id=identifier)
        # Compatibility with callers constructing a LocalCorpus directly without
        # parent snapshots; each supplied record is its own original document.
        for d in corpus.documents:
            if id(d) in covered:
                continue
            meta = SourceMetadata(
                collection="legacy_local",
                external_id=d.source_id,
                title=d.title,
                url=d.url,
                revision=digest(d.text),
                extractor_version="legacy-record-v1",
            )
            self.ingest(meta, d.text, d.text.encode("utf-8"), source_id=d.source_id)

    def _snapshot(
        self,
        c,
        collections: tuple[str, ...] = (),
        source_ids: tuple[str, ...] = (),
        version_id: str | None = None,
    ) -> LocalCorpus:
        if version_id:
            manifest = c.execute(
                "SELECT source_version_ids FROM corpus_snapshots WHERE version_id=?",
                (version_id,),
            ).fetchone()
            if manifest is None:
                raise CorpusUnavailable("unknown_corpus_version")
            members = json.loads(manifest[0])
            if version_id != "corpus-v1-" + digest(canonical(sorted(members))):
                raise CorpusUnavailable("corpus_manifest_hash_mismatch")
            rows = c.execute(
                "SELECT v.*,s.collection FROM versions v JOIN sources s USING(source_id) ORDER BY s.source_id"
            ).fetchall()
            rows = [r for r in rows if r["version_id"] in members]
            if len(rows) != len(members):
                raise CorpusUnavailable("missing_corpus_version_member")
        else:
            rows = c.execute(
                "SELECT v.*,s.collection FROM sources s JOIN versions v ON v.version_id=s.active_version ORDER BY s.source_id"
            ).fetchall()
        rows = [
            r
            for r in rows
            if (not collections or r["collection"] in collections)
            and (not source_ids or r["source_id"] in source_ids)
        ]
        snapshots = {}
        groups = {}
        text_hashes = set()
        versions = []
        occurrence_count = 0
        for r in rows:
            self.read_blob(r["raw_sha256"])
            text = self.read_blob(r["text_sha256"]).decode("utf-8")
            snapshots[r["version_id"]] = text
            text_hashes.add(r["text_sha256"])
            versions.append(r["version_id"])
            meta = SourceMetadata.model_validate_json(r["metadata"])
            for o in c.execute(
                "SELECT o.*,p.text FROM occurrences o JOIN passages p USING(passage_id) WHERE version_id=? ORDER BY start_char",
                (r["version_id"],),
            ):
                if (
                    text[o["start_char"] : o["end_char"]] != o["text"]
                    or digest(o["text"]) != o["passage_id"]
                ):
                    raise CorpusUnavailable("indexed_passage_integrity")
                owner = SourceOccurrence(
                    o["occurrence_id"],
                    o["source_id"],
                    r["version_id"],
                    r["text_sha256"],
                    o["start_char"],
                    o["end_char"],
                    meta.title,
                    o["url"],
                    meta.collection,
                    meta.revision,
                    meta.authors,
                    meta.license_url,
                    meta.license_name,
                    meta.rights_status,
                    meta.attribution_url,
                    meta.attribution_text,
                )
                groups.setdefault(o["passage_id"], (o["text"], []))[1].append(owner)
                occurrence_count += 1
        docs = []
        for passage, (text, owners) in sorted(groups.items()):
            owners = tuple(
                sorted(
                    owners,
                    key=lambda o: (
                        o.source_id,
                        o.source_document_id,
                        o.source_start_char,
                    ),
                )
            )
            primary = owners[0]
            docs.append(
                CorpusDocument(
                    primary.source_id,
                    primary.title,
                    primary.url,
                    text,
                    primary.source_document_id,
                    primary.source_document_sha256,
                    primary.source_start_char,
                    primary.source_end_char,
                    passage,
                    owners,
                )
            )
        # Preserve legacy source-label ordering where possible; deterministic ties.
        docs.sort(key=lambda d: (d.source_id, d.source_start_char, d.passage_id))
        corpus = LocalCorpus(docs, snapshots, reuse_statistics=True)
        corpus.retrieval_index_path = self.path.with_suffix(".retrieval.sqlite3")
        corpus.corpus_info = dict(
            version="corpus-v1-" + digest(canonical(sorted(versions))),
            schema_version=1,
            source_version_ids=sorted(versions),
            source_origins=len(rows),
            unique_text_documents=len(text_hashes),
            unique_passages=len(docs),
            passage_occurrences=occurrence_count,
            lexical_status="READY",
            dense_status="NOT_PERSISTED_REQUEST_SCOPED",
            offset_unit="unicode_code_points",
            end_exclusive=True,
        )
        return corpus

    def _remember_snapshot(self, c, corpus):
        c.execute(
            "INSERT OR IGNORE INTO corpus_snapshots VALUES(?,?,?)",
            (
                corpus.corpus_info["version"],
                canonical(corpus.corpus_info["source_version_ids"]),
                now(),
            ),
        )

    def snapshot(
        self,
        *,
        collections: tuple[str, ...] = (),
        source_ids: tuple[str, ...] = (),
        version_id: str | None = None,
    ) -> LocalCorpus:
        try:
            with self.connect() as c:
                c.execute("BEGIN")
                corpus = self._snapshot(c, collections, source_ids, version_id)
                known = c.execute(
                    "SELECT 1 FROM corpus_snapshots WHERE version_id=?",
                    (corpus.corpus_info["version"],),
                ).fetchone()
            if known is None:
                with self.connect() as c:
                    self._remember_snapshot(c, corpus)
            return corpus
        except (OSError, sqlite3.Error, ValueError, UnicodeError) as e:
            raise CorpusUnavailable("corpus_snapshot_unavailable") from e

    def list_sources(
        self, offset=0, limit=20, collection: str | None = None
    ) -> list[dict]:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("invalid_pagination")
        with self.connect() as c:
            rows = c.execute(
                "SELECT s.*,v.ingested_at,v.text_sha256 FROM sources s LEFT JOIN versions v ON v.version_id=s.active_version WHERE (? IS NULL OR s.collection=?) ORDER BY s.source_id LIMIT ? OFFSET ?",
                (collection, collection, limit, offset),
            ).fetchall()
        return [
            {
                **json.loads(r["metadata"]),
                "source_id": r["source_id"],
                "version_id": r["active_version"],
                "ingested_at": r["ingested_at"],
                "text_sha256": r["text_sha256"],
                "content_status": "INDEXED_PROSE"
                if r["active_version"]
                else "METADATA_ONLY",
            }
            for r in rows
        ]

    def get_version(self, version_id: str) -> dict | None:
        with self.connect() as c:
            r = c.execute(
                "SELECT * FROM versions WHERE version_id=?", (version_id,)
            ).fetchone()
        if not r:
            return None
        self.read_blob(r["raw_sha256"])
        return {
            **dict(r),
            "metadata": json.loads(r["metadata"]),
            "text": self.read_blob(r["text_sha256"]).decode("utf-8"),
        }

    def find_version(
        self, source_id: str, revision: str, extractor_version: str
    ) -> dict | None:
        with self.connect() as c:
            r = c.execute(
                "SELECT v.* FROM versions v JOIN sources s ON s.active_version=v.version_id WHERE s.source_id=?",
                (source_id,),
            ).fetchone()
        if r:
            m = json.loads(r["metadata"])
            if (
                m["revision"] == revision
                and m["extractor_version"] == extractor_version
            ):
                return self.get_version(r["version_id"])
        return None

    def stats(self) -> dict:
        result = dict(self.snapshot().corpus_info)
        with self.connect() as c:
            result["metadata_only_sources"] = c.execute(
                "SELECT count(*) FROM sources WHERE active_version IS NULL"
            ).fetchone()[0]
            result["failed_ingestion_items"] = c.execute(
                "SELECT count(*) FROM ingestion_items WHERE status='FAILED'"
            ).fetchone()[0]
        return result

    def start_run(self, configuration: dict) -> str:
        identifier = uuid4().hex
        with self.connect() as c:
            c.execute(
                "INSERT INTO ingestion_runs(run_id,started_at,status,configuration) VALUES(?,?,'RUNNING',?)",
                (identifier, now(), canonical(configuration)),
            )
        return identifier

    def record_item(
        self, run_id: str, item_key: str, status: str, result: dict
    ) -> None:
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO ingestion_items VALUES(?,?,?,?,?)",
                (run_id, item_key, status, canonical(result), now()),
            )

    def finish_run(self, run_id: str, status: str, metrics: dict | None = None) -> None:
        with self.connect() as c:
            c.execute(
                "UPDATE ingestion_runs SET status=?,finished_at=?,metrics=? WHERE run_id=?",
                (status, now(), canonical(metrics or {}), run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self.connect() as c:
            row = c.execute(
                "SELECT * FROM ingestion_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            items = c.execute(
                "SELECT * FROM ingestion_items WHERE run_id=? ORDER BY item_key",
                (run_id,),
            ).fetchall()
        return (
            None
            if row is None
            else {
                **dict(row),
                "configuration": json.loads(row["configuration"]),
                "metrics": json.loads(row["metrics"]),
                "items": [
                    {**dict(i), "result": json.loads(i["result"])} for i in items
                ],
            }
        )
