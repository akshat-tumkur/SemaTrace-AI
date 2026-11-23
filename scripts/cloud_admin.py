"""Explicit cloud setup. Never creates indexes at application startup."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.cloud")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-index")
    create.add_argument(
        "--confirm-starter",
        action="store_true",
        help="Confirm the Pinecone project uses the free Starter plan, not a paid plan or trial.",
    )
    commands.add_parser("migrate")
    commands.add_parser("check")
    commands.add_parser("migration-status")
    grant = commands.add_parser("grant-admin")
    grant.add_argument("--user", required=True)
    revoke = commands.add_parser("revoke-admin")
    revoke.add_argument("--user", required=True)
    promotion = commands.add_parser("promote-wikipedia")
    promotion.add_argument("--from-owner", required=True)
    promotion.add_argument("--admin", required=True)
    promotion.add_argument("--retire-private", action="store_true")
    transfer = commands.add_parser("import-local")
    transfer.add_argument(
        "--owner", required=True, help="UUID of an existing Supabase Auth user"
    )
    transfer.add_argument(
        "--collection", choices=["wikipedia_en", "all"], default="wikipedia_en"
    )
    transfer.add_argument("--include-private", action="store_true")
    transfer.add_argument("--source-id", action="append")
    transfer.add_argument("--scope", choices=["global", "private"])
    transfer.add_argument(
        "--database",
        type=Path,
        default=ROOT / "backend/data/source-corpus/corpus.sqlite3",
    )
    reports = commands.add_parser("import-reports")
    reports.add_argument("--owner", required=True)
    reports.add_argument("--database", type=Path, required=True)
    reports.add_argument("--include-private", action="store_true")
    compare = commands.add_parser("compare-retrieval")
    compare.add_argument("--owner", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--max-sources", type=int, default=16)
    compare.add_argument(
        "--collection", choices=["wikipedia_en", "all"], default="wikipedia_en"
    )
    compare.add_argument(
        "--database",
        type=Path,
        default=ROOT / "backend/data/source-corpus/corpus.sqlite3",
    )
    args = parser.parse_args()
    if not args.env_file.is_file():
        parser.error(
            "Copy configs/cloud.env.example to .env.cloud and configure private server credentials first."
        )
    from dotenv import load_dotenv, set_key

    load_dotenv(args.env_file, override=True)
    from app.cloud.client import CloudUnavailable, connect, request, required

    if args.command == "create-index":
        if not args.confirm_starter:
            parser.error(
                "Select Pinecone Starter in the dashboard, then pass --confirm-starter. No automatic plan upgrades are made."
            )
        import os
        from app.cloud.pinecone import API_VERSION, provider_error

        name = os.getenv("PINECONE_INDEX_NAME") or "sematrace-minilm"
        if not name.replace("-", "").isalnum() or not 1 <= len(name) <= 45:
            parser.error("Invalid index name")
        headers = {
            "Api-Key": required("PINECONE_API_KEY"),
            "X-Pinecone-Api-Version": API_VERSION,
            "Content-Type": "application/json",
        }
        url = "https://api.pinecone.io/indexes"
        operation = "describe index"
        status, raw = request("GET", url + "/" + name, headers=headers)
        if status == 404:
            operation = "create index"
            status, raw = request(
                "POST",
                url,
                headers=headers,
                json={
                    "name": name,
                    "vector_type": "dense",
                    "dimension": 384,
                    "metric": "cosine",
                    "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
                },
            )
        if status not in {200, 201, 202}:
            raise CloudUnavailable(
                f"Pinecone {operation} failed (HTTP {status}, API {API_VERSION}): "
                + provider_error(raw, headers["Api-Key"])
                + ". No plan upgrade was attempted."
            )
        value = json.loads(raw)
        if (
            value.get("dimension") != 384
            or value.get("metric") != "cosine"
            or value.get("embed")
        ):
            raise CloudUnavailable(
                "Existing index is incompatible. Use a new dense index without integrated inference."
            )
        host = value.get("host")
        if not isinstance(host, str) or not host.endswith(".pinecone.io"):
            raise CloudUnavailable("Pinecone did not return an index host")
        set_key(str(args.env_file), "PINECONE_INDEX_NAME", name)
        set_key(str(args.env_file), "PINECONE_INDEX_HOST", host)
        print(
            json.dumps(
                {
                    "name": name,
                    "dimension": 384,
                    "metric": "cosine",
                    "api_version": API_VERSION,
                    "integrated_embedding_model": None,
                    "host_saved_to": str(args.env_file),
                    "state": value.get("status", {}).get("state"),
                }
            )
        )
    elif args.command == "migrate":
        with connect() as c:
            for name in ["schema.sql", "storage.sql", "shared-corpus.sql"]:
                c.execute(
                    (ROOT / "backend/cloud" / name).read_text(encoding="utf-8"),
                    prepare=False,
                )
        print(
            "Supabase schema, shared/private libraries, protected administrator roles and private storage bucket initialized."
        )
    elif args.command in {"grant-admin", "revoke-admin"}:
        from app.cloud.client import owner_id

        actor = owner_id(args.user)
        with connect() as c:
            if not c.execute(
                "SELECT 1 FROM auth.users WHERE id=%s", (actor,)
            ).fetchone():
                parser.error("Use the UUID of an existing Supabase Auth account.")
            if args.command == "grant-admin":
                c.execute(
                    "INSERT INTO st_admins(user_id) VALUES(%s) ON CONFLICT DO NOTHING",
                    (actor,),
                )
            else:
                c.execute("DELETE FROM st_admins WHERE user_id=%s", (actor,))
        print(
            json.dumps(dict(user_id=actor, administrator=args.command == "grant-admin"))
        )
    elif args.command == "migration-status":
        with connect() as c:
            rows = c.execute(
                "SELECT s.owner_id::text,s.metadata->>'collection' AS collection,count(*) AS sources,count(s.active_version) AS active FROM st_sources s GROUP BY s.owner_id,s.metadata->>'collection' ORDER BY s.owner_id,collection"
            ).fetchall()
            counts = c.execute(
                "SELECT owner_id::text,count(*) AS passages,count(*) FILTER(WHERE indexed) AS indexed FROM st_occurrences GROUP BY owner_id ORDER BY owner_id"
            ).fetchall()
            size = c.execute(
                "SELECT pg_database_size(current_database()) AS bytes"
            ).fetchone()["bytes"]
        print(
            json.dumps(
                dict(
                    libraries=rows,
                    indexing=counts,
                    database_mib=round(size / 1024**2, 2),
                )
            )
        )
    elif args.command == "promote-wikipedia":
        from app.cloud.promotion import promote_wikipedia

        for progress in promote_wikipedia(
            args.admin, args.from_owner, retire_private=args.retire_private
        ):
            print(json.dumps(progress), flush=True)
    elif args.command == "check":
        from app.cloud.readiness import ready

        print(json.dumps(ready()))
    elif args.command in {"import-reports", "compare-retrieval"}:
        from app.cloud.client import owner_id
        from app.cloud.records import CloudAnalysisStore, jsonb

        owner = owner_id(args.owner)
        with connect() as c:
            if not c.execute(
                "SELECT 1 FROM auth.users WHERE id=%s", (owner,)
            ).fetchone():
                parser.error("Create the Supabase Auth user first, then use its UUID.")
        if args.command == "import-reports":
            if not args.include_private:
                parser.error(
                    "--include-private is required to authorize copying existing private reports/jobs."
                )
            import sqlite3
            from app.storage.database import AnalysisStore

            path = args.database.resolve(strict=True)

            class ReadOnlyStore(AnalysisStore):
                def _connect(self):
                    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
                    db.row_factory = sqlite3.Row
                    return db

            local = ReadOnlyStore.__new__(ReadOnlyStore)
            local.database_path = path
            remote = CloudAnalysisStore(owner)
            with local._connect() as db:
                ids = [
                    r["id"] for r in db.execute("SELECT id FROM analyses ORDER BY id")
                ]
            for identifier in ids:
                value = local.get(identifier)
                existing = remote.get(identifier)
                if existing is not None and existing != value:
                    raise CloudUnavailable(
                        "Existing cloud report differs; no overwrite performed"
                    )
                if existing is None:
                    remote.save(identifier, value)
            jobs_path = path.with_suffix(".jobs.sqlite3")
            jobs_count = 0
            if jobs_path.exists():
                with sqlite3.connect(jobs_path.as_uri() + "?mode=ro", uri=True) as db:
                    for key, raw in db.execute(
                        "SELECT id,payload FROM product_jobs ORDER BY id"
                    ):
                        value = json.loads(raw)
                        if value.get("status") == "running":
                            value.update(
                                status="failed",
                                stage="Interrupted",
                                error="Migrated interrupted local job; upload again to run a new analysis.",
                            )
                        with connect() as c:
                            existing = c.execute(
                                "SELECT payload FROM st_jobs WHERE owner_id=%s AND job_id=%s",
                                (owner, key),
                            ).fetchone()
                            if existing and existing["payload"] != value:
                                raise CloudUnavailable(
                                    "Existing cloud job differs; no overwrite performed"
                                )
                            c.execute(
                                "INSERT INTO st_jobs VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                                (owner, key, jsonb(value)),
                            )
                        jobs_count += 1
            print(
                json.dumps(
                    dict(
                        reports=len(ids),
                        jobs=jobs_count,
                        original_local_files_unchanged=True,
                        note="Historical local jobs did not retain original upload bytes. Import their reference library separately.",
                    )
                )
            )
        else:
            from app.cloud.sources import CloudSourceStore
            from app.storage.source_corpus import SourceCorpusStore
            from app.retrieval.hybrid import HybridRetriever

            if not 1 <= args.max_sources <= 64:
                parser.error("--max-sources must be 1–64")
            output = args.output.resolve()
            if ROOT / ".local" not in output.parents:
                parser.error(
                    "Comparison output must stay under .local because source identities are private."
                )
            from app.retrieval.generations import GenerationStore, IndexSpec

            class ReadOnlySnapshots(SourceCorpusStore):
                def _remember_snapshot(self, connection, corpus):
                    pass

            store = ReadOnlySnapshots(args.database.resolve(strict=True))
            full = store.snapshot()
            collections = () if args.collection == "all" else (args.collection,)
            local = store.snapshot(collections=collections) if collections else full
            cloud = CloudSourceStore(owner).snapshot(collections=collections)
            if tuple(sorted(local.corpus_info["source_version_ids"])) != cloud.versions:
                parser.error(
                    "Import the same selected collection first; corpus identities must match."
                )
            generation, error = GenerationStore(full.retrieval_index_path).active(
                full, IndexSpec.local()
            )
            if generation is None or error:
                raise CloudUnavailable(
                    "Build the compatible local index before comparison"
                )
            # Reuse the verified full generation without re-embedding or publishing
            # a new local index merely to compare the eligible shared collection.
            positions = {d.passage_id: i for i, d in enumerate(full.documents)}
            vectors = generation.vectors[
                [positions[d.passage_id] for d in local.documents]
            ]
            retriever = HybridRetriever(local, vectors=vectors)
            remote = cloud.create_retriever()
            rows = []
            for version in cloud.versions[: args.max_sources]:
                source = cloud.store.get_version(version)
                from app.retrieval.corpus import _source_documents

                parts, _ = _source_documents(
                    source["text"],
                    source["source_id"],
                    source["metadata"]["title"],
                    source["metadata"].get("version_url") or source["metadata"]["url"],
                )
                query = parts[0].text
                expected = retriever.rank([query])
                actual = remote.rank([query])
                local_top = expected["documents"][:10]
                cloud_top = actual["documents"][:10]
                rows.append(
                    dict(
                        version_id=version,
                        local_top10=local_top,
                        cloud_top10=cloud_top,
                        local_hit=version in local_top,
                        cloud_hit=version in cloud_top,
                        shared_top10=len(set(local_top) & set(cloud_top)),
                    )
                )
            passed = bool(rows) and all(
                not r["local_hit"] or r["cloud_hit"] for r in rows
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    dict(
                        passed=passed,
                        scope="Deterministic indexed-passage source retrieval smoke comparison; not independent semantic accuracy.",
                        local_generation=local.corpus_info["version"],
                        cloud_snapshot=cloud.corpus_info["version"],
                        rows=rows,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    dict(passed=passed, queries=len(rows), private_output=str(output))
                )
            )
            if not passed:
                raise SystemExit(1)
    elif args.command == "import-local":
        if args.collection == "all" and not args.include_private:
            parser.error(
                "Use --include-private only if you authorize private references to be copied to your private cloud account."
            )
        from app.cloud.client import owner_id
        from app.cloud.sources import CloudSourceStore
        from app.storage.source_corpus import SourceCorpusStore
        from app.corpus.models import SourceMetadata

        owner = owner_id(args.owner)
        with connect() as c:
            if not c.execute(
                "SELECT 1 FROM auth.users WHERE id=%s", (owner,)
            ).fetchone():
                parser.error(
                    "Create the account in Supabase Authentication > Users first, then copy its user UUID."
                )
        local = SourceCorpusStore(args.database.resolve(strict=True))
        scope = args.scope or (
            "global" if args.collection == "wikipedia_en" else "private"
        )
        if scope == "global" and args.collection != "wikipedia_en":
            parser.error(
                "Bulk global import is restricted to Wikipedia. Share other authorized references individually through the administrator interface."
            )
        cloud = CloudSourceStore(owner, scope=scope)
        cloud.require_write()
        offset = 0
        count = 0
        while True:
            rows = local.list_sources(
                offset,
                100,
                collection=args.collection if args.collection != "all" else None,
            )
            if not rows:
                break
            for row in rows:
                if not row["version_id"] or (
                    args.source_id and row["source_id"] not in args.source_id
                ):
                    continue
                version = local.get_version(row["version_id"])
                reuse_from = None
                if scope == "global":
                    with connect() as c:
                        published = c.execute(
                            "SELECT 1 FROM st_sources WHERE owner_id=%s AND source_id=%s AND active_version=%s",
                            (cloud._owner(), row["source_id"], row["version_id"]),
                        ).fetchone()
                        original = c.execute(
                            """SELECT count(*) AS n,count(*) FILTER(WHERE o.indexed) AS indexed
                            FROM st_occurrences o JOIN st_sources s ON s.owner_id=o.owner_id AND s.active_version=o.version_id
                            WHERE o.owner_id=%s AND o.version_id=%s""",
                            (owner, row["version_id"]),
                        ).fetchone()
                    if (
                        not published
                        and original["n"]
                        and original["n"] == original["indexed"]
                    ):
                        reuse_from = owner
                result = cloud.ingest(
                    SourceMetadata.model_validate(version["metadata"]),
                    version["text"],
                    local.read_blob(version["raw_sha256"]),
                    source_id=row["source_id"],
                    reuse_from=reuse_from,
                )
                if result["version_id"] != row["version_id"]:
                    raise CloudUnavailable(
                        "Source identity changed during transfer; stopped"
                    )
                count += 1
                print(
                    json.dumps(
                        {
                            "source": row["source_id"],
                            "version": row["version_id"],
                            "duplicate": result["duplicate"],
                            "transferred": count,
                            "reused_verified_vectors": result.get(
                                "reused_verified_vectors"
                            )
                            or 0,
                        }
                    ),
                    flush=True,
                )
            offset += len(rows)
        print(
            json.dumps(
                {
                    "complete": True,
                    "sources": count,
                    "owner": owner,
                    "library_scope": scope,
                    "library_id": cloud._owner(),
                    "private_references_included": args.collection == "all",
                }
            )
        )


if __name__ == "__main__":
    from app.cloud.client import CloudUnavailable
    from fastapi import HTTPException

    try:
        main()
    except CloudUnavailable as error:
        raise SystemExit(str(error)) from None
    except HTTPException as error:
        raise SystemExit(str(error.detail)) from None
