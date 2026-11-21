"""Small, explicit corpus commands. Downloading never happens on application startup."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from unittest.mock import patch

from app.corpus.models import SourceMetadata, CorpusUnavailable
from app.corpus.wikimedia import (
    WikipediaClient,
    AcquisitionError,
    BudgetExceeded,
    EXTRACTOR_VERSION,
    validate_entry,
)
from app.storage.source_corpus import SourceCorpusStore, digest, canonical

SEED = Path(__file__).resolve().parents[2] / "corpus/wikipedia-stem.seed.json"


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        data.get("format_version") != 1
        or data.get("collection") != "wikipedia_en"
        or data.get("extractor_version") != EXTRACTOR_VERSION
    ):
        raise ValueError("unsupported_manifest")
    ids = [e["page_id"] for e in data["entries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate_manifest_page")
    for entry in data["entries"]:
        validate_entry(entry)
    return data


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, sort_keys=True)


def entry_meta(entry):
    return SourceMetadata(
        collection="wikipedia_en",
        external_id=str(entry["page_id"]),
        title=entry["title"],
        url=entry["url"],
        revision=str(entry["revision_id"]),
        extractor_version=EXTRACTOR_VERSION,
        authors=("Wikipedia contributors",),
        modified_at=entry["modified_at"],
        license_url=entry["license_url"],
        license_name=entry["license_name"],
        rights_status="WIKIMEDIA_DEFAULT_LICENSE",
        attribution_url=entry["attribution_url"],
        version_url=entry["version_url"],
    )


def ingest_manifest(store, manifest, client, *, limit=12, resume=None):
    if limit < 1:
        raise ValueError("invalid_document_limit")
    selected = manifest["entries"][:limit]
    configuration = dict(
        manifest_sha256=digest(canonical(manifest)),
        max_documents=limit,
        max_response_bytes=client.max_response,
        max_total_bytes=client.max_total,
        max_requests=client.max_requests,
        extractor_version=EXTRACTOR_VERSION,
    )
    previous = store.get_run(resume) if resume else None
    if resume and (not previous or previous["configuration"] != configuration):
        raise ValueError("resume_configuration_mismatch")
    run = resume or store.start_run(configuration)
    previous_metrics = previous["metrics"] if previous else {}
    client.requests = previous_metrics.get("http_requests", 0)
    client.bytes_read = previous_metrics.get("downloaded_bytes", 0)
    started = time.perf_counter()
    rights_checked = False
    for entry in selected:
        key = str(entry["page_id"])
        try:
            meta = entry_meta(entry)
            source_id = store.source_key(meta)
            cached = store.find_version(
                source_id, str(entry["revision_id"]), EXTRACTOR_VERSION
            )
            if cached:
                if (
                    entry.get("text_sha256")
                    and cached["text_sha256"] != entry["text_sha256"]
                ):
                    raise AcquisitionError("cached_text_checksum_mismatch")
                result = dict(
                    source_id=source_id,
                    version_id=cached["version_id"],
                    units_indexed=0,
                    duplicate=True,
                    persisted=True,
                )
                status = "CACHED"
            else:
                if not rights_checked:
                    client.rights()
                    rights_checked = True
                store.record_metadata(meta)
                actual, text, raw = client.article(entry)
                result = store.ingest(actual, text, raw)
                status = "INDEXED"
            store.record_item(run, key, status, result)
        except (
            AcquisitionError,
            CorpusUnavailable,
            ValueError,
            KeyError,
            OSError,
        ) as error:
            store.record_item(
                run,
                key,
                "FAILED",
                dict(
                    error=str(error)
                    if isinstance(error, AcquisitionError)
                    else type(error).__name__
                ),
            )
            if isinstance(error, BudgetExceeded):
                break
    current = store.get_run(run)
    items = current["items"]
    successful = sum(i["status"] in {"INDEXED", "CACHED"} for i in items)
    metrics = dict(
        http_requests=client.requests,
        downloaded_bytes=client.bytes_read,
        elapsed_seconds=time.perf_counter() - started,
        successful_items=successful,
        selected_items=len(selected),
        unattempted_items=len(selected) - len(items),
        model_calls=0,
        api_cost_usd=0,
    )
    metrics["invocations"] = previous_metrics.get("invocations", []) + [
        dict(
            http_requests=client.requests - previous_metrics.get("http_requests", 0),
            downloaded_bytes=client.bytes_read
            - previous_metrics.get("downloaded_bytes", 0),
            elapsed_seconds=metrics["elapsed_seconds"],
        )
    ]
    store.finish_run(
        run, "COMPLETE" if successful == len(selected) else "PARTIAL", metrics
    )
    return store.get_run(run)


def export_lock(store, manifest):
    result = json.loads(json.dumps(manifest))
    ready = 0
    for entry in result["entries"]:
        meta = entry_meta(entry)
        version = store.find_version(
            store.source_key(meta), str(entry["revision_id"]), EXTRACTOR_VERSION
        )
        if version:
            entry.update(
                text_sha256=version["text_sha256"],
                raw_response_sha256=version["raw_sha256"],
                content_status="INDEXED_PROSE",
            )
            ready += 1
        else:
            entry["content_status"] = "NOT_INGESTED"
    result["indexed_sources"] = ready
    result["corpus"] = store.snapshot(collections=("wikipedia_en",)).corpus_info
    result["snapshot_note"] = (
        "Checksums bind the extracted plaintext. Historical HTML can change through templates; a plaintext mismatch fails closed. Raw-response hash records the first preserved response."
    )
    return result


@contextmanager
def no_network():
    attempted = []

    def deny(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("Offline corpus demonstration forbids all network")

    with patch("socket.create_connection", deny), patch("socket.socket.connect", deny):
        yield attempted


def demo_submission(corpus):
    selected = []
    seen = set()
    for d in corpus.documents:
        if d.source_document_id not in seen and 12 <= len(d.text.split()) <= 80:
            selected.append(d)
            seen.add(d.source_document_id)
            if len(selected) == 3:
                break
    if len(selected) < 3:
        raise ValueError("demo_requires_three_ingested_source_documents")
    submission = "\n".join(d.text for d in selected)
    return selected, submission


def demonstrate(store):
    from app.agents.workflow import run_investigation
    from app.agents.llm_verifier import LLMEvidenceVerifier
    from app.config import VerificationSettings
    from app.retrieval.embeddings import EmbeddingIndex

    corpus = store.snapshot(collections=("wikipedia_en",))
    selected, submission = demo_submission(corpus)
    settings = VerificationSettings(enabled=False)
    with no_network() as attempts:
        index = EmbeddingIndex(corpus, api_key="")
        started = time.perf_counter()
        result = run_investigation(
            "real-corpus-exact-excerpts.txt",
            "txt",
            submission,
            corpus,
            llm_verifier=LLMEvidenceVerifier(api_key="", settings=settings),
            verification_settings=settings,
            embedding_index=index,
        )
        elapsed = time.perf_counter() - started
    assert not attempts and result["verification"]["metrics"]["llm_calls"] == 0
    for i, d in enumerate(selected, 1):
        assert any(
            m["unit_id"] == f"u{i}" and m.get("passage_id") == d.passage_id
            for m in result["matches"]
        )
    for m in result["matches"]:
        for o in m["source_occurrences"]:
            assert (
                corpus.source_documents[o["source_document_id"]][
                    o["source_start_char"] : o["source_end_char"]
                ]
                == m["source_text"]
            )
    return dict(
        scope="Exact excerpts from actual ingested Wikipedia prose; workflow smoke, not semantic accuracy or independent labels.",
        submission=submission,
        expected_passage_ids=[d.passage_id for d in selected],
        report=result,
        elapsed_seconds=elapsed,
        external_network_attempts=len(attempts),
        model_calls=0,
        api_cost_usd=0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--seed", type=Path, default=SEED)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--max-documents", type=int, default=12)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--resume")
    ingest.add_argument("--max-documents", type=int, default=12)
    ingest.add_argument("--max-total-bytes", type=int, default=25 * 1024 * 1024)
    ingest.add_argument("--max-response-bytes", type=int, default=2 * 1024 * 1024)
    ingest.add_argument("--max-requests", type=int, default=64)
    export = commands.add_parser("export-lock")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    stats = commands.add_parser("stats")
    demo = commands.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    for command in (ingest, export, stats, demo):
        command.add_argument("--db", type=Path)
    args = p.parse_args()
    if getattr(args, "output", None) and args.output.exists():
        p.error("Output exists; preserve it and choose a new path.")
    if args.command == "demo" and args.output.with_suffix(".submission.txt").exists():
        p.error("Demo submission output exists; choose a new output path.")
    if args.command == "resolve":
        if not 1 <= args.max_documents <= 50:
            p.error("Resolve between one and fifty titles per manifest.")
        client = WikipediaClient()
        manifest = client.resolve(
            json.loads(args.seed.read_text(encoding="utf-8"))["titles"][
                : args.max_documents
            ]
        )
        write_new(args.output, manifest)
        print(
            json.dumps(
                dict(
                    resolved=len(manifest["entries"]),
                    failures=manifest["resolve_failures"],
                    http_requests=client.requests,
                    bytes=client.bytes_read,
                )
            )
        )
        return
    store = SourceCorpusStore(args.db)
    if args.command == "ingest":
        client = WikipediaClient(
            max_total_bytes=args.max_total_bytes,
            max_response_bytes=args.max_response_bytes,
            max_requests=args.max_requests,
        )
        run = ingest_manifest(
            store,
            load_manifest(args.manifest),
            client,
            limit=args.max_documents,
            resume=args.resume,
        )
        print(json.dumps({k: run[k] for k in ("run_id", "status", "metrics")}))
        if run["status"] != "COMPLETE":
            raise SystemExit(1)
    elif args.command == "export-lock":
        result = export_lock(store, load_manifest(args.manifest))
        write_new(args.output, result)
        print(
            json.dumps(
                dict(indexed_sources=result["indexed_sources"], corpus=result["corpus"])
            )
        )
    elif args.command == "stats":
        print(json.dumps(store.stats()))
    elif args.command == "demo":
        result = demonstrate(store)
        write_new(args.output, result)
        with args.output.with_suffix(".submission.txt").open(
            "x", encoding="utf-8"
        ) as f:
            f.write(result["submission"])
        print(
            json.dumps(
                dict(
                    matches=len(result["report"]["matches"]),
                    coverage=result["report"]["coverage_percent"],
                    seconds=result["elapsed_seconds"],
                    model_calls=0,
                )
            )
        )


if __name__ == "__main__":
    main()
