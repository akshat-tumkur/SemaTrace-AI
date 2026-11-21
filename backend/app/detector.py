"""Deterministic offline upload detector. No agents, network or paid providers."""

from dataclasses import asdict
import hashlib, time, re
from app.alignment.lexical import align
from app.alignment.exact import exact_runs
from app.retrieval.hybrid import HybridRetriever
from app.parsing.sentences import split_into_units
from app.scoring import build_report
from app.corpus.models import CorpusUnavailable

# RRF retained12/12 raw-query sources,11/12 after actual upload normalization.
# Every retained/dropped source is reported; larger-corpus quality is not implied.
MAX_SOURCE_DOCUMENTS = 10


def run_detection(filename, file_type, text, corpus, *, retriever=None, units=None):
    if not corpus.documents:
        raise CorpusUnavailable("No indexed sources are available")
    started = time.perf_counter()
    units = units if units is not None else split_into_units(text)
    submission_sha = hashlib.sha256(text.encode()).hexdigest()
    retriever = retriever or HybridRetriever(corpus)
    ranking = retriever.rank(units)
    selected = ranking["documents"][:MAX_SOURCE_DOCUMENTS]
    sources = {}
    for sid in selected:
        owner = retriever.owners[sid]
        source = corpus.source_documents[sid]
        if hashlib.sha256(source.encode()).hexdigest() != owner.source_document_sha256:
            raise CorpusUnavailable("Source text integrity mismatch")
        sources[sid] = {
            "text": source,
            "title": owner.title,
            "url": owner.url,
            "revision": owner.revision,
            "source_id": owner.source_id,
            "sha256": owner.source_document_sha256,
            "attribution": {
                k: getattr(owner, k)
                for k in (
                    "authors",
                    "collection",
                    "license_url",
                    "license_name",
                    "rights_status",
                    "attribution_text",
                    "attribution_url",
                )
            },
        }
    findings = []
    candidates = []
    for sid, source in sources.items():
        passages = align(text, source["text"])
        brief = [
            r
            for r in exact_runs(text, source["text"])
            if not any(
                p["submission_start"] <= r["submission_start"]
                and r["submission_end"] <= p["submission_end"]
                and p["source_start"] <= r["source_start"]
                and r["source_end"] <= p["source_end"]
                for p in passages
            )
        ]
        for finding in passages + brief:
            a, b = finding["submission_start"], finding["submission_end"]
            x, y = finding["source_start"], finding["source_end"]
            fid = hashlib.sha256(
                f"{submission_sha}:{sid}:{a}:{b}:{x}:{y}".encode()
            ).hexdigest()
            marker = re.match(
                r"[\s.,;]*(\[\d+\]|\([^()\n]*\d{4}[^()\n]*\))", text[b : b + 64]
            )
            inside = re.search(r"\[\d+\]|\([^()\n]*\d{4}[^()\n]*\)", text[a:b])
            citation = (
                inside.group(0) if inside else marker.group(1) if marker else None
            )
            findings.append(
                dict(
                    finding,
                    finding_id=fid,
                    submission_document_id="submission:" + submission_sha,
                    source_document_id=sid,
                    source_revision=source["revision"],
                    source_document_sha256=source["sha256"],
                    source_title=source["title"],
                    source_url=source["url"],
                    submission_excerpt=text[a:b],
                    source_excerpt=source["text"][x:y],
                    source_attribution=source["attribution"],
                    citation_context=citation,
                    uncertainty="Lexical correspondence; no misconduct conclusion.",
                )
            )
            candidates.append(
                {
                    "candidate_id": fid,
                    "unit_id": fid,
                    "submission_text": text[a:b],
                    "source_text": source["text"][x:y],
                    "source_title": source["title"],
                    "source_url": source["url"],
                    "source_document_id": sid,
                    "source_id": source["source_id"],
                    "source_start_char": x,
                    "source_end_char": y,
                    "verification_method": "deterministic",
                    "verification_score": 1.0,
                    "final_match_status": "MATCH",
                    "match_type": "EXACT_OR_LEXICAL_OVERLAP",
                    "source_revision": source["revision"],
                    "citation_present": citation is not None,
                    "citation_markers": [citation] if citation else [],
                    "citation_support_status": "NOT_ASSESSED",
                    "source_occurrences": [
                        dict(
                            source_document_id=sid,
                            source_id=source["source_id"],
                            source_start_char=x,
                            source_end_char=y,
                            revision=source["revision"],
                            title=source["title"],
                            url=source["url"],
                            source_document_sha256=source["sha256"],
                        )
                    ],
                }
            )
    report = build_report(text, findings, sources)
    related = []
    from app.discovery.evidence import candidate

    for row in ranking["related"]:
        unit = units[row["unit_index"]]
        related.append(
            candidate(
                row, unit, text, corpus.source_documents[row["source_document_id"]]
            )
        )
    return {
        "report_version": "3.0",
        "filename": filename,
        "file_type": file_type,
        "status": "completed",
        "mode": "Deterministic local document alignment",
        "risk_level": "NOT_ASSESSED",
        "risk_score": 0.0,
        "coverage_percent": report["textual_similarity_percent"],
        "legacy_coverage_percent": 0.0,
        "verified_coverage_percent": 0.0,
        "risk_score_method": "NO_MISCONDUCT_SCORE",
        "strong_matches": len(findings),
        "possible_paraphrases": 0,
        "false_positives": 0,
        "candidate_count": len(selected),
        "partial_overlaps": 0,
        "manual_review_count": 0,
        "evidence_categories": {"lexical_passages": len(findings)},
        "matches": candidates,
        "verification": {
            "candidates": candidates,
            "unresolved_candidate_ids": [],
            "metrics": {"llm_calls": 0, "http_attempts": 0},
        },
        "findings": findings,
        "related_sources": related,
        "alignment_sources": {
            sid: s
            for sid, s in sources.items()
            if any(f["source_document_id"] == sid for f in findings)
        },
        "similarity_report": report,
        "retrieval": {
            "method": ranking["method"],
            "rankings": ranking["rankings"],
            "retained_sources": selected,
            "dropped_sources": ranking["documents"][MAX_SOURCE_DOCUMENTS:],
            "document_budget": MAX_SOURCE_DOCUMENTS,
            "generation_id": ranking.get("generation_id"),
            "query_count": ranking.get("query_count", len(units)),
            "query_batch_size": ranking.get("query_batch_size"),
        },
        "units": [asdict(u) for u in units],
        "units_analyzed": len(units),
        "document": {
            "document_id": "submission:" + submission_sha,
            "normalized_sha256": submission_sha,
            "text": text,
            "text_basis": "normalized_extracted_text",
            "offset_unit": "unicode_code_points",
            "end_exclusive": True,
            "layout_coordinates_available": False,
            "note": "Extracted-text offsets; not PDF page coordinates.",
        },
        "corpus": dict(
            corpus.corpus_info, dense_status="LOCAL_LEARNED_PERSISTENT_GENERATION"
        ),
        "summary": f"{len(findings)} lexical passage correspondences. Related sources are discovery results, not validated semantic reuse.",
        "recommendations": [
            "Inspect source attribution and relevant context.",
            "Use exclusions for quoted material or bibliography ranges where appropriate.",
        ],
        "report_synthesis": {
            "semantic_level": 1,
            "semantic_coverage": None,
            "combined_similarity": None,
        },
        "audit_log": [
            {
                "agent": "Local detector",
                "event": "completed",
                "details": f"Ranked {len(ranking['documents'])} source versions; aligned {len(selected)}. No external services. {time.perf_counter() - started:.2f}s.",
            }
        ],
    }
