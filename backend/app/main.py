from __future__ import annotations

from typing import Any
import hashlib
import sqlite3
import os
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from datetime import datetime, timezone
from threading import BoundedSemaphore
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.parsing.document import parse_document
from app.detector import run_detection
from app.alignment.lexical import AlignmentLimit
from app.retrieval.corpus import load_local_corpus
from app.storage.database import AnalysisStore
from app.corpus.models import CorpusUnavailable, SourceMetadata
from app.storage.source_corpus import SourceCorpusStore, digest
from app.retrieval.generations import IndexUnavailable
from app.limits import RequestLimits, MAX_FILE_SIZE, MAX_ANALYSIS_UNITS, analysis_slot
from app.privacy import LocalPrivacy
from app.parsing.sentences import split_into_units

app = FastAPI(
    title="SemaTrace AI",
    description="Evidence-backed semantic similarity analysis API.",
    version="3.1.0",
)
app.add_middleware(RequestLimits)
app.add_middleware(LocalPrivacy)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        s.strip()
        for s in os.getenv(
            "SEMATRACE_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],backend"
        ).split(",")
        if s.strip()
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        s.strip()
        for s in os.getenv("SEMATRACE_CORS_ORIGINS", "http://localhost:5173").split(",")
        if s.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

local_corpus = load_local_corpus()
analysis_store = AnalysisStore()
pdf_export_slot = BoundedSemaphore(1)


def get_source_store() -> SourceCorpusStore:
    """Per-request repository; reads CLI commits and supports FastAPI overrides."""
    try:
        store = SourceCorpusStore()
        if os.getenv("SEMATRACE_IMPORT_LEGACY_CORPUS", "true").lower() == "true":
            store.import_legacy(local_corpus)
        return store
    except (OSError, sqlite3.Error, CorpusUnavailable) as error:
        raise HTTPException(
            503, "Source corpus is unavailable; no substitute index was used."
        ) from error


class AnalysisSummary(BaseModel):
    analysis_id: str
    created_at: str | None = None
    filename: str
    status: str
    risk_level: str
    coverage_percent: float
    strong_matches: int
    possible_paraphrases: int
    false_positives: int
    candidate_count: int = 0
    matches: list[dict[str, Any]] = []
    risk_score: float = 0.0
    mode: str = "Deterministic local document alignment"
    audit_log: list[dict[str, str]] = []
    summary: str = ""
    recommendations: list[str] = []
    message: str | None = None
    legacy_coverage_percent: float = 0.0
    verified_coverage_percent: float = 0.0
    risk_score_method: str = "legacy_deterministic_v1"
    partial_overlaps: int = 0
    manual_review_count: int = 0
    evidence_categories: dict[str, int] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    corpus: dict[str, Any] | None = None
    units: list[dict[str, Any]] = Field(default_factory=list)
    units_analyzed: int = 0
    document: dict[str, Any] | None = None
    report_synthesis: dict[str, Any] | None = None
    similarity_report: dict[str, Any] | None = None
    report_version: str = "2.0"
    findings: list[dict[str, Any]] = Field(default_factory=list)
    related_sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval: dict[str, Any] | None = None
    semantic_review: dict[str, Any] = Field(default_factory=dict)
    semantic_assessment: dict[str, Any] = Field(default_factory=dict)
    web_discovery: dict[str, Any] = Field(default_factory=dict)
    processing: dict[str, Any] = Field(default_factory=dict)


class SimilarityExcludedRange(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    reason: str = Field(
        default="EXPLICIT_DOCUMENT_EXCLUSION", min_length=1, max_length=200
    )

    @model_validator(mode="after")
    def ordered_range(self):
        if self.end <= self.start:
            raise ValueError("An exclusion range must have start < end")
        if not self.reason.strip():
            raise ValueError("An exclusion needs a reason")
        return self


class SimilarityOptionsRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    minimum_match_words: int = Field(default=5, ge=3, le=100)
    excluded_source_ids: list[str] = Field(default_factory=list, max_length=100)
    excluded_ranges: list[SimilarityExcludedRange] = Field(
        default_factory=list, max_length=100
    )

    @model_validator(mode="after")
    def source_ids(self):
        if any(
            not value.strip() or len(value) > 300 for value in self.excluded_source_ids
        ):
            raise ValueError("Invalid excluded source ID")
        return self


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "sematrace-api",
        "storage": "sqlite",
        "workflow": "deterministic-document-alignment-v3",
        "web_search": "disabled",
        "reasoning": "deterministic-fallback",
        "evidence_verification": "deterministic-fallback",
        "source_uploads": "enabled"
        if os.getenv("SEMATRACE_SOURCE_UPLOAD_ENABLED", "true").lower() == "true"
        else "disabled",
    }


@app.get("/api/ready")
def ready(source_store: SourceCorpusStore = Depends(get_source_store)) -> dict:
    """Read-only readiness; never builds an index or invokes a paid provider."""
    try:
        corpus = source_store.snapshot()
        if not corpus.documents:
            raise CorpusUnavailable("empty_corpus")
        provider = "local-onnx"
        generation_id = None
        if provider == "local-onnx":
            from app.retrieval.generations import GenerationStore, IndexSpec
            from app.retrieval.local_model import verify_directory, default_directory

            verify_directory(default_directory())
            generation, error = GenerationStore(corpus.retrieval_index_path).active(
                corpus, IndexSpec.local()
            )
            if generation is None or error:
                raise IndexUnavailable("local_generation_unavailable")
            generation_id = generation.id
        return {
            "status": "ready",
            "passages": len(corpus.documents),
            "dense_provider": provider,
            "generation_id": generation_id,
            "semantic_support": "NOT_ESTABLISHED_BY_RETRIEVAL",
        }
    except (CorpusUnavailable, IndexUnavailable, OSError, sqlite3.Error) as error:
        raise HTTPException(
            503,
            "Corpus, local model or compatible retrieval generation is unavailable.",
        ) from error
    except RuntimeError as error:
        # LocalModelUnavailable is deliberately public only as this safe message.
        raise HTTPException(
            503, "Local retrieval model failed readiness validation."
        ) from error


@app.post("/api/analyze", response_model=AnalysisSummary, status_code=202)
async def create_analysis(
    file: UploadFile = File(...),
    corpus_version: str | None = Form(None),
    source_store: SourceCorpusStore = Depends(get_source_store),
) -> AnalysisSummary:
    if not analysis_slot.acquire(blocking=False):
        raise HTTPException(
            429,
            "An investigation is already running; retry when it completes.",
            headers={"Retry-After": "10"},
        )
    try:
        return await _create_analysis(file, corpus_version, source_store)
    finally:
        analysis_slot.release()


async def _create_analysis(file, corpus_version, source_store):
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
    parsed = await run_in_threadpool(
        parse_document, file.filename or "document.txt", content
    )
    unit_count = len(split_into_units(parsed.text))
    if unit_count > MAX_ANALYSIS_UNITS:
        raise HTTPException(
            413,
            f"Document contains {unit_count:,} sentences or text blocks; the limit is {MAX_ANALYSIS_UNITS:,} within 50,000 extracted characters.",
        )

    analysis_id = f"a_{uuid4().hex[:12]}"
    try:

        def investigate():
            snapshot = source_store.snapshot(version_id=corpus_version)
            return run_detection(
                parsed.filename, parsed.file_type, parsed.text, snapshot
            )

        result = await run_in_threadpool(investigate)
    except (CorpusUnavailable, IndexUnavailable) as error:
        raise HTTPException(
            503, "Source or retrieval index unavailable or failed integrity validation."
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            503, "The local model is unavailable; no substitute analysis was produced."
        ) from error
    except AlignmentLimit as error:
        raise HTTPException(
            413, "Analysis exceeded the bounded lexical alignment workload."
        ) from error
    except ValueError as error:
        raise HTTPException(
            500, "Evidence validation failed; no successful report was saved."
        ) from error
    result["analysis_id"] = analysis_id
    result["document"].update(
        original_text=parsed.original_text,
        raw_file_sha256=hashlib.sha256(content).hexdigest(),
        normalization_map={
            "starts": parsed.normalized_starts,
            "ends": parsed.normalized_ends,
        },
        extraction_version="document-parser-v2-offset-map",
        original_offset_basis="original extracted Unicode text, not PDF page coordinates",
    )
    from app.discovery.evidence import original_offsets

    original_offsets(result["document"], result["related_sources"])
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    try:
        await run_in_threadpool(analysis_store.save, analysis_id, result)
    except (sqlite3.Error, OSError) as error:
        raise HTTPException(
            503, "The report could not be saved; please retry later."
        ) from error
    return AnalysisSummary(message="Similarity analysis completed", **result)


@app.post("/api/sources", status_code=201)
async def add_source(
    file: UploadFile = File(...),
    title: str = Form(..., min_length=1, max_length=1000),
    url: str = Form(..., min_length=1, max_length=2000),
    source_store: SourceCorpusStore = Depends(get_source_store),
) -> dict[str, Any]:
    if os.getenv("SEMATRACE_SOURCE_UPLOAD_ENABLED", "true").lower() != "true":
        raise HTTPException(403, "Source uploads are disabled on this instance.")
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Source exceeds the 10 MB limit.")
    parsed = parse_document(file.filename or "source.txt", content)
    meta = SourceMetadata(
        collection="upload",
        external_id=digest(url + "\0" + parsed.text),
        title=title,
        url=url,
        revision=digest(content),
        extractor_version="document-parser-v1",
        rights_status="PRIVATE_UPLOAD",
    )
    try:

        def index_source():
            result = source_store.ingest(meta, parsed.text, content)
            from app.retrieval.generations import GenerationStore, IndexSpec

            corpus = source_store.snapshot()
            generation, metrics = GenerationStore(corpus.retrieval_index_path).build(
                corpus, IndexSpec.local(), api_key="", batch_size=32
            )
            return dict(result, generation_id=generation.id, indexing=metrics)

        if not analysis_slot.acquire(blocking=False):
            raise HTTPException(429, "Analysis or source indexing is already running.")
        try:
            result = await run_in_threadpool(index_source)
        finally:
            analysis_slot.release()
    except (CorpusUnavailable, sqlite3.Error) as error:
        raise HTTPException(
            503,
            "Source storage or index validation failed; inspect storage and rebuild the local retrieval index before analysis.",
        ) from error
    except ValueError as error:
        raise HTTPException(
            422, "Source contains no indexable passages or invalid source metadata."
        ) from error
    except (IndexUnavailable, RuntimeError) as error:
        raise HTTPException(
            503,
            "Source storage completed but the local model/index requires repair before analysis.",
        ) from error
    return {**result, "title": title, "url": url}


@app.get("/api/corpus")
def get_corpus(
    source_store: SourceCorpusStore = Depends(get_source_store),
) -> dict[str, Any]:
    try:
        return source_store.stats()
    except CorpusUnavailable as error:
        raise HTTPException(503, "Source index integrity check failed.") from error


@app.get("/api/corpus/sources")
def list_sources(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    collection: str | None = None,
    source_store: SourceCorpusStore = Depends(get_source_store),
) -> list[dict]:
    return source_store.list_sources(offset, limit, collection)


@app.get("/api/corpus/snapshots/{version_id}")
def get_corpus_snapshot(
    version_id: str, source_store: SourceCorpusStore = Depends(get_source_store)
) -> dict:
    try:
        return source_store.snapshot(version_id=version_id).corpus_info
    except CorpusUnavailable as error:
        raise HTTPException(
            503, "Corpus version unavailable or failed integrity validation."
        ) from error


@app.get("/api/corpus/versions/{version_id}")
def get_source_version(
    version_id: str, source_store: SourceCorpusStore = Depends(get_source_store)
) -> dict:
    try:
        result = source_store.get_version(version_id)
    except CorpusUnavailable as error:
        raise HTTPException(503, "Source snapshot integrity check failed.") from error
    if result is None:
        raise HTTPException(404, "Source version not found.")
    return result


@app.get("/api/corpus/runs/{run_id}")
def get_ingestion_run(
    run_id: str, source_store: SourceCorpusStore = Depends(get_source_store)
) -> dict:
    result = source_store.get_run(run_id)
    if result is None:
        raise HTTPException(404, "Ingestion run not found.")
    return result


@app.get("/api/analyses")
def list_analyses(
    offset: int = Query(0, ge=0, le=100000), limit: int = Query(20, ge=1, le=100)
) -> list[dict]:
    return analysis_store.list_reports(offset, limit)


@app.delete("/api/analysis/{analysis_id}")
def delete_analysis(analysis_id: str):
    if not analysis_store.delete(analysis_id):
        raise HTTPException(404, "Analysis not found.")
    from app.product.jobs import JobStore

    job_path = analysis_store.database_path.with_suffix(".jobs.sqlite3")
    if job_path.exists():
        JobStore(job_path).forget_analysis(analysis_id)
    return {"deleted": analysis_id}


@app.get("/api/analysis/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    result = analysis_store.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {key: value for key, value in result.items() if key != "alignment_sources"}


@app.get("/api/analysis/{analysis_id}/matches")
def get_matches(analysis_id: str) -> list[dict[str, Any]]:
    result = analysis_store.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return result["matches"]


@app.post("/api/analysis/{analysis_id}/similarity")
async def recalculate_similarity(
    analysis_id: str, options: SimilarityOptionsRequest
) -> dict[str, Any]:
    """Recalculate from saved evidence only: no retrieval or model calls."""
    try:
        result = await run_in_threadpool(analysis_store.get, analysis_id)
        if result is None:
            raise HTTPException(404, "Analysis not found.")
        document = result.get("document")
        if not isinstance(document, dict) or not isinstance(document.get("text"), str):
            raise HTTPException(
                409,
                "This older report has no normalized document text; similarity cannot be reconstructed safely.",
            )
        text = document["text"]
        if any(row.end > len(text) for row in options.excluded_ranges):
            raise HTTPException(
                422, "An exclusion range exceeds the normalized document."
            )
        if result.get("report_version") == "3.0":
            from app.scoring import build_report

            report = await run_in_threadpool(
                build_report,
                text,
                result["findings"],
                result["alignment_sources"],
                options.model_dump(),
            )
        else:
            from app.similarity import build_similarity_report

            report = await run_in_threadpool(
                build_similarity_report,
                text,
                result.get("units", []),
                result.get("verification", {}).get("candidates", []),
                "submission:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
                options.model_dump(),
            )
        if not await run_in_threadpool(
            analysis_store.update_similarity_report, analysis_id, report
        ):
            raise HTTPException(404, "Analysis not found.")
        return report
    except (sqlite3.Error, OSError) as error:
        raise HTTPException(
            503, "The similarity report could not be read or saved."
        ) from error
    except ValueError as error:
        raise HTTPException(
            422, "Invalid similarity exclusion options or saved evidence."
        ) from error


@app.get("/api/analysis/{analysis_id}/report")
def get_report(analysis_id: str) -> dict[str, Any]:
    result = analysis_store.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {
        "product": "SemaTrace AI",
        "report_type": "evidence-backed similarity assessment",
        "analysis_id": analysis_id,
        "filename": result["filename"],
        "risk_level": result["risk_level"],
        "risk_score": result["risk_score"],
        "coverage_percent": result["coverage_percent"],
        "summary": result["summary"],
        "recommendations": result["recommendations"],
        "matches": result["matches"],
        "audit_log": result["audit_log"],
        "units": result.get("units", []),
        **{
            key: result[key]
            for key in (
                "legacy_coverage_percent",
                "verified_coverage_percent",
                "risk_score_method",
                "partial_overlaps",
                "manual_review_count",
                "evidence_categories",
                "verification",
                "corpus",
                "document",
                "report_synthesis",
                "similarity_report",
                "created_at",
                "report_version",
                "findings",
                "related_sources",
                "retrieval",
                "semantic_review",
                "semantic_assessment",
                "web_discovery",
                "processing",
            )
            if key in result
        },
    }


@app.get("/api/analysis/{analysis_id}/report.pdf")
async def download_pdf(analysis_id: str):
    """Render persisted values and exclusions; never rerun detection to export."""
    try:
        result = await run_in_threadpool(analysis_store.get, analysis_id)
    except (sqlite3.Error, OSError) as error:
        raise HTTPException(503, "Saved report storage is unavailable.") from error
    if result is None:
        raise HTTPException(404, "Analysis not found.")
    if not pdf_export_slot.acquire(blocking=False):
        raise HTTPException(
            429,
            "A PDF export is already running; retry shortly.",
            headers={"Retry-After": "5"},
        )
    try:
        try:
            from app.reports.pdf import render_pdf
            from reportlab.platypus.doctemplate import LayoutError

            payload = await run_in_threadpool(render_pdf, result)
        except (OSError, RuntimeError, ImportError) as error:
            raise HTTPException(
                503, "PDF export is unavailable; the saved JSON remains available."
            ) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except LayoutError as error:
            raise HTTPException(
                409,
                "This report cannot fit the PDF layout; download the complete JSON instead.",
            ) from error
    finally:
        pdf_export_slot.release()
    return Response(
        payload,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="sematrace-report.pdf"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


class SemanticReviewRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    candidate_id: str = Field(pattern=r"^[a-f0-9]{64}$")


@app.get("/api/semantic/capabilities")
def semantic_capabilities():
    import importlib.util

    return {
        "automatic_discovery": "MiniLM; review only",
        "default_order": "minilm",
        "roberta_configured": bool(os.getenv("SEMATRACE_REVIEW_MODEL_DIR")),
        "investigation_enabled": os.getenv(
            "SEMATRACE_INVESTIGATION_ENABLED", "false"
        ).lower()
        == "true"
        and importlib.util.find_spec("langgraph") is not None,
        "provider": "local Ollama only; no external documents or paid API",
        "max_rerank_pairs": 8,
        "transfer_result": "MiniLM Hit@1 16/16; RoBERTa 14/16 on a small frozen controlled fixture. No demonstrated source-ranking improvement.",
    }


async def semantic_action(analysis_id, request, source_store, kind):
    if not analysis_slot.acquire(blocking=False):
        raise HTTPException(
            429, "An analysis or semantic review is running; retry when it finishes."
        )
    try:
        saved = await run_in_threadpool(analysis_store.get, analysis_id)
        if saved is None:
            raise HTTPException(404, "Analysis not found.")
        if saved.get("report_version") != "3.0":
            raise HTTPException(
                409,
                "Reanalyze this legacy report to discover grounded semantic candidates.",
            )
        from app.discovery.evidence import selected_rows

        selected, rows = selected_rows(saved, request.candidate_id)
        # Each submitted sentence has one immutable candidate pool/ranking.
        key = (
            str(selected["submission_start"])
            if kind == "rankings"
            else request.candidate_id
        )
        existing = saved.get("semantic_review", {}).get(kind, {}).get(key)
        if existing is not None:
            return existing
        if kind == "rankings":
            from app.discovery.evidence import rerank

            value = await run_in_threadpool(
                rerank, saved, request.candidate_id, source_store
            )
        else:
            try:
                from app.discovery.investigation import investigate

                value = await run_in_threadpool(
                    investigate, saved, request.candidate_id, source_store
                )
            except ImportError as error:
                raise HTTPException(
                    503,
                    "Install requirements-investigation.txt and configure local Ollama to enable investigation.",
                ) from error
        value["created_at"] = datetime.now(timezone.utc).isoformat()
        if not await run_in_threadpool(
            analysis_store.save_semantic_review, analysis_id, kind, key, value
        ):
            raise HTTPException(404, "Analysis was deleted during review.")
        return value
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (RuntimeError, OSError, sqlite3.Error) as error:
        raise HTTPException(
            503,
            "Optional semantic review is unavailable. Check local model configuration; saved textual evidence remains available.",
        ) from error
    finally:
        analysis_slot.release()


@app.post("/api/analysis/{analysis_id}/semantic/rerank")
async def rerank_semantic(
    analysis_id: str,
    request: SemanticReviewRequest,
    source_store: SourceCorpusStore = Depends(get_source_store),
):
    return await semantic_action(analysis_id, request, source_store, "rankings")


@app.post("/api/analysis/{analysis_id}/semantic/investigate")
async def investigate_semantic(
    analysis_id: str,
    request: SemanticReviewRequest,
    source_store: SourceCorpusStore = Depends(get_source_store),
):
    return await semantic_action(analysis_id, request, source_store, "investigations")


from app.product.routes import install_product_routes

install_product_routes(app, lambda: analysis_store, get_source_store)
