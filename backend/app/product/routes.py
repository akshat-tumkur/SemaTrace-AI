"""Durable upload jobs, explicit external-service choices, and saved AI review."""

import asyncio, hashlib, os, time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from app.limits import MAX_FILE_SIZE, MAX_ANALYSIS_UNITS, MAX_TEXT_CHARS, analysis_slot
from app.parsing.document import parse_document
from app.parsing.sentences import split_into_units
from app.product.jobs import JobStore


def install_product_routes(app, get_analysis_store, source_dependency):
    tasks = set()

    def jobs():
        return JobStore(get_analysis_store().database_path.with_suffix(".jobs.sqlite3"))

    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def product_lifespan(application):
        jobs().recover()
        async with previous_lifespan(application):
            yield
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    app.router.lifespan_context = product_lifespan

    @app.get("/api/product/status")
    def product_status():
        return {
            "version": "3.1",
            "cloud_review_available": bool(os.getenv("OPENAI_API_KEY")),
            "web_search_available": bool(os.getenv("TAVILY_API_KEY")),
            "review_model": os.getenv("SEMATRACE_REVIEW_MODEL")
            or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "source_uploads_enabled": os.getenv(
                "SEMATRACE_SOURCE_UPLOAD_ENABLED", "true"
            ).lower()
            == "true",
            "limits": {
                "file_megabytes": 10,
                "characters": MAX_TEXT_CHARS,
                "text_units": MAX_ANALYSIS_UNITS,
                "ai_units_per_review": 48,
                "ai_calls_per_review": 8,
                "web_queries": 2,
                "additional_web_sources": 3,
            },
            "privacy": "Local analysis stays on this computer. Opt-in AI review sends selected excerpts to OpenAI; opt-in web discovery sends two representative queries to Tavily and captures public pages.",
        }

    def perform(job_id, filename, content, options, source_store, analysis_store):
        records = JobStore(analysis_store.database_path.with_suffix(".jobs.sqlite3"))
        began = time.perf_counter()

        def progress(stage, percent):
            records.update(job_id, stage=stage, progress=percent)

        try:
            progress("Extracting text and original offsets", 8)
            parsed = parse_document(filename, content)
            units = split_into_units(parsed.text, preserve_urls=True)
            if len(units) > MAX_ANALYSIS_UNITS:
                raise ValueError(
                    f"Document has {len(units):,} text units; the limit is {MAX_ANALYSIS_UNITS:,}."
                )
            web = {"status": "disabled", "sources": [], "queries": [], "warnings": []}
            if options["web_search"]:
                from app.product.web import discover

                progress("Searching for additional public sources", 15)
                web = discover(parsed.text, source_store, progress)
            progress("Retrieving indexed sources and aligning text", 40)
            from app.detector import run_detection

            result = run_detection(
                parsed.filename,
                parsed.file_type,
                parsed.text,
                source_store.snapshot(),
                units=units,
            )
            result["analysis_id"] = "a_" + uuid4().hex[:12]
            result["created_at"] = datetime.now(timezone.utc).isoformat()
            result["document"].update(
                original_text=parsed.original_text,
                raw_file_sha256=hashlib.sha256(content).hexdigest(),
                normalization_map={
                    "starts": parsed.normalized_starts,
                    "ends": parsed.normalized_ends,
                },
                extraction_version="document-parser-v2-offset-map",
                segmentation_version="submission-url-aware-v1",
                original_offset_basis="original extracted Unicode text, not PDF page coordinates",
            )
            from app.discovery.evidence import original_offsets

            original_offsets(result["document"], result["related_sources"])
            result["web_discovery"] = web
            if options["cloud_review"]:
                from app.product.review import review_saved

                progress("Reviewing semantic passage evidence", 74)
                result["semantic_assessment"] = review_saved(
                    result, source_store, progress=progress
                )
            else:
                result["semantic_assessment"] = {
                    "status": "disabled",
                    "findings": [],
                    "note": "AI review was not requested. Source retrieval and matching-word analysis were completed locally.",
                }
            result["processing"] = {
                "method": "product-upload-workflow-v1",
                "options": options,
                "elapsed_seconds": time.perf_counter() - began,
                "coverage_scope": "All accepted text units were retrieved; AI review has a separate explicit budget.",
                "job_id": job_id,
            }
            assessment = result["semantic_assessment"]
            count = len(assessment.get("findings", []))
            result["summary"] = (
                f"{len(result['findings'])} lexical passage correspondences; {count} AI-identified semantic correspondences for human review."
            )
            result["audit_log"].append(
                {
                    "agent": "Product workflow",
                    "event": "completed",
                    "details": f"AI review: {assessment['status']}; {count} grounded advisory findings. Web: {web['status']}. Matching-word coverage is independent.",
                }
            )
            progress("Saving source-linked report", 97)
            analysis_store.save(result["analysis_id"], result)
            records.update(
                job_id,
                status="completed",
                stage="Report ready",
                progress=100,
                analysis_id=result["analysis_id"],
            )
        except Exception as error:
            if isinstance(error, HTTPException):
                message = str(error.detail)
            elif isinstance(error, ValueError):
                message = str(error)[:400]
            else:
                message = "Analysis could not complete. Check the indexed source/model setup and retry. No successful partial report was saved."
            records.update(
                job_id, status="failed", stage="Analysis stopped", error=message
            )

    async def worker(job_id, filename, content, options, source_store, analysis_store):
        try:
            await run_in_threadpool(
                perform,
                job_id,
                filename,
                content,
                options,
                source_store,
                analysis_store,
            )
        finally:
            analysis_slot.release()

    @app.post("/api/analysis-jobs", status_code=202)
    async def new_job(
        file: UploadFile = File(...),
        cloud_review: bool = Form(False),
        web_search: bool = Form(False),
        source_store=Depends(source_dependency),
    ):
        if not analysis_slot.acquire(blocking=False):
            raise HTTPException(
                429,
                "Another analysis or review is running. Please wait for it to finish.",
            )
        launched = False
        try:
            content = await file.read(MAX_FILE_SIZE + 1)
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(413, "File exceeds the10MB limit.")
            key = "job_" + uuid4().hex[:16]
            options = {"cloud_review": cloud_review, "web_search": web_search}
            value = jobs().create(
                key,
                (file.filename or "document.txt")
                .replace("\\", "/")
                .rsplit("/", 1)[-1][:255],
                options,
            )
            task = asyncio.create_task(
                worker(
                    key,
                    file.filename or "document.txt",
                    content,
                    options,
                    source_store,
                    get_analysis_store(),
                )
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            launched = True
            return value
        finally:
            if not launched:
                analysis_slot.release()

    @app.get("/api/analysis-jobs/{job_id}")
    def read_job(job_id: str):
        result = jobs().get(job_id)
        if result is None:
            raise HTTPException(404, "Analysis job not found.")
        return result

    class ReviewOptions(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        cloud_review: bool = False

    class InvestigationOptions(ReviewOptions):
        candidate_id: str = Field(pattern=r"^[a-f0-9]{64}$")

    @app.post("/api/analysis/{analysis_id}/cloud-investigate")
    async def cloud_investigate(
        analysis_id: str,
        options: InvestigationOptions,
        source_store=Depends(source_dependency),
    ):
        if not options.cloud_review:
            raise HTTPException(422, "Explicit cloud_review permission is required.")
        if not analysis_slot.acquire(blocking=False):
            raise HTTPException(429, "Another analysis or review is running.")
        try:
            store = get_analysis_store()
            saved = await run_in_threadpool(store.get, analysis_id)
            if saved is None:
                raise HTTPException(404, "Analysis not found.")
            key = "openai:" + options.candidate_id
            cached = saved.get("semantic_review", {}).get("investigations", {}).get(key)
            if cached:
                return cached
            from app.product.providers import OpenAIReviewer
            from app.discovery.investigation import investigate

            provider = OpenAIReviewer(max_calls=3)
            result = await run_in_threadpool(
                investigate, saved, options.candidate_id, source_store, provider
            )
            result["usage"] = provider.usage
            result["calls"] = provider.calls
            if not await run_in_threadpool(
                store.save_semantic_review, analysis_id, "investigations", key, result
            ):
                raise HTTPException(404, "Analysis was deleted during investigation.")
            return result
        except (RuntimeError, ValueError, ImportError) as error:
            raise HTTPException(
                503,
                "Investigation is unavailable. The saved source and AI review evidence remain available.",
            ) from error
        finally:
            analysis_slot.release()

    @app.post("/api/analysis/{analysis_id}/ai-review")
    async def more_review(
        analysis_id: str,
        options: ReviewOptions,
        source_store=Depends(source_dependency),
    ):
        if not options.cloud_review:
            raise HTTPException(
                422,
                "Enable cloud_review explicitly to send selected excerpts to OpenAI.",
            )
        if not analysis_slot.acquire(blocking=False):
            raise HTTPException(429, "Another analysis or review is running.")
        try:
            store = get_analysis_store()
            saved = await run_in_threadpool(store.get, analysis_id)
            if saved is None:
                raise HTTPException(404, "Analysis not found.")
            from app.product.review import review_saved

            result = await run_in_threadpool(
                review_saved, saved, source_store, saved.get("semantic_assessment")
            )
            if not await run_in_threadpool(
                store.save_product_section, analysis_id, "semantic_assessment", result
            ):
                raise HTTPException(404, "Analysis was deleted during review.")
            return result
        finally:
            analysis_slot.release()
