from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.parsing.document import parse_document
from app.agents.workflow import run_investigation
from app.retrieval.bm25 import LocalCorpus
from app.retrieval.corpus import add_source_to_corpus, load_local_corpus
from app.storage.database import AnalysisStore
from app.retrieval.tavily import TavilySearch
from app.agents.reasoner import OpenAIReasoner

MAX_FILE_SIZE = 10 * 1024 * 1024

app = FastAPI(
    title="SemaTrace AI",
    description="Evidence-backed semantic similarity analysis API.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

local_corpus = load_local_corpus()
analysis_store = AnalysisStore()
tavily_search = TavilySearch()
openai_reasoner = OpenAIReasoner()


class AnalysisSummary(BaseModel):
    analysis_id: str
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
    mode: str = "agentic deterministic workflow"
    audit_log: list[dict[str, str]] = []
    summary: str = ""
    recommendations: list[str] = []
    message: str | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "sematrace-api",
        "storage": "sqlite",
        "workflow": "langgraph",
        "web_search": "enabled" if tavily_search.enabled else "fallback",
        "reasoning": "openai" if openai_reasoner.enabled else "deterministic-fallback",
    }


@app.post("/api/analyze", response_model=AnalysisSummary, status_code=202)
async def create_analysis(file: UploadFile = File(...)) -> AnalysisSummary:
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
    parsed = parse_document(file.filename or "document.txt", content)

    analysis_id = f"a_{uuid4().hex[:12]}"
    result = run_investigation(parsed.filename, parsed.file_type, parsed.text, local_corpus, tavily_search, openai_reasoner)
    result["analysis_id"] = analysis_id
    analysis_store.save(analysis_id, result)
    return AnalysisSummary(message="Forensic analysis completed", **result)


@app.post("/api/sources", status_code=201)
async def add_source(
    file: UploadFile = File(...),
    title: str = Form(...),
    url: str = Form(...),
) -> dict[str, Any]:
    global local_corpus
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Source exceeds the 10 MB limit.")
    parsed = parse_document(file.filename or "source.txt", content)
    source_id = uuid4().hex[:10]
    source_units = len(parsed.text.split(".")) - 1
    local_corpus = add_source_to_corpus(local_corpus, source_id, title, url, parsed.text)
    return {"source_id": source_id, "title": title, "url": url, "units_indexed": max(source_units, 1)}


@app.get("/api/analysis/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    result = analysis_store.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return result


@app.get("/api/analysis/{analysis_id}/matches")
def get_matches(analysis_id: str) -> list[dict[str, Any]]:
    result = analysis_store.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return result["matches"]


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
    }
