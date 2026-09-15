# SemaTrace AI

**Agentic semantic plagiarism investigation with explainable evidence.**

SemaTrace AI goes beyond exact string matching. It decomposes a document into sentence-level units, searches local and web sources, combines lexical and dense retrieval, verifies meaningful overlap, filters false positives, checks citations, and produces an evidence-backed forensic report.

The system does not claim to mathematically prove plagiarism. It produces a transparent similarity assessment for human review.

For the complete architecture and judge preparation guide, read [docs/JUDGE_GUIDE.md](docs/JUDGE_GUIDE.md).

## What It Does

```text
Upload document
	|
Parse TXT, PDF, or DOCX
	|
Decompose into sentence units
	|
BM25 + dense retrieval + optional Tavily web search
	|
Merge and rank source candidates
	|
Verify meaningful content overlap
	|
Filter generic language and false positives
	|
Check citations
	|
Generate deterministic metrics and optional OpenAI explanation
	|
Persist and display an evidence-backed report
```

## Architecture

LangGraph coordinates the workflow:

```text
Decomposer Agent
	  |
Source Investigator Agent
   /         |          \
 BM25   Dense Retrieval  Tavily
   \         |          /
	 Candidate Merger
		  |
	Evidence Verifier
		  |
	False Positive Filter
		  |
	  Citation Checker
		  |
   Optional OpenAI Reasoning
		  |
	 Forensic Judge
		  |
	SQLite + React Report
```

Every stage appends an audit event, so the UI shows the actual investigation sequence instead of a simulated loading animation.

## Technology

- Python 3.11+
- FastAPI and Pydantic
- LangGraph orchestration
- PyMuPDF and python-docx parsing
- BM25 lexical retrieval
- Configurable dense retrieval boundary with a deterministic local fallback
- Tavily web discovery with graceful fallback
- Optional OpenAI structured report reasoning
- SQLite persistence
- React, TypeScript, Vite, and Lucide icons

## Run Locally

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000` and Swagger documentation is available at `http://localhost:8000/docs`.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

### Tests and build

```powershell
cd backend
pytest -q

cd ..\frontend
npm run build
```

## Environment

Create a local `.env` file. Never commit it.

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4
TAVILY_API_KEY=
EMBEDDING_MODEL=
RERANKER_MODEL=
DATABASE_URL=sqlite:///./data/app.db
TAVILY_SEARCH_ENABLED=true
TAVILY_MAX_UNITS=3
TAVILY_MAX_RESULTS=3
```

`gpt-4` works with the current compatibility path. `gpt-4o-mini` is a faster and cheaper development option.

If OpenAI or Tavily is unavailable, analysis still completes using deterministic local components.

## SQLite

SQLite is embedded in Python. No database server or separate installation is required.

On startup, SemaTrace automatically creates `data/app.db` and stores analysis summaries, sentence units, verified matches, audit events, summaries, and recommendations. The generated database is ignored by Git.

## Adding Sources

### Permanent local corpus

Place UTF-8 `.txt` files in `data/corpus/`. Restart the backend and they will be split into searchable source units.

### Known-source demo

For reliable exact-copy detection:

1. Open **Index a source before checking a copy**.
2. Select a TXT, PDF, or DOCX source file.
3. Enter the source title and original URL.
4. Click **Index source**.
5. Upload the suspected submission and run analysis.

The API equivalent is:

```powershell
curl.exe -X POST http://localhost:8000/api/sources `
  -F "file=@data\corpus\wikipedia-ai.txt" `
  -F "title=Wikipedia - Artificial Intelligence" `
  -F "url=https://en.wikipedia.org/wiki/Artificial_intelligence"
```

Sources added through the UI are indexed for the running backend process. To keep a source after restart, also save it under `data/corpus/`.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Reports storage, workflow, Tavily, and reasoning capabilities |
| `POST` | `/api/sources` | Indexes a known source file with title and URL |
| `POST` | `/api/analyze` | Analyzes an uploaded document |
| `GET` | `/api/analysis/{id}` | Reads a persisted analysis |
| `GET` | `/api/analysis/{id}/matches` | Reads verified matches |
| `GET` | `/api/analysis/{id}/report` | Reads the forensic report |

## Understanding the Output

```text
Candidate count = possible sources discovered
Verified matches = candidates that passed evidence verification
Strong matches = high meaningful-token overlap
Possible paraphrases = meaningful overlap with changed wording
Affected text = verified matching tokens / analyzable tokens
Risk score = coverage and verification strength combined
```

A retrieval score is not a plagiarism probability. Only verified matches appear in **Verified evidence matches**.

## Judge Demo

1. Index a known Wikipedia or sample source.
2. Upload a submission containing an exact copied passage, a paraphrase, and a common phrase.
3. Show the Decomposer, BM25, dense retrieval, Tavily, Verifier, Filter, Citation Checker, OpenAI, and Judge events.
4. Show candidate count versus verified match count.
5. Open an exact match and point to the source URL.
6. Open a paraphrase and explain that wording changed while the claim remained similar.
7. Show that generic language was filtered.
8. Show citation status, recommendations, and the downloadable report.

### Judge explanation

> We separate discovery from verification. The Investigator finds possible sources, the Verifier checks whether the source supports the suspected overlap, the filter removes generic language, the Citation Checker adds context, and the Judge aggregates the evidence. OpenAI explains retrieved evidence but does not invent sources or calculate numeric measurements.

## Important Limitations

- The default dense fallback is deterministic and local; production can replace it with a trained embedding provider.
- Tavily is discovery support and may not return exact page text.
- UI-indexed sources are process-local unless also saved under `data/corpus/`.
- Citation detection is pattern-based.
- Thresholds are engineering defaults, not legal or scientifically validated boundaries.
- Final academic or legal determination requires human review.

## Security

- Keep API keys in `.env` or deployment secret storage.
- Never commit `.env` or send keys to the React client.
- Treat uploaded files and web content as untrusted data.
- Limit file size and reject malformed/unsupported files.
- Keep provider failures from crashing deterministic analysis.

## Repository Guide

- [docs/JUDGE_GUIDE.md](docs/JUDGE_GUIDE.md): full architecture, workflow, formulas, troubleshooting, and judge preparation
- [implementation_plan.md](implementation_plan.md): original hackathon specification
- [SCRATCHPAD.md](SCRATCHPAD.md): shared handoff notes for Copilot, Codex, Claude, and teammates
