# SemaTrace AI Scratchpad

Shared working context for Copilot, Codex, Claude, and future contributors.

## Current phase

Phase 1 MVP complete: parser, lexical+dense retrieval, Tavily investigation, LangGraph agent workflow, OpenAI reasoning boundary, and SQLite storage.

## Implemented

- FastAPI backend with `POST /api/analyze` and `GET /api/analysis/{analysis_id}`.
- TXT upload parsing and a deterministic baseline analysis.
- React/Vite frontend with upload flow, progress state, overview metrics, and activity timeline.
- Environment template and initial project documentation.
- PDF, DOCX, and TXT parsing with normalized text output.
- Sentence units with stable character offsets.
- Dependency-free BM25-style local corpus retrieval with source metadata.
- Evidence candidate cards in the frontend.
- Real synchronous workflow with distinct Decomposer, Investigator, Verifier, False Positive Filter, Citation Checker, and Forensic Judge agents.
- API exposes workflow mode, risk score, and ordered audit events.
- Frontend renders the returned audit trail instead of hardcoded activity rows.
- SQLite persists analyses, units, matches, and audit events across process restarts.
- Dedicated persisted matches endpoint is available.
- BM25 and dense retrieval candidates are merged by unit/source with a transparent retrieval score.
- Dense retrieval defaults to a deterministic local fallback; the provider boundary is ready for sentence-transformers.
- Tavily web investigation loads the root `.env`, is bounded by `TAVILY_MAX_UNITS` and `TAVILY_MAX_RESULTS`, and degrades to local retrieval on provider failure.
- Real configured-key smoke test recorded `searched_web`; no usable web passages were returned for the small demo document, while local candidates remained available.
- LangGraph now owns the decomposer -> investigator -> verifier -> filter -> citation -> judge graph.
- OpenAI reasoning is optional, structured JSON, bounded to retrieved evidence, and falls back safely.
- Forensic summary and recommendations are persisted and displayed in the UI.
- Verification now removes stopword-driven and weak unrelated overlaps before evidence is shown as confirmed.
- UI explicitly labels confirmed cards as `Verified evidence matches` and shows candidate-to-verified counts.
- Tavily now receives shortened and quoted query variants from information-rich units.
- Added `POST /api/sources` to index a known source document with title and URL before analysis.
- Fixed `.env` loading root path: `backend/app/config.py` now resolves the workspace root correctly.
- OpenAI `gpt-4` compatibility fixed by sending `response_format` only to supported model families.
- Live smoke test now records `synthesized_report` and returns an AI-generated summary.
- Capability-aware `/api/health` reports SQLite, LangGraph, Tavily fallback, and reasoning mode.
- JSON forensic report endpoint and frontend download action are available.
- Controlled evaluation fixtures cover exact-copy and common-language false-positive behavior.

## Deliberate MVP boundaries

- No major architecture milestones remain for the planned MVP.
- The current analyzer is explicitly deterministic baseline mode; it must not be presented as a complete plagiarism detector.
- No API secrets are committed. OpenAI/Tavily integration remains provider-configurable work.

## Next recommended slice

1. Expand the labeled evaluation set and measure retrieval/classification metrics.
2. Add production web-page extraction and richer source metadata.
3. Add live progress streaming for a future hosted deployment.

## Decisions

- Backend: Python 3.11+, FastAPI, Pydantic.
- Frontend: React + TypeScript + Vite.
- Keep deterministic measurements separate from future LLM reasoning.
- Keep this file updated after each meaningful implementation slice.

## Validation run

- `python -m compileall app` passed.
- `npm run build` passed after adding Vite and React type declarations.
- `POST /api/analyze` smoke test passed with `data/demo/sample.txt`.
- Verified response: completed, `MODERATE`, 25.0% baseline coverage, 2 possible paraphrases, 1 filtered common phrase.
- `pytest -q` passed: 3 parser tests.
- `python -m compileall app` passed after parser integration.
- `pytest -q` passed: 5 parser and retrieval tests.
- Analysis endpoint smoke test returned 3 local lexical candidates.
- `npm run build` passed after adding the evidence candidate view.
- `pytest -q` passed: 7 parser, retrieval, and workflow tests.
- Workflow API smoke test returned 7 ordered audit events and evidence-based verdict fields.
- `npm run build` passed with the live workflow timeline.
- `pytest -q` passed: 8 parser, retrieval, workflow, and storage tests.
- SQLite API smoke test passed: analysis persisted and matches endpoint returned 2 evidence records.
- `pytest -q` passed: 10 tests with dense retrieval and candidate merger coverage.
- Dense workflow API smoke test returned 9 ordered audit events.
- `npm run build` passed with merged retrieval scores in evidence cards.
- `pytest -q` passed: 11 tests with Tavily fallback coverage.
- Live Tavily smoke test completed without exposing credentials or failing the local workflow.
- `pytest -q` passed: 11 tests with LangGraph workflow and fallback coverage.
- Deterministic API smoke test returned 11 ordered LangGraph/OpenAI-boundary events.
- `npm run build` passed with the forensic report panel.
- `pytest -q` passed: 12 tests including controlled evaluation fixtures.
- Health endpoint confirmed SQLite storage and LangGraph workflow capabilities.
- Report endpoint returned persisted JSON report fields.
- `npm run build` passed with report download control.
- OpenAI live smoke test passed with the configured `gpt-4` model after request compatibility fix.
- Evidence-quality regression suite passed: unrelated retrieval candidates are not confirmed.
- Demo output reduced 10 retrieved candidates to 4 verified matches (2 strong, 2 possible paraphrases).
- Exact copied-source regression passes when a Wikipedia-like source is indexed directly.
