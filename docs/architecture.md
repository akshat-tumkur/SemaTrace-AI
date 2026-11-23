# Architecture

SemaTrace separates source discovery, matching-word measurement and advisory meaning review. Fixed Python orchestration handles uploads; optional LangGraph investigation performs conditional tool calls.

## Components

| Component | Responsibility |
| --- | --- |
| React/TypeScript | Upload choices, progress, report views, exclusions and exports. |
| FastAPI/Uvicorn | One local origin, or an authenticated API for a separate frontend. |
| SourceCorpusStore | Source captures, immutable versions, metadata, occurrences and snapshots. |
| GenerationStore | Verified embedding spaces and corpus-aligned vector generations. |
| HybridRetriever | Batched local MiniLM/BM25 search, document aggregation and RRF. |
| Aligners/scorer | Seeded local alignment, exact runs and unique word accounting. |
| Product review | Bounded OpenAI comparisons and literal quotation validation. |
| Investigation graph | Conditional context, alternatives or experimental reranking. |
| AnalysisStore/JobStore | Reports, saved edits, progress and restart recovery. |
| Cloud adapters | Supabase Auth/PostgreSQL/private Storage and Pinecone vector retrieval, shared global references plus isolated private account data. |
| ReportLab | PDF generation from saved evidence. |

## Reference preparation

Ingestion extracts authorized text and retains its capture, fingerprint, attribution and revision. Sentence/text-unit passages retain original positions. Identical passage text shares a representation with separate source-occurrence records.

Pinned MiniLM produces 384-dimensional, mean-pooled, normalized vectors locally. Generation publication verifies the collection identity, passage ordering, model specification and vector fingerprints. Compatible vectors can be reused as references change.

## Upload flow

1. `POST /api/analysis-jobs` checks size/the shared heavy-work slot and returns HTTP202 with a durable job ID. The browser polls while a worker thread performs blocking processing.
2. Extraction retains original Unicode and normalized position maps. Segmentation handles URLs, abbreviations and some citation markers. Bounds fail explicitly instead of silently dropping the ending.
3. Optional Tavily discovery makes up to 2 queries, examines at most 6 URLs and captures at most 3 sources, with at most 3 provider-extraction fallbacks. Direct/provider captures have separate provenance and are indexed before the analysis snapshot.
4. Every accepted unit receives BM25/local MiniLM retrieval. Local query similarities stream in batches of 32; ONNX inference splits those into smaller batches of 8 by default. The neural representation has a256-token per-unit limit. Cloud mode computes BM25 in PostgreSQL and retrieves a bounded Pinecone passage pool, retaining the same local encoder and RRF formula.
5. Maximum query/passage scores aggregate to source documents. RRF adds `1/(60+rank)` contributions; up to 10 full versions enter lexical alignment. Retained/dropped IDs persist.
6. Independently, up to 8 semantic passage-occurrence candidates per unit retain literal text, provenance, positions and ranking signals. No cosine threshold declares reuse.
7. Lexical alignment measures wording; optional AI review compares selected meanings. Their results remain separate.
8. The saved report is written before the job becomes complete. The browser changes its job URL to the saved analysis URL.

Browser refresh reconnects to the existing job. A process restart cannot resume its in-memory upload/thread: unfinished jobs become interrupted failures. Completed reports persist.

Cloud mode stores original uploads privately before acknowledging the job. Report insertion and job completion commit in the same PostgreSQL transaction. Interrupted cloud uploads remain downloadable by their owner for resubmission; work is not silently replayed against paid providers.

## Wording and coverage

The main aligner uses five-word seeds,128-word tiles,96-word stride and padded source windows. Biopython local-alignment scores are2/−1/−1 for match/mismatch/gaps. Evidence requires 12 matched words,80 characters on both sides and density≥0.5. Brief exact runs cover qualifying sequences of at least 5 words with3 distinct non-stopword types.

The scorer validates source fingerprints, word boundaries and equal case-folded keys. An envelope may contain edits; only its independently checked matching runs count.

```text
coverage = 100 × unique eligible matched submitted word positions
                 / eligible submitted word positions
```

Source overlap counts once; different submitted repetitions remain distinct positions. Primary contribution uses stable source-version ordering, an accounting decision rather than authorship attribution. Source exclusions affect the numerator; explicit document ranges affect numerator and denominator. Recalculation reads stored evidence without retrieval, alignment or model calls.

## Meaning review

`product/review.py` selects up to 48 eligible, not-yet-reviewed units by their strongest saved retrieval signal. Eligibility requires 8 word-like tokens and at most 1,800 characters; explicit attribution/license/reference and URL-containing units are excluded. Up to 3 distinct source contexts per unit are compared in batches of 6, with at most 8 calls and explicit input bounds.

Structured relations are `PARAPHRASE`, `PARTIAL_CORRESPONDENCE`, `RELATED_ONLY`, `DIFFERENT_MEANING` and `INSUFFICIENT_CONTEXT`. Positives require valid unit/candidate aliases and exact, unique, bounded quotations of at least 4 words on both sides. Accepted spans project back to original extracted positions.

Grounding establishes quote existence/location, not semantic truth or copying history. Findings remain advisory and cannot change word coverage. Reviewed/remaining/skipped scope, warnings, actual model, usage and run history persist.

Optional provider failure can coexist with saved local results. Broken core evidence cannot become a successful partial analysis.

## Investigation

The graph cycles `decide -> tool -> decide` and then explains the evidence. Tools read immutable neighboring context, retrieve alternatives with the historical generation or request an optional RoBERTa ranking. Tools cannot repeat; the cloud route permits 2 tool steps/3 model calls.

Explanations undergo separate quote/identity checks and remain unverified interpretation. Completed results/traces are cached per candidate/provider. There is no persistent internal-node graph checkpointer for resuming a crashed request.

## Persistence and presentation

The default report database is `backend/data/beta-demo/analyses.sqlite3`; its historical directory name preserves existing reports. A sibling jobs database stores progress. Source metadata/blobs and retrieval SQLite live under `backend/data/source-corpus/`. Models live under `backend/data/models/`; ignored `.local/` holds runtime state and optional private artifacts.

With `SEMATRACE_STORAGE_PROVIDER=supabase`, factories instead return owner-scoped cloud stores. Supabase verifies bearer sessions; PostgreSQL retains passages, term frequencies, immutable source identities, snapshots, jobs, reports, exclusions and daily budgets. Content-addressed raw/extracted files live in private Storage. A single global library (UUID ending `0001`, not an Auth user) stores shared references; private library UUIDs belong to their users. Pinecone stores 384-dimensional cosine vectors in `global-v1` and isolated `user-UUID` namespaces. Analysis freezes the eligible union, deduplicates canonical versions, computes BM25 statistics over eligible distinct passages, merges comparable namespace cosine results before limits, then applies RRF once. Global writes require membership in protected `st_admins`, checked by the backend on each request; user-editable metadata confers no role. Global source changes do not depend on the uploading administrator remaining an active account. Frozen source-version filters, publication visibility checks and grounded passage lookups prevent stale/unpublished vectors from being treated as report evidence. Removal deactivates sources first and retains a vector-cleanup outbox for provider failures. BM25 term data stays in PostgreSQL; source-text caches and remote query results are bounded. BM25 corpus statistics materialize once per query and the term lookup has a covering index. Provider upserts hold at most 64 vectors while encoder microbatches remain eight. The report-size guard counts JSON incrementally before storage, retaining the same 24 MiB bound without allocating full duplicate payloads.

Cloud mode has no ephemeral SQLite fallback. The Render container contains only code, fonts and the pinned public encoder; replacing it does not own the lifetime of persistent cloud records. One instance/worker is required for the in-process heavy-work semaphore. Authenticated cloud report/export reads share that semaphore to bound transient memory. PostgreSQL daily budgets survive replacement; short per-user request-rate windows are process-local. [Setup and free-tier limitations](deployment.md#free-cloud-hosting).

Reports retain their original source/index scope, matching runs, candidates, original maps and explicit edits. Internal full alignment-source snapshots support recalculation but are omitted from public API payloads. Export reads saved values without rescoring. Deletion removes report children/job pointers while references and existing exports/backups remain separate.

`app/compat/` performs read-time safety projection for earlier saved evidence. Abandoned V2 graph/provider orchestration is archived privately and is not an alternative live pipeline.

## Code and API entry points

Start at `scripts/start_sematrace.py`, `backend/product_server.py`, `app/product/routes.py` and `frontend/src/App.tsx`. Core algorithms live in `app/retrieval/`, `app/alignment/`, `app/scoring.py` and `app/product/review.py`; investigation is in `app/discovery/investigation.py`.

FastAPI exposes `/openapi.json` and `/docs`. Main operations include `/api/analysis-jobs`, `/api/analysis/{id}` and its `/similarity`, `/ai-review`, `/cloud-investigate`, `/report` and `/report.pdf` routes, plus `/api/sources`, `/api/corpus` and `/api/analyses`.
