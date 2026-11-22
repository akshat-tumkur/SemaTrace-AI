# Deployment and development

## Native startup

Use Python 3.12 and Node.js 22.12+ with npm. From the repository root:

```sh
python scripts/start_sematrace.py
```

The interface and API are served at `http://127.0.0.1:8010`. `--port 8020` selects another port; `--rebuild` forces a frontend rebuild. An already-running product server is reused, so stop/start after backend changes.

Setup creates/checks `backend/.venv`, installs wheel-only runtime dependencies when needed, builds the frontend, verifies/downloads the pinned MiniLM encoder, seeds an empty library with 12 licensed Wikipedia revisions and builds a compatible index. No research training dependencies or private RoBERTa weights are needed.

Initial setup needs network access and disk space for the Python/npm environment, encoder and source captures. Later local-only analysis uses those files. To expand the Wikipedia collection to 60 topics, run `python scripts/provision_product_library.py`; reference uploads work directly through Source library.

Stop with `python scripts/stop_sematrace.py`. The Windows `start.ps1`/`stop.ps1` wrappers call the same helpers. The stop command verifies the recorded workspace-specific process identity, including its Windows virtual-environment child. Allow work to finish before stopping; interrupted jobs require re-upload after restart.

Runtime state/logs are private in `.local/runtime/`. The default report database remains `backend/data/beta-demo/analyses.sqlite3` to preserve earlier saved analyses. Source snapshots/indexes are under `backend/data/source-corpus/`; encoder files are under `backend/data/models/`.

## Configuration

Copy `.env.example` to `.env`. With empty provider keys, the application still performs local comparison. Optional AI review uses `OPENAI_API_KEY` and `OPENAI_MODEL`; web discovery uses `TAVILY_API_KEY`. The upload form controls whether excerpts/queries leave the backend for each analysis.

Advanced overrides include `SEMATRACE_DB_PATH`, `SEMATRACE_CORPUS_DB_PATH`, `SEMATRACE_LOCAL_MODEL_DIR`, `SEMATRACE_CORS_ORIGINS` and `SEMATRACE_WRITE_REQUESTS_PER_MINUTE`. Use explicit paths appropriate to the current machine. Native startup adds its own localhost/loopback origins and development port5173.

`SEMATRACE_REVIEW_MODEL_DIR` can point to the separately preserved private review-only RoBERTa export. Absence is ordinary: search and API-backed review work without it. The original classifier remains rejected. The older local-Ollama advisor is optional and disabled by default; it is not required for the cloud-backed investigation button.

Never put actual keys into frontend environment variables or tracked files. [Privacy and licensing](privacy.md).

## Frontend development

Run the native backend, then:

```sh
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Vite's `/api` proxy defaults to `127.0.0.1:8010`. `SEMATRACE_API_PROXY_TARGET` changes it for another backend port. Native startup already serves the production build and does not require Vite.

## Validation

Install development dependencies into the existing environment:

```sh
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
backend/.venv/Scripts/python.exe -m pytest backend/tests -q
```

Use `backend/.venv/bin/python` on macOS/Linux. The active suite covers immutable sources, numerical/index validation, extraction/offset mapping, alignments, overlap-safe scoring, legacy saved readers, API persistence, bounded requests, jobs, provider failures and research newline portability. Private model checks skip when those artifacts are absent.

Frontend checks:

```sh
cd frontend
npm test
npm run build
npm audit --audit-level=high
```

The installed-Edge harness `scripts/browser_smoke.py` accepts `--url`, `--analysis-id` and an output directory under `.local`. Its full journey expects an existing public-fixture AI report with a saved investigation and job record; it then performs actual local and explicitly selected paid-provider uploads using public/CC0 inputs. `--inspect-only` checks a saved AI report on desktop/mobile without submitting another document. This is separate from normal offline regression tests.

Historical experimental scores are documented in [research](../research/README.md), not inferred from test counts. Unit/browser controls demonstrate specific contracts and do not estimate general semantic precision/recall.

## Troubleshooting

- **Port occupied:** stop the previous managed server or choose another port. The launcher does not kill an unrelated service.
- **Backend/proxy error:** use the native one-origin URL and inspect `.local/runtime/server.stderr.log`. An empty/non-JSON response remains an HTTP failure, not a successful empty analysis.
- **Model download interrupted or rejected:** rerun the launcher after checking your connection. Downloads are verified against the pinned sizes and hashes before publication; do not change the lock file to accept different bytes.
- **Source/index unavailable:** inspect the stored source integrity and build a compatible index from `backend` with `.venv/Scripts/python.exe -m app.retrieval.index_cli build --provider local-onnx --batch-size 32` (use `.venv/bin/python` on macOS/Linux). The launcher also checks the index before serving. Do not disable hash checks to get a result.
- **No AI findings:** inspect review status, warnings and reviewed/remaining/skipped counts. Missing configuration, invalid output, an unreviewed unit and a completed negative comparison are different outcomes.
- **Job interrupted:** upload again after restart; completed saved reports remain available.
- **Scanned/complex document:** the implementation reads available text and DOCX paragraphs; it does not provide OCR or complete visual document reconstruction.

## Docker

Docker configuration is supplied, but image build/runtime was not verified on the release machine because Docker was unavailable. Native execution is the validated workflow.

The root multi-stage Dockerfile builds the frontend, installs CPU runtime requirements and runs as UID10001. Compose publishes only `127.0.0.1:8010`, uses runtime `.env`, mounts source/model directories and retains reports in a named volume. The build context excludes credentials, `.local`, datasets, environments and installed dependencies. The private research reranker is not shipped.

First provision native sources/models, stop the native server, and copy `.env.example` to `.env` if necessary. Container overrides must use container paths; avoid carrying Windows model paths into `.env`. On Linux, the source bind mount must be writable by UID10001. Then:

```sh
docker compose build
docker compose up -d
docker compose logs -f sematrace
docker compose down
```

The readiness health check validates the local corpus/index. There is no public-service authentication; keep both native and container ports private.
