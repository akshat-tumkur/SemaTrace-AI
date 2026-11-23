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

The Docker image builds and runs locally. It includes the same pinned FP32 MiniLM encoder, downloaded and hash-verified at build time. No GPU, PyTorch, private research weights or training dataset is included.

The root multi-stage Dockerfile builds the frontend, installs CPU runtime requirements and runs as UID10001. Compose publishes only `127.0.0.1:8010`, uses runtime `.env` and retains both corpus and reports in named Linux volumes. The build context excludes credentials, `.local`, datasets, environments and installed dependencies. The private research reranker is not shipped.

First provision the native source library, stop the native server, and copy `.env.example` to `.env` if necessary. Container overrides must use container paths; avoid carrying Windows model paths into `.env`. Then:

```sh
docker compose build
docker compose up -d
docker compose logs -f sematrace
docker compose down
```

On first startup, the short-lived `corpus-init` service copies the existing native corpus into a Linux volume and assigns it to UID10001. It publishes the copy by atomic directory rename and never overwrites an initialized corpus. The API always runs as non-root. The original `backend/data/source-corpus` stays unchanged as a native library/backup; subsequent Docker reference uploads live in the named volume, and the two libraries do not automatically synchronize. Docker uses the identical pinned encoder baked into the image. This avoids slow Windows/WSL filesystem round trips for SQLite and model reads. Keep both named volumes; `docker compose down -v` deletes persistent Docker data.

The readiness health check validates the local corpus/index. Local SQLite mode has no sign-in and must stay private. Public hosting requires the authenticated cloud mode below.

### Measured memory and regression checks

Local Linux Docker validation used a strict **512 MiB memory limit with no additional swap**, the existing 13,643-passage library plus small CC0 controls, and the original pinned **FP32** MiniLM model. No quantization or model-quality tradeoff was required.

| Earlier reference run stage | Cumulative container memory peak |
| --- | ---: |
| Ready after startup | 185.3 MiB |
| Incremental reference ingestion and embedding | 300.6 MiB |
| Paraphrased submission | 317.3 MiB |
| 1,000-unit accepted submission and concurrent health reads | 323.9 MiB |
| PDF/DOCX, PDF/JSON exports, exclusions and AI review | **371.5 MiB** |

These are Linux cgroup `memory.peak` readings, not estimates. In that run the process RSS high-water mark was 419.0 MiB; RSS and cgroup accounting differ for file-backed/shared pages. A later repeat with the batched loader and the accumulated local SQLite test state reached the **512 MiB cgroup cap**, with a 454.3 MiB RSS high-water mark and no OOM kill. The kernel reclaimed file-cache pages at the limit. Therefore the below-400 MiB target is **not consistently demonstrated** across local runs. Both runs and an earlier archive-transfer run are retained privately; the transfer archive was removed before the functional repeats. Do not interpret the smallest measurement as a guaranteed deployment peak.

Profiling identified decoded Python-float vector retention and incremental index construction as major costs. The implementation streams verified vectors/index writes, batches SQLite vector reads to avoid per-passage Windows filesystem round trips, bounds BM25/text caches and inference batches, verifies model hashes in chunks, lazily imports document parsers and returns unused Linux allocator memory after jobs. Fixed local ranking/coverage controls matched the previous implementation; model weights, alignment evidence rules and scoring were preserved. The shared-corpus backend run passed 313 tests with one unavailable private-artifact check skipped; all 33 frontend tests passed. Cloud mode keeps vectors and BM25 data outside the backend. Render itself still requires a hosted validation; local Docker measurements do not certify its CPU or cold-start behavior.

The cloud persistence journey used real PostgreSQL and the real MiniLM encoder, with durable Supabase/Pinecone protocol doubles. A replacement backend recovered its source/version, original upload, report, job, exclusion and PDF export, while another account could not read them. Those earlier protocol-double checks alone did not certify live providers. The later shared-corpus live checks below cover Supabase/Pinecone; Render CPU performance and Vercel deployment still require hosted validation. The small AI check used one provider call on the CC0 fixture and returned four grounded advisory findings; it does not estimate semantic accuracy.

## Free cloud hosting

The supported configuration is a Vercel frontend, one Render Docker backend, Supabase PostgreSQL/private Storage/Auth, and Pinecone vectors. MiniLM still generates embeddings locally on CPU. Local SQLite/NumPy operation remains available without these services. Never expose local mode as a public service.

Free hosting has real limits: Render sleeps after 15 minutes idle, has an ephemeral filesystem and a 512 MB free-instance memory limit. Supabase Free includes 500 MB database space, 1 GB Storage and 5 GB egress, and may pause inactive projects after seven days. Pinecone Starter includes 2 GB storage, 2 million write units and 1 million read units per month, with AWS us-east-1 indexes. Vercel Hobby has request and transfer quotas. These are small-demo quotas, not an always-on or unlimited service promise. Exceeding a free quota can make the application unavailable. See [Render Free](https://render.com/docs/free), [Supabase billing](https://supabase.com/docs/guides/platform/billing-on-supabase), [Pinecone pricing](https://www.pinecone.io/pricing/) and [Vercel Hobby](https://vercel.com/docs/plans/hobby).

Choose the **Free/Starter/Hobby** plans throughout. Do not activate a paid trial, upgrade, persistent Render disk or Render database. OpenAI usage is separate and optional; Tavily also has its own quota. The backend does not create paid resources or change plans. No keep-alive service is required or recommended.

### Shared-corpus live verification

The shared-library journey was tested against real Supabase Auth/PostgreSQL/Storage and Pinecone, using one disposable administrator and two new ordinary accounts. All three could access the shared corpus; ordinary users could not mutate it, even after editing their user metadata or calling the database REST API directly. Private sources, reports, jobs and original uploads remained isolated, including from the administrator. Updating/deactivating a global reference retained historical report evidence. Completed records and exclusions survived replacement by fresh containers without local data mounts.

The live corpus contains 60 licensed Wikipedia source versions, 13,342 indexed passage occurrences and 13,208 distinct passage texts. Final verification retired 45 redundant private Wikipedia search copies after checking 10,370 reused vectors. All original private versions, text, provenance and occurrence positions matched the pre-migration evidence; original files and historical records remain. Disposable QA accounts and their live test data were removed after their verification records were archived privately. A deterministic comparison on the same collection retained the corresponding source in the top ten for 16/16 queries in both local and cloud retrieval. Seven top-ten rankings were identical; the average overlap was nine sources. The bounded approximate Pinecone search can change other rankings. A 20-passage exact Wikipedia control received 100% matched-word coverage with literal source positions and licensing checked. These are functional/retrieval controls, not an independent semantic-accuracy evaluation.

Actual cloud containers were capped at 512 MiB, with no additional swap and one CPU. Tests included near-49,000-character reference ingestion, global/private analysis, exports, persisted exclusions and a repetitive stress case that correctly exceeded the existing 24 MiB report-size limit. The initial eager size check peaked at 508.7 MiB; replacing its full JSON string/byte copies with an incremental size check reduced the fresh-container stress peak to **325.2 MiB**, with no OOM kill. The encoder remains FP32 with inference microbatches of eight; provider upserts batch at most 64 vectors. The score formula, aligner and evidence rules are unchanged. The maximum 1,000-unit case was checked locally in the earlier validation, not against live cloud providers; these measurements do not guarantee every input's peak or Render's actual CPU/cold-start behavior.

Cloud BM25 materializes eligible-corpus statistics once per query and uses a term lookup index. Live query-plan checks caught repeated aggregation in the earlier plan; materialization preserves the formula while removing that repeated work. Query timeouts are explicitly set inside each database transaction because the transaction pooler may ignore startup options.

### 1. Prepare private configuration

From the repository root in PowerShell:

```powershell
if (!(Test-Path .env.cloud)) { Copy-Item configs/cloud.env.example .env.cloud }
notepad .env.cloud
backend\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r backend/requirements-cloud.txt
```

Keep `.env.cloud` private; it is ignored by Git and Docker. Do not replace the existing local `.env`. Set actual values where the template contains placeholders. Unix users can use the corresponding virtual-environment Python path.

### 2. Create Supabase Free

1. Create a Free project in your own Supabase account and retain its database password privately.
2. In the project's connection dialog, copy the **Transaction pooler** PostgreSQL URI (port 6543). This supports IPv4 on Free. Put it in `SUPABASE_DB_URL`; URL-encode special characters in the password. The client requires SSL and disables prepared statements. Do not paste the URI into chat.
3. Copy the project URL into `SUPABASE_URL`, the **publishable** key into `SUPABASE_PUBLISHABLE_KEY`, and the **secret** key into `SUPABASE_SECRET_KEY`. The backend also accepts older `SUPABASE_ANON_KEY`/`SUPABASE_SERVICE_ROLE_KEY` installations, but new projects should use the modern keys. Secret/database keys stay server-side. [Supabase key types](https://supabase.com/docs/guides/getting-started/api-keys), [connection options](https://supabase.com/docs/guides/database/connecting-to-postgres).
4. Keep anonymous sign-in disabled. For a closed email/password demo, disable public new-user signup and add users from **Authentication → Users**, setting a strong password and confirming the email. To admit new Google users, enable new-user signup and follow the Google setup below. Retain your initial user's UUID for library migration. Email/password sign-in remains available; password resets are operator-managed.
5. Apply the schema and private bucket:

```powershell
backend\.venv\Scripts\python.exe scripts/cloud_admin.py migrate
```

Alternatively run `backend/cloud/schema.sql`, `backend/cloud/storage.sql`, then `backend/cloud/shared-corpus.sql` in that order in Supabase SQL Editor. The additive shared-corpus migration preserves existing private data. Browser database writes are denied. Authenticated readers can access the global corpus and only their own private records. The bucket stays private, with matching read policies and backend-only uploads. No local filesystem or SQLite copy is used as fallback in cloud mode.

### 3. Create Pinecone Starter with your own vectors

Create a **Starter** project, copy its API key into `.env.cloud`, then run:

```powershell
backend\.venv\Scripts\python.exe scripts/cloud_admin.py create-index --confirm-starter
```

The helper explicitly creates or validates `sematrace-minilm`: **dense, 384 dimensions, cosine, AWS us-east-1, no integrated embedding model**. It saves the returned index host in `.env.cloud`. Confirm Starter in the dashboard before passing the flag; the helper does not upgrade a plan.

The helper and runtime adapter pin Pinecone API `2026-04`, matching the existing Vectors API request/response format. `2026-07` changes index creation to the Documents API schema; mixing that version with the older vector-index payload causes HTTP 422. Provider failures include the HTTP status and a credential-redacted message. [Versioned vector-index API](https://docs.pinecone.io/reference/api/2026-04/control-plane/create_index).

MiniLM does not need to appear in Pinecone's model gallery. SemaTrace generates its own vectors and writes them with ordinary API **upserts**, which use Starter's included write quota. The console's paid **Import** feature imports large datasets from object storage; SemaTrace does not use it. [Vector upsert documentation](https://sdk.pinecone.io/python/how-to/vectors/upsert-and-query.html), [bulk import documentation](https://sdk.pinecone.io/python/how-to/vectors/bulk-import.html).

Passage occurrences have stable IDs and source-version metadata. The global corpus uses one `global-v1` namespace; private references use isolated `user-USER_UUID` namespaces. Publication checks observe uploaded vectors before activating a source version. Removing a source immediately removes it from future source snapshots; failed remote cleanup is retained for retry. Existing reports retain their immutable evidence.

### 4. Publish the shared corpus and assign an administrator

On the laptop that contains the existing reference library:

```powershell
backend\.venv\Scripts\python.exe scripts/cloud_admin.py grant-admin --user YOUR_USER_UUID
backend\.venv\Scripts\python.exe scripts/cloud_admin.py migration-status
backend\.venv\Scripts\python.exe scripts/cloud_admin.py import-local --owner YOUR_USER_UUID --scope global
backend\.venv\Scripts\python.exe scripts/cloud_admin.py check
```

The default imports only the licensed `wikipedia_en` collection into the shared global library, preserving source/version identities, fingerprints, positions and provenance. The actor supplied with `--owner` must be an administrator. Complete versions already in that actor's private library reuse verified vectors; only missing vectors require the unchanged pinned MiniLM encoder. Repeating the command resumes incomplete indexing and skips newly embedded versions already published. Run it on the laptop, not in an interactive Render shell. Readiness requires an active global corpus, a verified global vector and at least one administrator.

Administrator membership lives in protected `st_admins`, not user metadata or a frontend flag. Only the trusted CLI/database operator can grant or revoke it (`revoke-admin --user UUID`). The interface shows Global Source Library, My Private Sources and My Reports. Global uploads, updates and removals require backend-verified administrator membership. Updating keeps immutable previous versions for saved reports. New users immediately search the shared corpus; they need no private references.

If an older import populated a private Wikipedia library, finish the global import above, then promote/verify and retire only those redundant private search copies:

```powershell
backend\.venv\Scripts\python.exe scripts/cloud_admin.py promote-wikipedia --from-owner ORIGINAL_OWNER_UUID --admin YOUR_USER_UUID --retire-private
backend\.venv\Scripts\python.exe scripts/cloud_admin.py migration-status
```

Promotion verifies every copied vector's 384-dimensional float32 fingerprint and source metadata before global publication or private retirement. It records a protected verification ledger, deactivates the exact transferred private version, and removes only that version's redundant private vectors. Original raw/text evidence, private source/version rows, reports and unrelated private references remain intact. Failed cleanup is resumable by repeating the command. Running without `--retire-private` verifies publication and retains both search copies; retrieval deduplicates identical canonical versions. Historical snapshots can resolve an identical verified global version after retirement.

If Docker is your current working library, export its live corpus first, without overwriting the native library:

```powershell
New-Item -ItemType Directory -Force .local/cloud-corpus-export | Out-Null
docker compose cp sematrace:/app/backend/data/source-corpus/data/. .local/cloud-corpus-export
backend\.venv\Scripts\python.exe scripts/cloud_admin.py import-local --owner YOUR_USER_UUID --database .local/cloud-corpus-export/corpus.sqlite3
```

Allow analyses/reference uploads to finish before exporting. Use the same `--database` option with `compare-retrieval` for that exported corpus. Docker reports/jobs can similarly be exported from `/app/backend/data/reports/` into a separate private directory and passed to `import-reports --database`. Keep exports private.

To include authorized personal references, explicitly choose `--collection all --include-private --scope private`. Identical versions already active globally are skipped. Other private references remain owner-isolated and are combined with the global corpus during analysis. Do not copy the shared corpus for each user. The default budget is 20,000 distinct passage fingerprints across libraries/retained versions, plus a 450 MiB physical PostgreSQL safety budget; duplicate promotion evidence still consumes database space and must be monitored.

Existing local reports and exclusions can also be copied explicitly:

```powershell
backend\.venv\Scripts\python.exe scripts/cloud_admin.py import-reports --owner YOUR_USER_UUID --database backend/data/beta-demo/analyses.sqlite3 --include-private
```

This reads the local database without changing it, preserves report IDs and job records, and refuses conflicting cloud records rather than overwriting them. Local jobs did not retain original upload bytes; newly created cloud jobs do. Import the corresponding references as well if you want new retrieval/investigation against the same sources.

Before accepting cloud retrieval, import the same licensed collection and run the bounded comparison:

```powershell
backend\.venv\Scripts\python.exe scripts/cloud_admin.py compare-retrieval --owner YOUR_USER_UUID --output .local/cloud-retrieval-comparison.json
```

This compares document rankings for up to 16 predetermined indexed passages on identical source-version collections. The default compares only `wikipedia_en`; use `--collection all` only when the full eligible libraries match. Verified vectors are selected from the existing local generation without re-embedding or changing it. It is an engineering smoke check, not an independent semantic-accuracy evaluation. Review missing source hits and ranking differences before deployment. Cloud Pinecone search is bounded to 256 passage hits per query; BM25 uses the same scoring formula in PostgreSQL. Approximate retrieval can differ from the local exhaustive scan. The source-hit comparison must pass on the live project; top-ten differences should still be reviewed. Do not claim identical cloud quality from unit-test counts.

### 5. Deploy Render Free

Connect the repository in Render and create a Blueprint using `render.yaml`, or create a **Free Docker Web Service** manually from the repository root. The blueprint declares only one free instance; automatic deployment is off. Do not create a paid preview or database.

Deploy from a reviewed Git branch containing these implementation/configuration changes. Local uncommitted files are not available to Render or Vercel; commit and push the application changes first, keeping `.env.cloud`, runtime data, exports and private artifacts excluded.

Copy the private server values from `.env.cloud` into Render's environment controls. Set `SEMATRACE_STORAGE_PROVIDER=supabase`, `SEMATRACE_VECTOR_PROVIDER=pinecone` and `SEMATRACE_SERVE_FRONTEND=false`. Supply the actual Vercel frontend origin in `SEMATRACE_CORS_ORIGINS` and the Render hostname, plus `localhost,127.0.0.1`, in `SEMATRACE_ALLOWED_HOSTS`. Wildcards are rejected in cloud mode. Origins include `https://`; allowed hosts do not. When setting up the two services for the first time, create their URLs, then update these values and redeploy. Keep preview deployments away from production documents unless their exact origins are deliberately authorized.

Use branch `main`, repository root `.`, Dockerfile `./Dockerfile`, one Free instance, and health path `/api/ready`. Leave custom build/start commands empty so the Dockerfile controls them. The Blueprint supplies the non-secret defaults. In a manual Web Service, copy the corresponding defaults from `configs/cloud.env.example` and `render.yaml`.

| Environment group | Variable names | Configuration |
| --- | --- | --- |
| Supabase connection | `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SECRET_KEY`, `SUPABASE_DB_URL` | Copy the existing project's values privately from `.env.cloud`; retain the IPv4 transaction-pooler URI. |
| Pinecone connection | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_INDEX_HOST` | Reuse the existing populated index; do not create another index or reimport the corpus. |
| Application routing | `SEMATRACE_STORAGE_PROVIDER`, `SEMATRACE_VECTOR_PROVIDER`, `SEMATRACE_SERVE_FRONTEND`, `SEMATRACE_CORS_ORIGINS`, `SEMATRACE_ALLOWED_HOSTS` | Use the cloud provider defaults and exact final service origins/hosts described above. |
| Storage/retrieval bounds | `SUPABASE_STORAGE_BUCKET`, `SEMATRACE_PINECONE_TOP_K`, `SEMATRACE_CLOUD_MAX_PASSAGES`, `SEMATRACE_INFERENCE_BATCH_SIZE` | Defaults are in the example; keep the bucket provisioned by the SQL migration. |
| Upload/request controls | `SEMATRACE_SOURCE_UPLOAD_ENABLED`, `SEMATRACE_WRITE_REQUESTS_PER_MINUTE` | Retain the example/Blueprint defaults. |
| Daily application limits | `SEMATRACE_ANALYSIS_USER_DAILY_LIMIT`, `SEMATRACE_ANALYSIS_GLOBAL_DAILY_LIMIT`, `SEMATRACE_REFERENCE_USER_DAILY_LIMIT`, `SEMATRACE_REFERENCE_GLOBAL_DAILY_LIMIT` | Retain the example defaults. |
| Optional external services | `OPENAI_API_KEY`, `OPENAI_MODEL`, `TAVILY_API_KEY`, `SEMATRACE_AI_USER_DAILY_LIMIT`, `SEMATRACE_AI_GLOBAL_DAILY_LIMIT`, `SEMATRACE_WEB_USER_DAILY_LIMIT`, `SEMATRACE_WEB_GLOBAL_DAILY_LIMIT` | Keep keys absent and budgets disabled initially. Both user/global budgets and a server key are required to enable a provider. |

Render assigns `PORT`; no laptop model path, SQLite path, administrator environment variable, persistent disk or local bind mount is required. Administrator authorization is already stored in the protected database table. Keep API/database credentials exclusively in Render's server environment.

The image binds `0.0.0.0` using Render's assigned `PORT`, runs as UID10001, and includes the hash-verified encoder. Use `/api/ready` for health checks. Cloud reports, jobs, reference text and original submissions are stored outside the container. Restarted in-flight jobs become explicitly interrupted; completed reports remain available. There is no automatic retry of a paid operation.

### 6. Deploy Vercel Hobby

Import branch `main` of the repository, set **Root Directory** to `frontend`, framework Vite, Node.js `22.x`, build command `npm run build`, output `dist`, and these public build variables:

```dotenv
VITE_API_URL=https://YOUR_SERVICE.onrender.com
VITE_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=YOUR_PUBLIC_PUBLISHABLE_KEY
```

Add the final Vercel URL to Supabase's Auth URL settings and Render's exact CORS origins. Redeploy after changing Vite variables because they are compiled into the browser bundle. Do not set database passwords, secret/service-role keys, Pinecone, OpenAI or Tavily keys as `VITE_*` variables. The build rejects recognizable secret-key values in public Supabase key configuration.

Sign in with the user created in Supabase. Session refresh is handled by the Supabase client; bearer credentials are sent only to the configured API origin. The frontend waits for cold startup, polls job progress and never retries an upload automatically. Briefly busy report reads retry with a bounded delay. Local one-origin Docker/native mode remains available without Supabase frontend variables.

### Google sign-in

The login page offers **Continue with Google** alongside email/password sign-in. It uses the same Supabase client and PKCE session storage. The return URL is the current frontend origin's `/` path; Supabase's browser SDK performs the code exchange once, restores the session and refreshes it. Callback codes/errors are removed from the browser URL. No Render callback endpoint or new frontend environment variable is required.

1. In Google Cloud's Auth Platform, configure a **Web application** OAuth client and the consent screen. Add the deployed frontend origin as an authorized JavaScript origin. Add the **Supabase callback URL shown in Authentication → Sign In / Providers → Google** as Google's authorized redirect URI; it ends in `/auth/v1/callback`. This is the Supabase URL, not the Vercel return URL.
2. Enter the Google client ID and client secret only in Supabase's Google provider settings and enable that provider. Keep Google client secrets out of Git, Vercel variables and frontend code. If the Google app is in Testing, add the intended Google accounts as test users or complete Google's audience/publishing setup.
3. In Supabase Authentication URL settings, set Site URL to the production frontend and allow its exact root return URL. Separately authorize any development/preview origin you intentionally use. Keep the existing email provider enabled. Enable **Allow new users to sign up** for first-time Google accounts.
4. Sign in with Google, reload the frontend to check session restoration, then sign out. Cancelling/denying consent or returning an expired code should show a retryable login error. Email/password sign-in remains usable.

New Google users have ordinary-user permissions and immediately search the global corpus. The frontend does not request or assign roles; protected backend `st_admins` membership remains the sole administrator authority. Supabase may link a verified Google identity to an existing account with the same email; that existing account retains its own user ID and permissions. Private source/report ownership remains unchanged. See [Supabase's Google setup](https://supabase.com/docs/guides/auth/social-sign-in/auth-google).

### 7. Usage and verification

Cloud limits default to 10 analyses per user/30 total daily and 3 reference uploads per user/10 total daily. OpenAI and Tavily start disabled: both per-user and global daily budgets must be positive, and the corresponding server API key must be present. A reservation is consumed even if an operation fails, preventing free repeated retries from bypassing budgets. Existing per-operation excerpt, model-output, web-query and page limits also apply. Hosting quotas and provider account spend limits must still be monitored; these application budgets are not an exact dollar ceiling.

Authenticated cloud users can read **`GET /api/usage`** with their normal Supabase bearer session, regardless of email/password or Google sign-in. The read-only response contains `day`, `timezone`, `server_time`, `reset_at`, and `usage` entries for `analysis`, `reference` and `ai`. Each entry reports personal `used`, `limit`, `remaining`, aggregate `global_remaining`, actionable `available`, `enabled`, `status` and a message. `available` is the smaller personal/shared remainder, or zero when disabled. No user identifiers, individual users' counts or credentials are returned; responses are not cached. Missing rows mean zero usage, while database failures remain errors.

The endpoint reads the existing `st_usage` reservations and the same quota environment variables used by enforcement. Its day is PostgreSQL `CURRENT_DATE`; reset is next midnight in that connection's actual database timezone, not browser midnight or a fixed 24-hour interval. The **Today's Usage** panel shows personal progress, remaining/shared capacity, disabled/exhausted states and the reset in the viewer's local time. It refreshes on quota-consuming request start and settlement (including failures), every five seconds while such requests are pending, every minute while visible, on focus and at the returned reset boundary. Counters measure reservations, including failed processing and partially reserved requests; they do not measure successful reports, model calls or tokens. Server enforcement remains authoritative.

Public operations require validated Supabase sessions. Reports, original uploads and exclusions are checked against the authenticated owner. Private source modification requires ownership; global modification requires protected administrator membership. BM25 uses one set of statistics across eligible distinct passage texts. Dense results use the same embedding space/cosine measure in both namespaces, merge and deduplicate before candidate limits, and enter the existing RRF ranking once. One heavy analysis/index/review operation runs at a time; cloud report/PDF reads are also serialized with it to bound memory. Other requests receive an explicit busy response instead of an unbounded queue.

After both services are deployed, verify `/api/ready`, sign-in, reference upload, exact and paraphrased submission, progress reconnect, report/PDF/JSON export, exclusion/recalculation and a second account's inability to open the first account's IDs. Restart/redeploy Render and repeat the saved-report checks. Test optional AI review only after deliberately enabling a small budget and accepting its excerpt disclosure. The live-provider checks above do not replace this final hosted Render/Vercel validation. Complete their dashboard setup and verify the deployed URLs; provider doubles alone are not proof of hosted service behavior.

Supabase Free is not a backup service. Export important reports and keep your original source library/private backups. Deactivating a source preserves evidence; it does not promise erasure from existing exports, backups or saved reports. Monitor retained versions and reports before quota exhaustion, and remove unused accounts/projects deliberately through their dashboards.
