# Privacy, evidence and licensing

## Data handling

In local mode, extraction, BM25/MiniLM retrieval, lexical alignment, word accounting, optional private ONNX reranking and saved-report export run locally. The native server binds loopback by default. Source captures, reports, model files and runtime logs are excluded from Git.

If AI review is enabled, selected submission/source excerpts are sent to OpenAI. Cloud LangGraph investigation also sends selected evidence for its decisions/explanation. The adapter requests `store:false`; this is not a blanket guarantee about every provider/account retention policy. Review the terms and controls applicable to the account used.

If public-web discovery is enabled, up to two representative queries go to Tavily. The backend contacts candidate public pages, with a bounded Tavily extraction fallback when direct capture fails. Provider-extracted snapshots are labeled separately. Keep external options off for confidential submissions.

Keys remain in backend configuration and are never returned to the frontend. Quote/identity checks, structured output and treating document instructions as untrusted input reduce risk, but cannot certify that an AI interpretation is correct.

Source uploads remain references until separately managed; deleting an analysis removes its live report/evidence and completed-job pointers. Existing exports, private backups and the reference corpus have separate lifetimes. Submitted original extracted text is retained for evidence mapping; original PDF layout reconstruction is not provided.

Local SQLite mode is a single-user application without authentication and must remain private. Optional cloud mode requires Supabase sign-in: submissions, report/job records, exclusions and private references are scoped to that account. The global reference corpus is deliberately shared with all authenticated users and can be changed only by administrators. Uploading a global reference makes its original file, extracted text and metadata available to those users. Raw submissions and reference files are retained in private Supabase Storage; extracted passages/metadata/reports are stored in PostgreSQL; MiniLM vectors and provenance identifiers are sent to Pinecone in one global namespace and separate private account namespaces. Reference text is not included in vector metadata. Hosting these services sends data outside the laptop even when OpenAI and Tavily are disabled.

Cloud credentials stay server-side; only Supabase's public publishable key and a user's session belong in the frontend. PostgreSQL row policies and private Storage policies provide additional owner isolation. Public signup should be disabled for the operator-provisioned portfolio demo. Host/origin checks, bounded requests, persistent daily operation budgets and single-heavy-task concurrency are additional safeguards. Cloud AI/web calls default off and still require explicit excerpt/query opt-in. Free providers can pause or exhaust quotas; see [deployment](deployment.md#free-cloud-hosting).

Removing a cloud source deactivates it for new searches and schedules remote vector cleanup; immutable historical source versions and saved-report evidence remain. Deleting a cloud report removes its live report/job pointers, while content-addressed uploaded blobs may remain until the operator manages account/storage retention. Account deletion, exports, provider retention and private backups have separate lifetimes; do not treat a UI removal as a guarantee of erasure everywhere.

## Evidence interpretation

Matching-word coverage measures a union of qualifying normalized submitted words. It is not a misconduct probability. Semantic candidates are retrieval results; AI correspondences/explanations are advisory. A mechanically valid quote does not prove semantic equivalence, authorship, historical copying or correct citation support. Missing evidence does not establish originality.

Reports retain immutable source identity, revision, fingerprints, literal excerpts and positions. Primary source contribution is deterministic accounting across overlapping references, not an authorship decision. Copyright, quotation permission and academic context require separate consideration.

## Data and model rights

- Wikipedia material retains revision-specific author/license attribution. Included adapted demonstration text and exact-copy controls retain their attribution/CC BY-SA notices.
- Purpose-written thermal-storage and semantic-control fixtures are CC0 software demonstration material. They are not actual published experiments or independently representative accuracy datasets.
- Private public-web snapshots and uploaded references do not acquire redistribution clearance through ingestion. They are excluded from the public source repository.
- ARC-Sim research data is distributed by its authors under CC BY-NC-SA 3.0 terms. Research inputs and derived private weights are not published here; a pretrained model's license does not erase dataset obligations. See [the official dataset repository](https://github.com/maclaughlin/ARC-Sim) and [our research protocol](../research/README.md).
- PAN reference corpora/software used in historical research remain private. The production aligner uses Biopython; it does not import the academic-only PAN2014/2015 reference implementation.

## Software and asset notices

- Biopython's alignment implementation offers its documented BSD/Biopython licensing; [the retained notice](legal/biopython-LICENSE.txt) accompanies this project.
- ReportLab's [BSD-style notice](legal/reportlab-LICENSE.txt) is retained.
- Bundled Noto Sans fonts retain the [SIL Open Font License 1.1](../backend/app/reports/assets/OFL.txt); source identity is recorded in `configs/report-fonts.lock.json`.
- PyMuPDF/MuPDF are available under AGPL or commercial licensing terms. Consult [upstream's licensing statement](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) for the applicable obligations.
- Provisioned pretrained models and other dependencies retain their own licenses; downloading or exporting them does not relicense them.

No project-wide source license is declared here. Public availability does not itself grant unrestricted reuse or commercial redistribution permission. These notices preserve known provenance and do not certify compliance for every distribution/use case.
