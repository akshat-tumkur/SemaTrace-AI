# Privacy, evidence and licensing

## Data handling

Extraction, BM25/MiniLM retrieval, lexical alignment, word accounting, optional private ONNX reranking and saved-report export run locally. The native server binds loopback by default. Source captures, reports, model files and runtime logs are excluded from Git.

If AI review is enabled, selected submission/source excerpts are sent to OpenAI. Cloud LangGraph investigation also sends selected evidence for its decisions/explanation. The adapter requests `store:false`; this is not a blanket guarantee about every provider/account retention policy. Review the terms and controls applicable to the account used.

If public-web discovery is enabled, up to two representative queries go to Tavily. The backend contacts candidate public pages, with a bounded Tavily extraction fallback when direct capture fails. Provider-extracted snapshots are labeled separately. Keep external options off for confidential submissions.

Keys remain in backend configuration and are never returned to the frontend. Quote/identity checks, structured output and treating document instructions as untrusted input reduce risk, but cannot certify that an AI interpretation is correct.

Source uploads remain references until separately managed; deleting an analysis removes its live report/evidence and completed-job pointers. Existing exports, private backups and the reference corpus have separate lifetimes. Submitted original extracted text is retained for evidence mapping; original PDF layout reconstruction is not provided.

This is a single-user local application without multi-user authentication. Host/origin checks, body/rate limits, parser guards, one-heavy-task concurrency and checked public-address fetches are safeguards, not approval for internet exposure.

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
