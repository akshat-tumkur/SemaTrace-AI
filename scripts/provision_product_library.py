"""One bounded licensed-source expansion. No research datasets or model calls."""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.corpus.wikimedia import WikipediaClient
from app.corpus.cli import ingest_manifest, export_lock
from app.storage.source_corpus import SourceCorpusStore
from app.retrieval.generations import GenerationStore, IndexSpec

TITLES = [
    "Artificial intelligence",
    "Neural network",
    "Deep learning",
    "Plagiarism",
    "Academic integrity",
    "Peer review",
    "Scientific method",
    "Statistics",
    "Probability",
    "Linear regression",
    "Logistic regression",
    "Decision tree",
    "Random forest",
    "Support vector machine",
    "Cluster analysis",
    "Cosine similarity",
    "Tf–idf",
    "Database",
    "Relational database",
    "Database normalization",
    "Computer network",
    "Internet protocol suite",
    "Computer security",
    "Cryptography",
    "Hash function",
    "Operating system",
    "Distributed computing",
    "Cloud computing",
    "Software testing",
    "Version control",
    "Algorithm",
    "Data structure",
    "Photosynthesis",
    "Cellular respiration",
    "DNA",
    "Genetics",
    "Evolution",
    "Ecosystem",
    "Biodiversity",
    "Climate change",
    "Carbon cycle",
    "Greenhouse effect",
    "Renewable energy",
    "Wind power",
    "Hydroelectricity",
    "Recycling",
    "Groundwater",
    "Aquifer",
]


def main():
    out = ROOT / ".local/product-library-20261121"
    out.mkdir(parents=True, exist_ok=True)
    client = WikipediaClient(max_total_bytes=100 * 1024 * 1024, max_requests=64)
    path = out / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = client.resolve(TITLES)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Resolved", len(manifest["entries"]), "source revisions", flush=True)
    store = SourceCorpusStore(ROOT / "backend/data/source-corpus/corpus.sqlite3")
    result = ingest_manifest(store, manifest, client, limit=len(manifest["entries"]))
    (out / "ingestion.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Ingestion", result["status"], result["metrics"], flush=True)
    lock = export_lock(store, manifest)
    (out / "sources.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    corpus = store.snapshot()
    print(
        "Building persistent MiniLM index:",
        len(corpus.documents),
        "passages",
        flush=True,
    )
    generation, metrics = GenerationStore(corpus.retrieval_index_path).build(
        corpus, IndexSpec.local(), api_key="", batch_size=32
    )
    (out / "index.json").write_text(
        json.dumps(
            {
                "generation": generation.id,
                "metrics": metrics,
                "corpus": dict(corpus.corpus_info),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Ready",
        len(corpus.source_documents),
        "sources",
        len(corpus.documents),
        "unique passages",
        flush=True,
    )


if __name__ == "__main__":
    main()
