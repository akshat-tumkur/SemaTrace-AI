"""One fixed source-passage transfer study. Never selects thresholds or weights."""

import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np
from app.retrieval.local_model import local_encoder
from app.discovery.reranker import review_model
from discovery_fixture import load_fixture

path = ROOT / "demo/semantic-discovery-v1.json"
data, fixture_sha = load_fixture(path)
out = ROOT / "research/results/discovery-transfer-v1.json"
if out.exists():
    raise SystemExit("Study already published. Do not repeat for tuning.")
enc = local_encoder()
model = review_model()
sources = data["sources"]
sv = np.asarray(enc.encode([s["text"] for s in sources]))
qv = np.asarray(enc.encode([q["text"] for q in data["queries"]]))
rows = []
start = time.perf_counter()
for query, vector in zip(data["queries"], qv):
    scores = sv @ vector
    indices = np.argsort(-scores, kind="stable")[:8].tolist()
    pairs = [
        (
            " [SSS] ".join(
                u.text
                for u in __import__(
                    "app.parsing.sentences", fromlist=["split_into_units"]
                ).split_into_units(sources[i]["text"])
            ),
            query["text"],
        )
        for i in indices
    ]
    measured = model.score(pairs)
    ordered = sorted(range(len(indices)), key=lambda j: (-measured[j]["score"], j))
    rows.append(
        {
            "query_id": query["id"],
            "category": query["category"],
            "relevant": query["relevant"],
            "minilm": [sources[i]["id"] for i in indices],
            "roberta": [sources[indices[j]]["id"] for j in ordered],
            "scores": [
                {
                    "source_id": sources[i]["id"],
                    "minilm": float(scores[i]),
                    "roberta_margin": m["score"],
                }
                for i, m in zip(indices, measured)
            ],
        }
    )


def metrics(key):
    positives = [r for r in rows if r["relevant"]]
    ranks = [
        next((j for j, s in enumerate(r[key], 1) if s in r["relevant"]), None)
        for r in positives
    ]
    return {
        "queries": len(ranks),
        "mrr": sum(1 / r if r else 0 for r in ranks) / len(ranks),
        **{
            "hit_at_" + str(k): sum(r is not None and r <= k for r in ranks)
            / len(ranks)
            for k in [1, 3, 5, 8]
        },
    }


result = {
    "fixture_sha256": fixture_sha,
    "model_checkpoint_sha256": model.manifest["checkpoint_sha256"],
    "protocol": data["primary"],
    "limitations": data["limitations"],
    "negative_queries": len([r for r in rows if not r["relevant"]]),
    "negative_policy": "Ranks always returned even when no source is relevant. No threshold, autonomous match, false-positive classification or semantic coverage is inferred.",
    "metrics": {k: metrics(k) for k in ["minilm", "roberta"]},
    "warm_study_seconds": time.perf_counter() - start,
    "rows": rows,
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
