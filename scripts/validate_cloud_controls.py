"""One bounded real-provider functional check on frozen CC0 controls."""

import json, sys, os, hashlib
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.product.providers import OpenAIReviewer
from app.product.review import SCHEMA, PROMPT, ground

for key, value in dotenv_values(ROOT / ".env").items():
    if key in ("OPENAI_API_KEY", "OPENAI_MODEL") and value:
        os.environ[key] = value
path = ROOT / "demo/product-semantic-controls.json"
data = json.loads(path.read_text())
provider = OpenAIReviewer(max_calls=2)
rows = []
out = ROOT / ".local/product-completion-validation/cloud-controls.json"
if out.exists():
    raise SystemExit(
        "Control result already recorded; do not repeat it for a better result."
    )
for at in range(0, len(data["cases"]), 6):
    batch = data["cases"][at : at + 6]
    payload = {
        "units": [
            {
                "unit_id": str(i),
                "submission": c["submission"],
                "sources": [{"candidate_id": "C0", "source": c["source"]}],
            }
            for i, c in enumerate(batch)
        ]
    }
    answer = provider.ask(PROMPT, payload, SCHEMA, 2200)
    returned = {r["unit_id"]: r for r in answer["reviews"]}
    for i, case in enumerate(batch):
        item = returned.get(str(i))
        accepted = False
        grounded = False
        if item and item.get("candidate_id") == "C0":
            try:
                finding = ground(
                    item,
                    {"submission_start": 0, "submission_excerpt": case["submission"]},
                    {
                        "candidate_id": case["id"],
                        "source_context_start": 0,
                        "source_context": case["source"],
                    },
                )
                accepted = bool(finding)
                grounded = bool(finding)
            except ValueError:
                pass
        rows.append(
            {
                "id": case["id"],
                "expected_positive": case["positive"],
                "accepted_positive": accepted,
                "relation": item.get("relation") if item else "MISSING",
                "grounded_positive": grounded,
                "pass": accepted == case["positive"],
            }
        )
result = {
    "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "scope": data["purpose"],
    "model": provider.actual_model,
    "calls": provider.calls,
    "usage": provider.usage,
    "cases": rows,
    "all_controls_pass": all(r["pass"] for r in rows),
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
