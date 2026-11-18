"""Automatic advisory semantic review with source- and quote-grounded findings.

This is a distinct LLM review method, not acceptance of the failed RoBERTa
classifier. No model outcome contributes to deterministic matching-word coverage.
"""

import hashlib, re, time
from collections import defaultdict
from datetime import datetime, timezone
from app.discovery.evidence import verify_row
from app.product.providers import OpenAIReviewer, ProviderUnavailable

MAX_UNITS = 48
BATCH_SIZE = 6
RELATIONS = [
    "PARAPHRASE",
    "PARTIAL_CORRESPONDENCE",
    "RELATED_ONLY",
    "DIFFERENT_MEANING",
    "INSUFFICIENT_CONTEXT",
]
ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "unit_id": {"type": "string"},
        "candidate_id": {"type": "string"},
        "relation": {"type": "string", "enum": RELATIONS},
        "submission_quote": {"type": "string"},
        "source_quote": {"type": "string"},
        "explanation": {"type": "string"},
        "differences": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "unit_id",
        "candidate_id",
        "relation",
        "submission_quote",
        "source_quote",
        "explanation",
        "differences",
    ],
}
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"reviews": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["reviews"],
}
PROMPT = """Compare each submitted sentence with its supplied source passages. Choose the source that best supports a specific meaning correspondence, or return RELATED_ONLY, DIFFERENT_MEANING or INSUFFICIENT_CONTEXT.
PARAPHRASE: the same specific proposition, entities, causal direction and important qualifiers expressed differently. PARTIAL_CORRESPONDENCE: only a clearly identifiable portion has the same meaning; explain changed details.
Shared vocabulary, a broad subject, boilerplate and merely related facts are RELATED_ONLY. Contradictory quantities, dates, negation or reversed causal roles must be stated in differences; never call a central contradiction a full paraphrase.
For a positive relation, copy one exact contiguous quote of at least four words from EACH side. Quotes must be verbatim, not your rewrite. Keep quotes below400characters and explanations below350characters. This identifies possible meaning correspondence for human review; it cannot establish copying history. Return one review per unit. Candidate IDs are local aliases from that unit only. Never invent a source or combine quotes."""


def eligible(text):
    return (
        8 <= len(re.findall(r"\b\w+\b", text))
        and len(text) <= 1800
        and not re.match(
            r"(?i)\s*(source(?: attribution)?|license|references|bibliography)\s*:",
            text,
        )
        and not re.search(r"https?://", text)
    )


def context_for(saved, row, store, cache):
    sid = row["source_document_id"]
    if sid not in cache:
        cache[sid] = verify_row(saved, row, store)
    version = cache[sid]
    text = version["text"]
    a, b = row["source_start"], row["source_end"]
    if text[a:b] != row["source_excerpt"]:
        raise ValueError("Saved source excerpt changed")
    left = text.rfind("\n\n", 0, a)
    right = text.find("\n\n", b)
    x = left + 2 if left >= 0 else 0
    y = right if right >= 0 else len(text)
    if y - x > 1800:
        x = max(x, a - 450)
        y = min(y, max(b, a + 900))
    if y - x > 2200:
        return None
    return dict(
        row, source_context_start=x, source_context_end=y, source_context=text[x:y]
    )


def unique_span(text, quote):
    if (
        not isinstance(quote, str)
        or not 12 <= len(quote) <= 800
        or len(re.findall(r"\b\w+\b", quote)) < 4
    ):
        return None
    first = text.find(quote)
    if first < 0 or text.find(quote, first + 1) >= 0:
        return None
    return first, first + len(quote)


def ground(item, unit, candidate):
    relation = item.get("relation")
    if relation not in RELATIONS:
        raise ValueError("Invalid relation")
    explanation = item.get("explanation")
    differences = item.get("differences")
    if (
        not isinstance(explanation, str)
        or not explanation.strip()
        or len(explanation) > 1200
        or not isinstance(differences, list)
        or len(differences) > 8
        or any(not isinstance(s, str) or len(s) > 800 for s in differences)
    ):
        raise ValueError("Invalid explanation")
    if re.search(
        r"plagiar|misconduct|definitely copied|confirmed copying", explanation, re.I
    ):
        raise ValueError("Unsupported misconduct verdict")
    if relation not in ("PARAPHRASE", "PARTIAL_CORRESPONDENCE"):
        return None
    target = unit["submission_excerpt"]
    source = candidate["source_context"]
    a = unique_span(target, item.get("submission_quote"))
    b = unique_span(source, item.get("source_quote"))
    if not a or not b:
        raise ValueError("Literal quotes are absent, too short or ambiguous")
    start = unit["submission_start"] + a[0]
    end = unit["submission_start"] + a[1]
    x = candidate["source_context_start"] + b[0]
    y = candidate["source_context_start"] + b[1]
    finding_id = hashlib.sha256(
        f"{candidate['candidate_id']}:{start}:{end}:{x}:{y}".encode()
    ).hexdigest()
    return {
        k: v
        for k, v in dict(
            candidate,
            finding_id=finding_id,
            relation=relation,
            submission_start=start,
            submission_end=end,
            submission_excerpt=target[a[0] : a[1]],
            source_start=x,
            source_end=y,
            source_excerpt=source[b[0] : b[1]],
            submission_context=target,
            source_context_excerpt=source,
            explanation=explanation,
            differences=differences,
            status="AI_IDENTIFIED_CORRESPONDENCE",
            score_eligible=False,
            independently_validated=False,
            score_is_probability=False,
            review_required=True,
            warning="Model-identified meaning correspondence; inspect the literal evidence. It is not proof of copying.",
        ).items()
        if k not in ("source_context", "source_context_start", "source_context_end")
    }


def review_saved(saved, store, previous=None, provider=None, progress=None):
    previous = previous or {}
    done = set(previous.get("reviewed_unit_starts", []))
    skipped = list(previous.get("skipped_units", []))
    skip_ids = {r["submission_start"] for r in skipped}
    groups = defaultdict(list)
    for row in saved.get("related_sources", []):
        if row.get("candidate_id") and eligible(row["submission_excerpt"]):
            groups[row["submission_start"]].append(row)
    starts = sorted(
        (s for s in groups if s not in done and s not in skip_ids),
        key=lambda s: (-max(r["score"] for r in groups[s]), s),
    )[:MAX_UNITS]
    result = dict(
        method="grounded-openai-passage-review-v1",
        provider="openai",
        status="complete",
        findings=list(previous.get("findings", [])),
        reviews=list(previous.get("reviews", [])),
        reviewed_unit_starts=sorted(done),
        eligible_units=len(groups),
        warnings=[],
        semantic_coverage_percent=None,
        score_eligible=False,
        independently_validated=False,
        skipped_units=skipped,
        note="AI-reviewed semantic correspondences are advisory. Only literal quoted spans are highlighted; no semantic or combined percentage is calculated.",
    )
    if not starts:
        result["remaining_units"] = len(set(groups) - done - skip_ids)
        for key in (
            "model",
            "usage",
            "calls",
            "elapsed_seconds",
            "created_at",
            "review_runs",
            "warnings",
        ):
            if key in previous:
                result[key] = previous[key]
        return result
    try:
        provider = provider or OpenAIReviewer(max_calls=8)
    except ProviderUnavailable as e:
        result.update(
            status="unavailable",
            warnings=[str(e)],
            remaining_units=len(set(groups) - done - skip_ids),
        )
        return result
    cache = {}
    began = time.perf_counter()
    rejected = 0
    for at in range(0, len(starts), BATCH_SIZE):
        if progress:
            progress(
                f"AI passage review: {min(at + BATCH_SIZE, len(starts))} / {len(starts)} selected sentences",
                75 + int(18 * at / max(1, len(starts))),
            )
        prepared = []
        lookup = {}
        for index, start in enumerate(starts[at : at + BATCH_SIZE]):
            unit = sorted(groups[start], key=lambda r: r.get("minilm_rank", 999))[0]
            candidates = []
            # Distinct source contexts, including competitors; no cosine cutoff.
            seen = set()
            remaining_context = 5700 - len(unit["submission_excerpt"])
            for row in sorted(groups[start], key=lambda r: r.get("minilm_rank", 999)):
                c = context_for(saved, row, store, cache)
                if c is None:
                    continue
                if len(c["source_context"]) > remaining_context:
                    continue
                key = (
                    c["source_document_id"],
                    c["source_context_start"],
                    c["source_context_end"],
                )
                if key in seen:
                    continue
                seen.add(key)
                remaining_context -= len(c["source_context"]) + 80
                alias = f"C{len(candidates)}"
                candidates.append(
                    {"candidate_id": alias, "source": c["source_context"]}
                )
                lookup[(f"U{index}", alias)] = (unit, c)
                if len(candidates) == 3:
                    break
            if candidates:
                prepared.append(
                    {
                        "unit_id": f"U{index}",
                        "submission": unit["submission_excerpt"],
                        "sources": candidates,
                    }
                )
            else:
                skip_ids.add(start)
                skipped.append(
                    {
                        "submission_start": start,
                        "reason": "No source context fit the bounded review input. Local retrieval evidence remains available.",
                    }
                )
        if not prepared:
            continue
        try:
            answer = provider.ask(PROMPT, {"units": prepared}, SCHEMA, 2200)
        except ProviderUnavailable as e:
            result["status"] = "partial" if done else "failed"
            result["warnings"].append(str(e))
            break
        items = answer.get("reviews", [])
        returned = set()
        if not isinstance(items, list):
            items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            uid = item.get("unit_id")
            key = (uid, item.get("candidate_id"))
            if uid in returned or key not in lookup:
                rejected += 1
                continue
            returned.add(uid)
            unit, c = lookup[key]
            try:
                finding = ground(item, unit, c)
                if finding:
                    from app.discovery.evidence import original_offsets

                    original_offsets(saved["document"], [finding])
                    result["findings"].append(finding)
                result["reviews"].append(
                    {
                        "submission_start": unit["submission_start"],
                        "candidate_id": c["candidate_id"],
                        "relation": item["relation"],
                        "explanation": item["explanation"],
                        "differences": item["differences"],
                        "evidence_status": "GROUNDED_QUOTES"
                        if finding
                        else "NO_POSITIVE_FINDING",
                    }
                )
                done.add(unit["submission_start"])
            except ValueError:
                rejected += 1
                result["reviews"].append(
                    {
                        "submission_start": unit["submission_start"],
                        "candidate_id": c["candidate_id"],
                        "relation": "INSUFFICIENT_CONTEXT",
                        "evidence_status": "REJECTED_UNGROUNDED_OUTPUT",
                    }
                )
                done.add(unit["submission_start"])
    result.update(
        reviewed_unit_starts=sorted(done),
        remaining_units=len(set(groups) - done - skip_ids),
        model=provider.actual_model,
        calls=provider.calls,
        usage=provider.usage,
        elapsed_seconds=time.perf_counter() - began,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    if result["remaining_units"] and result["status"] == "complete":
        result["status"] = "partial"
    if rejected:
        result["warnings"].append(
            f"{rejected} model outputs failed identity, quote or explanation validation and were not accepted."
        )
    if skipped:
        result["warnings"].append(
            f"{len(skipped)} units could not fit supported source-context bounds and were not AI-reviewed."
        )
    result["review_runs"] = list(previous.get("review_runs", [])) + [
        {
            "model": provider.actual_model,
            "calls": provider.calls,
            "usage": provider.usage,
            "elapsed_seconds": result["elapsed_seconds"],
            "created_at": result["created_at"],
            "reviewed_units": len(done) - len(previous.get("reviewed_unit_starts", [])),
        }
    ]
    return result
