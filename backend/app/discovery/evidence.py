"""Grounding shared by automatic nominations, reranking and investigation tools."""

import hashlib, re
from app.parsing.sentences import split_into_units


def candidate(row, unit, text, source):
    a, b = unit.start_char, unit.end_char
    x, y = row["source_start"], row["source_end"]
    if (
        text[a:b] != unit.text
        or source[x:y] != row["source_excerpt"]
        or hashlib.sha256(source.encode()).hexdigest()
        != row.get("source_document_sha256")
    ):
        raise ValueError("Invalid semantic evidence provenance")
    key = hashlib.sha256(
        f"{hashlib.sha256(text.encode()).hexdigest()}:{row['source_document_id']}:{a}:{b}:{x}:{y}".encode()
    ).hexdigest()
    citation = re.search(r"\[\d+\]|\([^()\n]{0,50}\d{4}[^()\n]{0,30}\)", text[a:b])
    marker = re.match(
        r"[ \t.,;]*(\[\d+\]|\([^()\n]{0,50}\d{4}[^()\n]{0,30}\))", text[b : b + 90]
    )
    return dict(
        row,
        candidate_id=key,
        submission_start=a,
        submission_end=b,
        submission_excerpt=text[a:b],
        citation_context=citation.group()
        if citation
        else marker.group(1)
        if marker
        else None,
        source_offset_basis="immutable indexed extracted Unicode text",
        score_is_probability=False,
        review_required=True,
    )


def original_offsets(document, rows):
    """Project normalized submission intervals to original extracted Unicode."""
    from app.parsing.document import normalize_with_map

    text = document["text"]
    original = document["original_text"]
    mapping = document["normalization_map"]
    for row in rows:
        a, b = row["submission_start"], row["submission_end"]
        if not 0 <= a < b <= len(text) or text[a:b] != row["submission_excerpt"]:
            raise ValueError("Invalid submission evidence")
        x, y = mapping["starts"][a], mapping["ends"][b - 1]
        excerpt = original[x:y]
        if normalize_with_map(excerpt)[0] != text[a:b]:
            raise ValueError("Invalid original text projection")
        row.update(
            original_submission_start=x,
            original_submission_end=y,
            original_submission_excerpt=excerpt,
            original_offset_basis="original extracted Unicode text; not PDF page coordinates",
        )


def selected_rows(saved, candidate_id):
    rows = saved.get("related_sources", [])
    selected = next((r for r in rows if r.get("candidate_id") == candidate_id), None)
    if selected is None:
        raise ValueError("Choose a saved semantic candidate")
    group = [
        r
        for r in rows
        if (r["submission_start"], r["submission_end"])
        == (selected["submission_start"], selected["submission_end"])
    ][:8]
    return selected, group


def verify_row(saved, row, source_store):
    text = saved["document"]["text"]
    a, b = row["submission_start"], row["submission_end"]
    version = source_store.get_version(row["source_document_id"])
    if version is None:
        raise ValueError("Saved source version is unavailable")
    source = version["text"]
    x, y = row["source_start"], row["source_end"]
    if (
        not 0 <= x < y <= len(source)
        or source[x:y] != row["source_excerpt"]
        or hashlib.sha256(source.encode()).hexdigest() != row["source_document_sha256"]
        or text[a:b] != row["submission_excerpt"]
    ):
        raise ValueError("Saved evidence integrity failed")
    if version["metadata"]["revision"] != row["source_revision"]:
        raise ValueError("Source revision mismatch")
    return version


def rerank(saved, candidate_id, source_store):
    import time
    from app.discovery.reranker import review_model

    _, rows = selected_rows(saved, candidate_id)
    for row in rows:
        verify_row(saved, row, source_store)
    start = time.perf_counter()
    model = review_model()
    pairs = [
        (
            " [SSS] ".join(u.text for u in split_into_units(r["source_excerpt"])),
            r["submission_excerpt"],
        )
        for r in rows
    ]
    values = model.score(pairs)
    scored = [
        {
            "candidate_id": r["candidate_id"],
            "margin": v["score"],
            "truncated": v["truncated"],
        }
        for r, v in zip(rows, values)
    ]
    scored.sort(key=lambda r: -r["margin"])
    return {
        "method": "private-roberta-review-reranker-v1",
        "checkpoint_sha256": model.manifest["checkpoint_sha256"],
        "ranking": scored,
        "elapsed_seconds": time.perf_counter() - start,
        "score_is_probability": False,
        "score_eligible": False,
        "limitation": "Experimental application transfer; rankings can prioritize related independent writing. No confirmed reuse, semantic coverage or classifier acceptance.",
    }
