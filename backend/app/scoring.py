"""Coverage from saved aligner word runs, with no second alignment or inference."""

import hashlib
from app.similarity import words, _options, _percent, _ranges, STOPWORDS


def build_report(text, findings, sources, options=None):
    policy = _options(options, len(text))
    tokens = words(text)
    source_tokens = {}
    for sid, source in sources.items():
        if hashlib.sha256(source["text"].encode()).hexdigest() != source["sha256"]:
            raise ValueError("Saved source integrity mismatch")
        source_tokens[sid] = words(source["text"])
    excluded = {
        i
        for i, t in enumerate(tokens)
        if any(
            t.start < r["end"] and r["start"] < t.end for r in policy["excluded_ranges"]
        )
    }
    raw = {sid: set() for sid in sources}
    included = {sid: set() for sid in sources}
    ledger = []
    for finding in findings:
        sid = finding["source_document_id"]
        source = sources[sid]["text"]
        for run in finding["matched_runs"]:
            a, b, x, y = (
                run[k]
                for k in (
                    "submission_start",
                    "submission_end",
                    "source_start",
                    "source_end",
                )
            )
            if not (0 <= a < b <= len(text) and 0 <= x < y <= len(source)):
                raise ValueError("Invalid evidence coordinates")
            if not (
                finding["submission_start"] <= a < b <= finding["submission_end"]
                and finding["source_start"] <= x < y <= finding["source_end"]
            ):
                raise ValueError("Word run exceeds its passage")
            left, right = words(text[a:b]), words(source[x:y])
            if [w.key for w in left] != [w.key for w in right]:
                raise ValueError("Word evidence does not match source")
            original = [w for w in source_tokens[sid] if x <= w.start and w.end <= y]
            if (
                not original
                or original[0].start != x
                or original[-1].end != y
                or len(original) != len(right)
            ):
                raise ValueError("Invalid source token boundaries")
            ids = [i for i, t in enumerate(tokens) if a <= t.start and t.end <= b]
            if (
                not ids
                or tokens[ids[0]].start != a
                or tokens[ids[-1]].end != b
                or len(ids) != len(left)
            ):
                raise ValueError("Invalid document token boundaries")
            # Re-evaluate minimum length after splitting explicit document exclusions.
            groups = []
            for offset, i in enumerate(ids):
                state = i in excluded
                if groups and groups[-1][2] == state:
                    groups[-1][1] = offset + 1
                else:
                    groups.append([offset, offset + 1, state])
            for lo, hi, is_excluded in groups:
                selected = ids[lo:hi]
                keys = [tokens[i].key for i in selected]
                why = (
                    "DOCUMENT_RANGE_EXCLUDED"
                    if is_excluded
                    else "BELOW_MINIMUM_WORDS"
                    if len(selected) < policy["minimum_match_words"]
                    else "GENERIC_LANGUAGE"
                    if len(set(keys) - STOPWORDS) < 3
                    else None
                )
                eligible = why is None
                if eligible:
                    raw[sid].update(selected)
                if eligible and sid not in policy["excluded_source_ids"]:
                    included[sid].update(selected)
                why = why or (
                    "SOURCE_EXCLUDED" if sid in policy["excluded_source_ids"] else None
                )
                ds, de = tokens[selected[0]].start, tokens[selected[-1]].end
                ss, se = x + right[lo].start, x + right[hi - 1].end
                ledger.append(
                    {
                        "entry_id": hashlib.sha256(
                            f"{sid}:{ds}:{de}:{ss}:{se}".encode()
                        ).hexdigest(),
                        "document_start": ds,
                        "document_end": de,
                        "source_start": ss,
                        "source_end": se,
                        "source_document_id": sid,
                        "source_revision": sources[sid]["revision"],
                        "source_document_sha256": sources[sid]["sha256"],
                        "submission_excerpt": text[ds:de],
                        "source_excerpt": source[ss:se],
                        "document_token_ids": selected,
                        "score_eligible": why is None,
                        "exclusion_reason": why,
                        "evidence_category": "NORMALIZED_ALIGNED_WORD_RUN",
                        "mechanically_grounded": True,
                    }
                )
    owners = {}
    for sid in sorted(included):
        for i in included[sid]:
            owners.setdefault(i, []).append(sid)
    primary = {
        sid: {i for i, values in owners.items() if values[0] == sid} for sid in sources
    }
    denominator = len(tokens) - len(excluded)
    total = _percent(len(owners), denominator)
    shares = {
        sid: len(v) * 10000 // denominator if denominator else 0
        for sid, v in primary.items()
    }
    order = sorted(
        sources,
        key=lambda sid: (
            -(len(primary[sid]) * 10000 % denominator if denominator else 0),
            sid,
        ),
    )
    for sid in order[: total - sum(shares.values())]:
        shares[sid] += 1
    rows = []
    for sid, source in sources.items():
        if not raw[sid] and not any(r["source_document_id"] == sid for r in ledger):
            continue
        rows.append(
            {
                "source_id": sid,
                "name": source["title"],
                "url": source["url"],
                "attribution": source["attribution"],
                "raw_matched_words": len(raw[sid]),
                "attributed_matched_words": len(primary[sid]),
                "attributed_percent": shares[sid] / 100,
                "raw_overlap_percent": _percent(len(raw[sid]), denominator) / 100,
                "excluded": sid in policy["excluded_source_ids"],
                "intervals": _ranges(included[sid], tokens),
                "raw_intervals": _ranges(raw[sid], tokens),
                "alternative_source_ids": sorted(
                    {
                        other
                        for i in included[sid]
                        for other in owners[i]
                        if other != sid
                    }
                ),
            }
        )
    return {
        "method": "aligned-word-union-v1",
        "document_id": "submission:" + hashlib.sha256(text.encode()).hexdigest(),
        "textual_similarity_percent": total / 100,
        "textual_similarity_percent_hundredths": total,
        "total_document_words": len(tokens),
        "total_eligible_words": denominator,
        "unique_matched_words": len(owners),
        "sources": rows,
        "ledger": ledger,
        "options": policy,
        "unresolved_findings": [],
        "exclusions": {
            "excluded_document_words": len(excluded),
            "excluded_source_ids": policy["excluded_source_ids"],
            "document_ranges": policy["excluded_ranges"],
        },
        "methodology": "Unique eligible normalized aligned submission words divided by eligible document words. Passage envelopes include edits; only grounded matching runs count. Overlaps count once; primary attribution uses the first eligible source-version ID. Source exclusions affect numerator; explicit document ranges affect both numerator and denominator.",
        "limitations": [
            "Similarity is not a probability of plagiarism or proof of copying.",
            "Only indexed sources admitted to alignment are examined. Missing matches do not establish originality.",
            "Semantic candidates receive no coverage.",
            "Quotes and bibliographies require explicit exclusion; no automatic claim of reliable detection.",
        ],
        "offset_unit": "unicode_code_points",
        "end_exclusive": True,
    }
