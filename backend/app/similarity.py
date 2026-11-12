"""Preliminary, deterministic word-overlap accounting. Never assigns plagiarism risk.

Character positions always refer to the original normalized strings. Casefolded
word keys align; punctuation/case differences are not claimed as exact characters.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import unicodedata

VERSION = "normalized_word_runs_v1"
STOPWORDS = set(
    "a an the and or but if as at by for from in into of on to with without is are was were be been being it this that these those not no we our their its has have had may can will would could should".split()
)
BOILERPLATE = [
    "further research is needed to understand the implications of these results",
    "further research is needed",
    "in conclusion",
    "according to the study",
    "the results of this study",
    "this paper presents",
    "the results show",
]


@dataclass(frozen=True)
class Word:
    start: int
    end: int
    text: str
    key: str


def words(text: str) -> list[Word]:
    """Unicode letter/number words; marks attach, internal apostrophes stay."""
    result = []
    i = 0
    while i < len(text):
        if unicodedata.category(text[i])[0] not in "LN":
            i += 1
            continue
        start = i
        i += 1
        while i < len(text):
            category = unicodedata.category(text[i])[0]
            if category in "LNM":
                i += 1
            elif (
                text[i] in "'’"
                and i + 1 < len(text)
                and unicodedata.category(text[i + 1])[0] in "LN"
            ):
                i += 1
            else:
                break
        token = text[start:i]
        result.append(Word(start, i, token, token.casefold()))
    return result


def _id(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _percent(numerator, denominator):
    if not denominator:
        return 0
    return (2 * numerator * 10000 + denominator) // (
        2 * denominator
    )  # hundredths of a percent, half-up


def _ranges(ids, tokens):
    """Merge consecutive counted words, preserving punctuation between them."""
    out = []
    for index in sorted(ids):
        if out and index == out[-1][2] + 1:
            out[-1][1], out[-1][2] = tokens[index].end, index
        else:
            out.append([tokens[index].start, tokens[index].end, index])
    return [dict(start=a, end=b) for a, b, _ in out]


def _positions(haystack, needle):
    return [
        i
        for i in range(len(haystack) - len(needle) + 1)
        if haystack[i] == needle[0] and haystack[i : i + len(needle)] == needle
    ]


def _origins(candidate, source):
    candidates = candidate.get("source_occurrences") or [candidate]
    valid = {}
    for origin in candidates:
        start, end = origin.get("source_start_char"), origin.get("source_end_char")
        document = origin.get("source_document_id")
        if (
            isinstance(document, str)
            and document
            and type(start) is int
            and type(end) is int
            and start >= 0
            and end - start == len(source)
        ):
            key = (document, start, end)
            item = dict(
                source_id=document,
                source_document_id=document,
                start=start,
                end=end,
                occurrence_id=origin.get("occurrence_id"),
                source_revision=origin.get("revision") or document,
                source_document_sha256=origin.get("source_document_sha256"),
                name=origin.get("title") or candidate.get("source_title") or document,
                url=origin.get("url") or candidate.get("source_url") or "",
                attribution={
                    k: origin.get(k)
                    for k in (
                        "authors",
                        "collection",
                        "license_url",
                        "license_name",
                        "rights_status",
                        "attribution_url",
                        "attribution_text",
                    )
                },
            )
            # Metadata collisions resolved deterministically, never by observation order.
            if key not in valid or json.dumps(item, sort_keys=True) < json.dumps(
                valid[key], sort_keys=True
            ):
                valid[key] = item
    return [valid[k] for k in sorted(valid)]


def _options(options, length):
    supplied = options or {}
    if set(supplied) - {
        "minimum_match_words",
        "excluded_ranges",
        "excluded_source_ids",
    }:
        raise ValueError("Unknown similarity option")
    minimum = supplied.get("minimum_match_words", 5)
    if type(minimum) is not int or not 3 <= minimum <= 100:
        raise ValueError("minimum_match_words must be an integer from 3 to 100")
    ranges = supplied.get("excluded_ranges", [])
    sources = supplied.get("excluded_source_ids", [])
    if (
        not isinstance(ranges, list)
        or not isinstance(sources, list)
        or len(ranges) > 100
        or len(sources) > 100
    ):
        raise ValueError("At most 100 ranges/sources")
    if any(not isinstance(s, str) or not s or len(s) > 300 for s in sources):
        raise ValueError("Source exclusions require version identities")
    normalized = []
    for r in ranges:
        if not isinstance(r, dict) or set(r) - {"start", "end", "reason"}:
            raise ValueError("Invalid explicit range")
        a, b, reason = (
            r.get("start"),
            r.get("end"),
            r.get("reason", "EXPLICIT_DOCUMENT_EXCLUSION"),
        )
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b <= length
            or not isinstance(reason, str)
            or not reason
            or len(reason) > 200
        ):
            raise ValueError("Invalid document exclusion offsets/reason")
        normalized.append(dict(start=a, end=b, reason=reason))
    return dict(
        minimum_match_words=minimum,
        excluded_source_ids=sorted(set(sources)),
        excluded_ranges=sorted(
            normalized, key=lambda r: (r["start"], r["end"], r["reason"])
        ),
    )


def build_similarity_report(
    text: str,
    units: list[dict],
    candidates: list[dict],
    document_id: str,
    options: dict | None = None,
) -> dict:
    policy = _options(options, len(text))
    tokens = words(text)
    excluded_words = {
        i
        for i, t in enumerate(tokens)
        if any(
            t.start < r["end"] and r["start"] < t.end for r in policy["excluded_ranges"]
        )
    }
    eligible_words = set(range(len(tokens))) - excluded_words
    word_at = {(t.start, t.end): i for i, t in enumerate(tokens)}
    unit_map = {u["id"]: u for u in units}
    ledger = {}
    unresolved = {}
    source_info = {}

    def pending(candidate, reason):
        key = _id(
            [
                candidate.get("unit_id"),
                candidate.get("source_id"),
                candidate.get("source_text"),
                reason,
            ]
        )
        unresolved[key] = dict(
            candidate_id=candidate.get("candidate_id"),
            unit_id=candidate.get("unit_id"),
            source_id=candidate.get("source_document_id") or candidate.get("source_id"),
            category=candidate.get("match_type", "UNRESOLVED"),
            reason=reason,
            effective_semantic_support=candidate.get("effective_semantic_support"),
            source_title=candidate.get("source_title"),
            source_url=candidate.get("source_url"),
        )

    for candidate in sorted(
        candidates,
        key=lambda c: (
            str(c.get("unit_id")),
            str(c.get("source_id")),
            c.get("source_text", ""),
            str(c.get("candidate_id")),
        ),
    ):
        submission, source = (
            candidate.get("submission_text", ""),
            candidate.get("source_text", ""),
        )
        reason = None
        if candidate.get("dual_assessment") is not None:
            reason = "EXPERIMENTAL_DUAL_EXCLUDED"
        elif candidate.get("final_match_status") == "FILTERED_FALSE_POSITIVE":
            reason = "EXISTING_GENERIC_OR_SHORT_FILTER"
        elif (
            candidate.get("verification_method") == "llm"
            and candidate.get("llm_verdict") != "UNRELATED"
        ):
            reason = "MODEL_REPORTED_REUSE_NOT_TEXTUAL_SCORE_EVIDENCE"
        if reason:
            pending(candidate, reason)
            continue
        unit = unit_map.get(candidate.get("unit_id"))
        if (
            unit is None
            or type(unit.get("start_char")) is not int
            or type(unit.get("end_char")) is not int
            or not 0 <= unit["start_char"] <= unit["end_char"] <= len(text)
            or unit.get("text") != submission
            or text[unit["start_char"] : unit["end_char"]] != submission
        ):
            pending(candidate, "UNRESOLVED_SUBMISSION_OFFSETS")
            continue
        origins = _origins(candidate, source)
        if not origins:
            pending(candidate, "UNRESOLVED_ORIGINAL_SOURCE_PROVENANCE")
            continue
        for origin in origins:
            old = source_info.get(origin["source_id"])
            if old is None or (origin["name"], origin["url"]) < (
                old["name"],
                old["url"],
            ):
                source_info[origin["source_id"]] = origin
        left, right = words(submission), words(source)
        a, b = [w.key for w in left], [w.key for w in right]
        if len(a) * len(b) > 2_000_000 or max(len(a), len(b)) > 3000:
            pending(candidate, "ALIGNMENT_WORK_LIMIT")
            continue
        mapped = [
            word_at.get((unit["start_char"] + w.start, unit["start_char"] + w.end))
            for w in left
        ]
        if any(i is None for i in mapped):
            pending(candidate, "UNRESOLVED_TOKEN_BOUNDARIES")
            continue
        pairs = set()
        for block in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            if not block.size:
                continue
            needle = a[block.a : block.a + block.size]
            if block.size < policy["minimum_match_words"]:
                pairs.add((block.a, block.b, block.size))
                continue
            for li in _positions(a, needle):
                for ri in _positions(b, needle):
                    pairs.add((li, ri, block.size))
                    if len(pairs) * len(origins) > 512:
                        break
                if len(pairs) * len(origins) > 512:
                    break
            if len(pairs) * len(origins) > 512:
                break
        if len(pairs) * len(origins) > 512:
            pending(candidate, "AMBIGUOUS_ALIGNMENT_LIMIT")
            continue  # Do not arbitrarily publish the first subset of occurrences.
        qualifying = False
        for li, ri, size in sorted(pairs):
            # Split at explicit document exclusions so highlights cannot cover a
            # removed quote/bibliography token. Reapply minimum per remaining run.
            slices = []
            for offset in range(size):
                excluded = mapped[li + offset] in excluded_words
                if slices and slices[-1][2] == excluded:
                    slices[-1][1] += 1
                else:
                    slices.append([offset, 1, excluded])
            for offset, count, excluded in slices:
                l, r = li + offset, ri + offset
                keys = a[l : l + count]
                generic = any(" ".join(keys) in phrase for phrase in BOILERPLATE)
                content = {k for k in keys if k not in STOPWORDS}
                exclusion = (
                    "DOCUMENT_RANGE_EXCLUDED"
                    if excluded
                    else "BELOW_MINIMUM_WORDS"
                    if count < policy["minimum_match_words"]
                    else "GENERIC_LANGUAGE"
                    if generic or len(content) < 3
                    else None
                )
                policy_eligible = exclusion is None
                qualifying |= policy_eligible
                for origin in origins:
                    sid = origin["source_id"]
                    why = exclusion or (
                        "SOURCE_EXCLUDED"
                        if sid in policy["excluded_source_ids"]
                        else None
                    )
                    start, end = (
                        unit["start_char"] + left[l].start,
                        unit["start_char"] + left[l + count - 1].end,
                    )
                    source_start, source_end = (
                        origin["start"] + right[r].start,
                        origin["start"] + right[r + count - 1].end,
                    )
                    passage = (
                        candidate.get("passage_id")
                        or "text-sha256:" + hashlib.sha256(source.encode()).hexdigest()
                    )
                    identity = _id(
                        [
                            document_id,
                            start,
                            end,
                            sid,
                            passage,
                            source_start,
                            source_end,
                        ]
                    )
                    provenance = candidate.get("candidate_id") or _id(
                        [candidate.get("unit_id"), candidate.get("source_id"), source]
                    )
                    if identity in ledger:
                        ledger[identity]["provenance_ids"] = sorted(
                            set(ledger[identity]["provenance_ids"] + [provenance])
                        )
                        continue
                    ledger[identity] = dict(
                        entry_id=identity,
                        document_id=document_id,
                        document_start=start,
                        document_end=end,
                        document_token_ids=mapped[l : l + count],
                        source_document_id=sid,
                        source_passage_id=passage,
                        source_start=source_start,
                        source_end=source_end,
                        source_revision=origin["source_revision"],
                        source_occurrence_id=origin["occurrence_id"],
                        source_document_sha256=origin["source_document_sha256"],
                        source_passage_start=origin["start"],
                        source_passage_end=origin["end"],
                        submission_excerpt=text[start:end],
                        source_excerpt=source[
                            right[r].start : right[r + count - 1].end
                        ],
                        evidence_category="NORMALIZED_CONTIGUOUS_WORD_RUN",
                        provenance_ids=[provenance],
                        mechanically_grounded=True,
                        score_eligible=why is None,
                        policy_eligible=policy_eligible,
                        exclusion_reason=why,
                        citation_present=candidate.get("citation_present"),
                        citation_markers=candidate.get("citation_markers", []),
                        following_citation_markers=candidate.get(
                            "following_citation_markers", []
                        ),
                        citation_support_status=candidate.get(
                            "citation_support_status", "NOT_ASSESSED"
                        ),
                        effective_semantic_support=candidate.get(
                            "effective_semantic_support"
                        ),
                        alignment_ambiguous=sum(
                            1
                            for x in pairs
                            if x[2] == size
                            and a[x[0] : x[0] + size] == a[li : li + size]
                        )
                        > 1,
                    )
        if not qualifying:
            pending(candidate, "NO_ELIGIBLE_CONTIGUOUS_ALIGNMENT")

    entries = sorted(
        ledger.values(),
        key=lambda e: (
            e["document_start"],
            e["document_end"],
            e["source_document_id"],
            e["source_start"],
            e["entry_id"],
        ),
    )
    raw_sets = {sid: set() for sid in source_info}
    source_sets = {sid: set() for sid in source_info}
    for entry in entries:
        sid = entry["source_document_id"]
        if entry["policy_eligible"]:
            raw_sets[sid].update(entry["document_token_ids"])
        if entry["score_eligible"]:
            source_sets[sid].update(entry["document_token_ids"])
    owners = {}
    for sid in sorted(source_sets):
        for token in sorted(source_sets[sid]):
            owners.setdefault(token, []).append(sid)
    attributed = {
        sid: {i for i, alternatives in owners.items() if alternatives[0] == sid}
        for sid in source_info
    }
    denominator, numerator = len(eligible_words), len(owners)
    overall_units = _percent(numerator, denominator)
    units_by_source = {
        sid: len(ids) * 10000 // denominator if denominator else 0
        for sid, ids in attributed.items()
    }
    # Largest remainder to the rounded overall total; integer contributions add
    # exactly, even when independently rounded percentages otherwise would not.
    ranked = sorted(
        source_info,
        key=lambda sid: (
            -(len(attributed[sid]) * 10000 % denominator if denominator else 0),
            sid,
        ),
    )
    for sid in ranked[: overall_units - sum(units_by_source.values())]:
        units_by_source[sid] += 1
    sources = []
    for sid in sorted(source_info):
        info = source_info[sid]
        alternatives = sorted(
            {other for i in source_sets[sid] for other in owners[i] if other != sid}
        )
        sources.append(
            dict(
                source_id=sid,
                name=info["name"],
                url=info["url"],
                revision=info["source_revision"],
                attribution=info["attribution"],
                raw_matched_words=len(raw_sets[sid]),
                attributed_matched_words=len(attributed[sid]),
                raw_overlap_percent=_percent(len(raw_sets[sid]), denominator) / 100,
                attributed_percent=units_by_source[sid] / 100,
                attributed_percent_hundredths=units_by_source[sid],
                contribution_fraction=dict(
                    numerator=len(attributed[sid]), denominator=denominator
                ),
                excluded=sid in policy["excluded_source_ids"],
                intervals=_ranges(source_sets[sid], tokens),
                attributed_intervals=_ranges(attributed[sid], tokens),
                raw_intervals=_ranges(raw_sets[sid], tokens),
                alternative_source_ids=alternatives,
            )
        )
    assert (
        numerator <= denominator
        and sum(len(a) for a in attributed.values()) == numerator
    )
    return dict(
        method=VERSION,
        document_id=document_id,
        offset_unit="unicode_code_points",
        end_exclusive=True,
        total_document_words=len(tokens),
        total_eligible_words=denominator,
        unique_matched_words=numerator,
        textual_similarity_percent=overall_units / 100,
        textual_similarity_percent_hundredths=overall_units,
        sources=sources,
        ledger=entries,
        options=policy,
        tokens=[
            dict(
                index=i,
                start=t.start,
                end=t.end,
                text=t.text,
                eligible=i in eligible_words,
                matched=i in owners,
                primary_source_id=owners[i][0] if i in owners else None,
                alternative_source_ids=owners.get(i, [])[1:],
            )
            for i, t in enumerate(tokens)
        ],
        exclusions=dict(
            excluded_document_words=len(excluded_words),
            excluded_source_ids=policy["excluded_source_ids"],
            document_ranges=policy["excluded_ranges"],
            ineligible_ledger_entries=sum(not e["score_eligible"] for e in entries),
        ),
        unresolved_findings=[unresolved[k] for k in sorted(unresolved)],
        methodology="100 × unique eligible aligned document words / eligible document words. Unicode words, casefold keys, punctuation ignored. Contiguous runs only; minimum and content floor apply after explicit range exclusion. Source IDs are immutable document versions; primary is lexicographically first eligible source ID per word. Largest-remainder display shares sum to the rounded total. No probability of plagiarism.",
        limitations=[
            "Preliminary textual overlap only; never replaces legacy coverage/risk or establishes copying/misconduct.",
            "Only admitted corpus/web candidates are examined; no claim to search the whole internet or reproduce Turnitin's algorithm.",
            "Semantic/dual findings, missing original provenance, ambiguous work-limit overflows and existing filtered candidates are excluded.",
            "No automatic quotation or bibliography detection; select explicit normalized-document ranges. Citation presence does not exclude words automatically.",
            "SequenceMatcher is bounded greedy alignment, not exhaustive optimal plagiarism alignment. Generic-language controls are conservative and incomplete.",
            "Explicit document ranges remove intersecting words from denominator and matches. Source/minimum exclusions affect numerator only.",
        ],
    )
