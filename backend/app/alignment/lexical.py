"""Seeded word-level Smith-Waterman using permissively licensed Biopython.

The local scoring follows textreuse's documented defaults (2/-1/-1).
Seed windows, minimum length and overlapping-window reconciliation are explicit
computational adaptations, not a PAN2015 implementation. No reference code used.
"""

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from Bio.Align import PairwiseAligner
from app.similarity import words

VERSION = "seeded-word-smith-waterman-v3"


class AlignmentLimit(ValueError):
    pass


@dataclass(frozen=True)
class AlignmentConfig:
    seed_words: int = 5
    tile_words: int = 128
    stride_words: int = 96
    source_padding: int = 64
    minimum_matched_words: int = 12
    minimum_characters: int = 80
    minimum_density: float = 0.5
    maximum_windows: int = 4000


def align(submission: str, source: str, config: AlignmentConfig = AlignmentConfig()):
    a, b = words(submission), words(source)
    if min(len(a), len(b)) < config.seed_words:
        return []
    ak, bk = [w.key for w in a], [w.key for w in b]
    index = defaultdict(list)
    k = config.seed_words
    for j in range(len(b) - k + 1):
        index[tuple(bk[j : j + k])].append(j)
    vocabulary = {key: i for i, key in enumerate(sorted(set(ak) | set(bk)))}
    av = np.asarray([vocabulary[t] for t in ak], dtype=np.int32)
    bv = np.asarray([vocabulary[t] for t in bk], dtype=np.int32)
    engine = PairwiseAligner(
        mode="local",
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-1,
        extend_gap_score=-1,
    )
    findings = []
    windows = 0
    for lo in range(0, len(a), config.stride_words):
        hi = min(lo + config.tile_words, len(a))
        regions = set()
        for i in range(lo, max(lo, hi - k + 1)):
            for j in index.get(tuple(ak[i : i + k]), ()):
                # Preserve every verified maximal exact occurrence, including
                # shorter repeated copies displaced by a stronger local optimum.
                if i == 0 or j == 0 or ak[i - 1] != bk[j - 1]:
                    n = k
                    while i + n < len(a) and j + n < len(b) and ak[i + n] == bk[j + n]:
                        n += 1
                    if (
                        n >= config.minimum_matched_words
                        and min(
                            a[i + n - 1].end - a[i].start, b[j + n - 1].end - b[j].start
                        )
                        >= config.minimum_characters
                    ):
                        findings.append(
                            dict(
                                x=i,
                                y=i + n,
                                u=j,
                                v=j + n,
                                pairs={(i + t, j + t) for t in range(n)},
                            )
                        )
                predicted = j - (i - lo)
                start = max(
                    0,
                    (predicted // config.source_padding) * config.source_padding
                    - config.source_padding,
                )
                regions.add(
                    (
                        start,
                        min(
                            len(b),
                            start + config.tile_words + 3 * config.source_padding,
                        ),
                    )
                )
        for start, end in sorted(regions):
            windows += 1
            if windows > config.maximum_windows:
                raise AlignmentLimit(
                    "Lexical alignment exceeds the bounded window budget; no partial success returned"
                )
            candidates = engine.align(av[lo:hi], bv[start:end])
            try:
                best = candidates[0]
            except IndexError:
                continue
            pairs = []
            for (x, y), (u, v) in zip(*best.aligned):
                pairs.extend(
                    (lo + i, start + j)
                    for i, j in zip(range(int(x), int(y)), range(int(u), int(v)))
                    if ak[lo + i] == bk[start + j]
                )
            if len(pairs) < config.minimum_matched_words:
                continue
            x, y = pairs[0][0], pairs[-1][0] + 1
            u, v = pairs[0][1], pairs[-1][1] + 1
            if len(pairs) / max(y - x, v - u) < config.minimum_density:
                continue
            if (
                min(a[y - 1].end - a[x].start, b[v - 1].end - b[u].start)
                < config.minimum_characters
            ):
                continue
            findings.append(dict(x=x, y=y, u=u, v=v, pairs=set(pairs)))
    # Reconcile overlapping windows only when both coordinate axes overlap.
    merged = []
    for item in sorted(findings, key=lambda r: (r["x"], r["u"], r["y"], r["v"])):
        compatible = [
            r
            for r in merged
            if item["x"] < r["y"]
            and r["x"] < item["y"]
            and item["u"] < r["v"]
            and r["u"] < item["v"]
            and abs((item["u"] - item["x"]) - (r["u"] - r["x"]))
            <= config.source_padding
        ]
        if compatible:
            for r in compatible:
                item = dict(
                    x=min(item["x"], r["x"]),
                    y=max(item["y"], r["y"]),
                    u=min(item["u"], r["u"]),
                    v=max(item["v"], r["v"]),
                    pairs=item["pairs"] | r["pairs"],
                )
                merged.remove(r)
        merged.append(item)
    result = []
    for item in sorted(merged, key=lambda r: (r["x"], r["u"])):
        x, y, u, v = (item[k] for k in ("x", "y", "u", "v"))
        runs = []
        for i, j in sorted(item["pairs"]):
            if runs and i == runs[-1][1] and j == runs[-1][3]:
                runs[-1][1] += 1
                runs[-1][3] += 1
            else:
                runs.append([i, i + 1, j, j + 1])
        precise = [
            {
                "submission_start": a[i].start,
                "submission_end": a[j - 1].end,
                "source_start": b[u].start,
                "source_end": b[v - 1].end,
                "words": j - i,
            }
            for i, j, u, v in runs
        ]
        region = {
            "submission_start": a[x].start,
            "submission_end": a[y - 1].end,
            "source_start": b[u].start,
            "source_end": b[v - 1].end,
        }
        result.append(
            dict(
                region,
                method=VERSION,
                matched_words=len({i for i, j in item["pairs"]}),
                matched_runs=precise,
                category="EXACT_REUSE"
                if submission[region["submission_start"] : region["submission_end"]]
                == source[region["source_start"] : region["source_end"]]
                else "MODIFIED_REUSE",
                boundary_granularity="lexical_passage",
                score_eligible=False,
                accounting_note="Passage includes edits; only separately grounded matching word runs may receive textual coverage.",
            )
        )
    return result
