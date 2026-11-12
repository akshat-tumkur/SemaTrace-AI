"""Exact normalized word seeds for brief wording below passage-length limits.

Uses the existing report minimum (five words), not a semantic threshold. Only
maximal exact word runs with at least three content-word types are emitted.
"""

from collections import defaultdict
from app.similarity import words, STOPWORDS
from app.alignment.lexical import AlignmentLimit


def exact_runs(text, source, minimum=5):
    a, b = words(text), words(source)
    ak, bk = [w.key for w in a], [w.key for w in b]
    index = defaultdict(list)
    for j in range(len(b) - minimum + 1):
        index[tuple(bk[j : j + minimum])].append(j)
    findings = []
    work = 0
    for i in range(len(a) - minimum + 1):
        for j in index.get(tuple(ak[i : i + minimum]), ()):
            work += 1
            if work > 200000:
                raise AlignmentLimit("Brief wording search exceeds its work budget")
            if i and j and ak[i - 1] == bk[j - 1]:
                continue
            n = minimum
            while i + n < len(a) and j + n < len(b) and ak[i + n] == bk[j + n]:
                n += 1
            if len(set(ak[i : i + n]) - STOPWORDS) < 3:
                continue
            x, y, u, v = a[i].start, a[i + n - 1].end, b[j].start, b[j + n - 1].end
            findings.append(
                {
                    "submission_start": x,
                    "submission_end": y,
                    "source_start": u,
                    "source_end": v,
                    "category": "EXACT_REUSE"
                    if text[x:y] == source[u:v]
                    else "MODIFIED_REUSE",
                    "method": "exact-word-seeds-v1",
                    "matched_words": n,
                    "boundary_granularity": "word_run",
                    "score_eligible": False,
                    "matched_runs": [
                        {
                            "submission_start": x,
                            "submission_end": y,
                            "source_start": u,
                            "source_end": v,
                            "words": n,
                        }
                    ],
                }
            )
    return findings
