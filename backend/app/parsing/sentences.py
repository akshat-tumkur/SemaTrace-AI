from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_TITLE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof)\.(?=[ \t]+\S)", re.IGNORECASE)
_DOTTED = re.compile(r"\b(?:U\.S\.|U\.K\.|e\.g\.|i\.e\.)", re.IGNORECASE)
_INITIALS = re.compile(r"(?<![\w.])(?:[A-Z]\.[ \t]+){1,3}(?=[A-Z][a-z])")
_FOLLOWING_MARKERS = re.compile(
    r"""[ \t]*(?P<closing>["”’']?)[ \t]*(?P<marker>\[\d+\]|\([^()\n]{2,40},?[ \t]*\d{4}\))"""
)
_ABBREVIATION_CONTINUATIONS = {
    "Army",
    "Navy",
    "Air",
    "Department",
    "Government",
    "Capitol",
    "Congress",
    "Supreme",
}


@dataclass(frozen=True)
class SentenceUnit:
    id: str
    text: str
    start_char: int
    end_char: int
    unit_type: str = "sentence"
    following_citation_spans: tuple[tuple[int, int], ...] = ()


def split_into_units(text: str, *, preserve_urls: bool = False) -> list[SentenceUnit]:
    # Mask only boundary detection, at equal string length. Returned text is always
    # an untouched original slice; no normalized/fabricated evidence is introduced.
    masked = list(text)
    if preserve_urls:
        for match in re.finditer(r'https?://[^\s<>"\']+', text):
            url = match.group().rstrip(".,;:!?")
            for at in range(match.start(), match.start() + len(url)):
                if text[at] in ".!?":
                    masked[at] = "\uffff"
    for match in re.finditer(r"(?<=\d)\.(?=\d)", text):
        masked[match.start()] = "\uffff"
    for pattern in (_TITLE, _INITIALS):
        for match in pattern.finditer(text):
            for at in range(match.start(), match.end()):
                if text[at] == ".":
                    masked[at] = "\uffff"
    for match in _DOTTED.finditer(text):
        following = re.match(r"[ \t]+([A-Za-z]+)", text[match.end() :])
        word = following.group(1) if following else ""
        continuation = match.group().lower() in {"e.g.", "i.e."} or (
            word and (word[0].islower() or word in _ABBREVIATION_CONTINUATIONS)
        )
        for at in range(match.start(), match.end() - 1):
            if text[at] == ".":
                masked[at] = "\uffff"
        if continuation:
            masked[match.end() - 1] = "\uffff"
    units: list[SentenceUnit] = []
    consumed = 0
    for match in _SENTENCE_PATTERN.finditer("".join(masked)):
        start, end = max(match.start(), consumed), match.end()
        if start >= end:
            continue
        sentence_end = end
        following_citations = []
        # A trailing citation on the SAME line belongs to this completed sentence.
        # Never reach across a paragraph/newline to claim attribution.
        if text[end - 1] in ".!?":
            while marker := _FOLLOWING_MARKERS.match(text, end):
                if not following_citations and marker.group("closing"):
                    sentence_end = marker.end("closing")
                following_citations.append(marker.span("marker"))
                end = marker.end()
                while end < len(text) and text[end] in ".!?":
                    end += 1
        consumed = end
        raw = text[start:sentence_end]
        sentence = raw.strip()
        if not sentence:
            continue
        start_char = start + len(raw) - len(raw.lstrip())
        units.append(
            SentenceUnit(
                id=f"u{len(units) + 1}",
                text=sentence,
                start_char=start_char,
                end_char=start_char + len(sentence),
                following_citation_spans=tuple(following_citations),
            )
        )
    return units
