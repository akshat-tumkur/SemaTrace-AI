from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)


@dataclass(frozen=True)
class SentenceUnit:
    id: str
    text: str
    start_char: int
    end_char: int
    unit_type: str = "sentence"


def split_into_units(text: str) -> list[SentenceUnit]:
    units: list[SentenceUnit] = []
    for index, match in enumerate(_SENTENCE_PATTERN.finditer(text), start=1):
        sentence = match.group().strip()
        if not sentence:
            continue
        leading_whitespace = len(match.group()) - len(match.group().lstrip())
        start_char = match.start() + leading_whitespace
        units.append(
            SentenceUnit(
                id=f"u{index}",
                text=sentence,
                start_char=start_char,
                end_char=start_char + len(sentence),
            )
        )
    return units
