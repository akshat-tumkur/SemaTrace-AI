"""Preserved V2 read-time warnings, isolated from research orchestration."""

import re

_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _detail_conflicts(submission: str, source: str) -> list[str]:
    left = set(_NUMBER_PATTERN.findall(submission))
    right = set(_NUMBER_PATTERN.findall(source))
    conflicts = ["different_numbers"] if left and right and left != right else []
    negations = {"not", "never", "no", "without", "cannot"}
    if bool(set(_TOKEN_PATTERN.findall(submission.lower())) & negations) != bool(
        set(_TOKEN_PATTERN.findall(source.lower())) & negations
    ):
        conflicts.append("different_negation")
    return conflicts
