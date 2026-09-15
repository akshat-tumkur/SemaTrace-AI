from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with", "without",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def query_variants(text: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    content = [token for token in tokens if token not in _STOPWORDS]
    variants = [text.strip()]
    if len(content) >= 6:
        variants.append(" ".join(content[:12]))
        variants.append('"' + " ".join(content[:8]) + '"')
    return list(dict.fromkeys(variant for variant in variants if variant))
