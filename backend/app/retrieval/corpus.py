from __future__ import annotations

from pathlib import Path
import hashlib
import re

from app.parsing.sentences import split_into_units
from app.retrieval.bm25 import CorpusDocument, LocalCorpus

CORPUS_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus"
# Stable source labels use the old boundary ordinal, not a renumbering of later
# records when repaired boundaries join old fragments. Never used to split text.
_LEGACY_BOUNDARY = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)


def _source_documents(
    text: str, identifier: str, title: str, url: str
) -> tuple[list[CorpusDocument], str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    parent_id = f"{identifier}@sha256:{digest}"
    old_spans = [
        (m.start(), m.end())
        for m in _LEGACY_BOUNDARY.finditer(text)
        if m.group().strip()
    ]
    documents = []
    ordinal = 0
    for unit in split_into_units(text):
        while ordinal < len(old_spans) and old_spans[ordinal][1] <= unit.start_char:
            ordinal += 1
        if (
            ordinal == len(old_spans)
            or not old_spans[ordinal][0] <= unit.start_char < old_spans[ordinal][1]
        ):
            raise ValueError("source_boundary_identity")
        documents.append(
            CorpusDocument(
                source_id=f"{identifier}-{ordinal + 1}",
                title=title,
                url=f"{url}#u{ordinal + 1}",
                text=unit.text,
                source_document_id=parent_id,
                source_document_sha256=digest,
                source_start_char=unit.start_char,
                source_end_char=unit.end_char,
            )
        )
    return documents, parent_id


def load_local_corpus(path: Path = CORPUS_PATH) -> LocalCorpus:
    documents: list[CorpusDocument] = []
    snapshots: dict[str, str] = {}
    for source_path in sorted(path.glob("*.txt")):
        text = _read_source_text(source_path)
        parts, parent_id = _source_documents(
            text,
            source_path.stem,
            source_path.stem.replace("-", " ").title(),
            f"local://corpus/{source_path.name}",
        )
        documents.extend(parts)
        snapshots[parent_id] = text
    return LocalCorpus(documents, snapshots)


def _read_source_text(source_path: Path) -> str:
    content = source_path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def add_source_to_corpus(
    corpus: LocalCorpus, source_id: str, title: str, url: str, text: str
) -> LocalCorpus:
    documents = list(corpus.documents)
    parts, parent_id = _source_documents(text, source_id, title, url)
    documents.extend(parts)
    snapshots = dict(corpus.source_documents)
    snapshots[parent_id] = text
    return LocalCorpus(documents, snapshots)
