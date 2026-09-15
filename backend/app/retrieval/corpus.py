from __future__ import annotations

from pathlib import Path

from app.parsing.sentences import split_into_units
from app.retrieval.bm25 import CorpusDocument, LocalCorpus

CORPUS_PATH = Path(__file__).resolve().parents[3] / "data" / "corpus"


def load_local_corpus(path: Path = CORPUS_PATH) -> LocalCorpus:
    documents: list[CorpusDocument] = []
    for source_path in sorted(path.glob("*.txt")):
        text = source_path.read_text(encoding="utf-8").strip()
        for index, unit in enumerate(split_into_units(text), start=1):
            documents.append(
                CorpusDocument(
                    source_id=f"{source_path.stem}-{index}",
                    title=source_path.stem.replace("-", " ").title(),
                    url=f"local://corpus/{source_path.name}#u{index}",
                    text=unit.text,
                )
            )
    return LocalCorpus(documents)


def add_source_to_corpus(corpus: LocalCorpus, source_id: str, title: str, url: str, text: str) -> LocalCorpus:
    documents = list(corpus.documents)
    for index, unit in enumerate(split_into_units(text), start=1):
        documents.append(
            CorpusDocument(
                source_id=f"{source_id}-{index}",
                title=title,
                url=f"{url}#u{index}",
                text=unit.text,
            )
        )
    return LocalCorpus(documents)
