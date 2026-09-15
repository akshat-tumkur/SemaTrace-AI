from pathlib import Path

from app.retrieval.corpus import load_local_corpus


def test_corpus_loader_accepts_legacy_text_encoding(tmp_path: Path) -> None:
    (tmp_path / "legacy.txt").write_bytes(
        "Global warming affects climate systems.".encode("cp1252")
    )

    corpus = load_local_corpus(tmp_path)

    assert len(corpus.documents) == 1
    assert "Global warming" in corpus.documents[0].text