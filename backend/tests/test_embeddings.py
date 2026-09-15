from app.retrieval.bm25 import CorpusDocument, LocalCorpus
from app.retrieval.embeddings import EmbeddingIndex
from app.retrieval.merger import merge_candidates


def test_dense_index_returns_related_passage() -> None:
    corpus = LocalCorpus([
        CorpusDocument("voice", "Voice", "local://voice", "Voice cloning recreates a speaker from audio."),
        CorpusDocument("finance", "Finance", "local://finance", "Financial systems manage transaction risk."),
    ])

    index = EmbeddingIndex(corpus, api_key="")
    matches = index.search("speaker audio cloning")

    assert index.provider == "deterministic-hash-fallback"
    assert matches[0].source_id == "voice"
    assert matches[0].semantic_score > 0


def test_merger_combines_signals_by_unit_and_source() -> None:
    merged = merge_candidates(
        [{"unit_id": "u1", "source_id": "s1", "lexical_score": 1.0, "source_text": "source"}],
        [{"unit_id": "u1", "source_id": "s1", "semantic_score": 0.9, "source_text": "source"}],
    )

    assert len(merged) == 1
    assert merged[0]["lexical_score"] == 1.0
    assert merged[0]["semantic_score"] == 0.9
    assert merged[0]["retrieval_score"] > 0
