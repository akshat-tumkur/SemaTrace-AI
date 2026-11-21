from app.retrieval.bm25 import CorpusDocument, LocalCorpus


from app.retrieval.embeddings import EmbeddingIndex


def test_dense_index_returns_related_passage() -> None:
    corpus = LocalCorpus(
        [
            CorpusDocument(
                "voice",
                "Voice",
                "local://voice",
                "Voice cloning recreates a speaker from audio.",
            ),
            CorpusDocument(
                "finance",
                "Finance",
                "local://finance",
                "Financial systems manage transaction risk.",
            ),
        ]
    )

    index = EmbeddingIndex(corpus, api_key="")
    matches = index.search("speaker audio cloning")

    assert index.provider == "deterministic-hash-fallback"
    assert matches[0].source_id == "voice"
    assert matches[0].semantic_score > 0
