from app.retrieval.bm25 import CorpusDocument, LocalCorpus


def test_search_ranks_matching_passage_first() -> None:
    corpus = LocalCorpus([
        CorpusDocument("voice", "Voice cloning", "local://voice", "Voice cloning can recreate a speaker voice from seconds of audio."),
        CorpusDocument("finance", "Finance", "local://finance", "Financial systems process transactions and manage risk."),
    ])

    matches = corpus.search("speaker voice seconds audio")

    assert matches[0].source_id == "voice"
    assert matches[0].score > 0


def test_search_returns_no_match_for_unknown_terms() -> None:
    corpus = LocalCorpus([CorpusDocument("one", "One", "local://one", "A short source passage.")])

    assert corpus.search("volcanic geology") == []
