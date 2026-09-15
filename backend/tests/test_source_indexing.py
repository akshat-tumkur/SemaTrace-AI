from app.agents.workflow import run_investigation
from app.retrieval.bm25 import CorpusDocument, LocalCorpus
from app.retrieval.corpus import add_source_to_corpus


def test_indexed_source_detects_exact_copied_sentence() -> None:
    corpus = LocalCorpus([])
    source_text = "The history of artificial intelligence includes many important milestones and discoveries."
    corpus = add_source_to_corpus(corpus, "wiki", "Wikipedia article", "https://en.wikipedia.org/wiki/Artificial_intelligence", source_text)

    result = run_investigation("submission.txt", "txt", source_text, corpus, web_search=None, reasoner=None)

    assert result["strong_matches"] == 1
    assert result["matches"][0]["source_url"].startswith("https://en.wikipedia.org")
