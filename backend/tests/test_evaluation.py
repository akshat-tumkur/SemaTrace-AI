from app.agents.workflow import run_investigation
from app.retrieval.bm25 import CorpusDocument, LocalCorpus


CASES = [
    {
        "label": "EXACT",
        "source": "The rapid development of artificial intelligence created opportunities in healthcare and education.",
        "submission": "The rapid development of artificial intelligence created opportunities in healthcare and education.",
        "expected": "EXACT_OR_LEXICAL_OVERLAP",
    },
    {
        "label": "COMMON_PHRASE",
        "source": "In conclusion, the results show useful findings.",
        "submission": "In conclusion, the results show useful findings.",
        "expected": "LIKELY_FALSE_POSITIVE",
    },
]


def test_controlled_evaluation_cases() -> None:
    outcomes = []
    for index, case in enumerate(CASES):
        corpus = LocalCorpus([CorpusDocument(f"src-{index}", case["label"], f"local://{index}", case["source"])])
        result = run_investigation("evaluation.txt", "txt", case["submission"], corpus)
        if result["matches"]:
            outcome = result["matches"][0]["match_type"]
        elif result["false_positives"]:
            outcome = "LIKELY_FALSE_POSITIVE"
        else:
            outcome = "NO_MATCH"
        outcomes.append((case["label"], outcome))

    assert outcomes == [("EXACT", "EXACT_OR_LEXICAL_OVERLAP"), ("COMMON_PHRASE", "LIKELY_FALSE_POSITIVE")]
