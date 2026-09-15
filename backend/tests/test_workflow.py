from app.agents.workflow import run_investigation
from app.retrieval.bm25 import CorpusDocument, LocalCorpus


def test_workflow_runs_distinct_agents_and_judges_evidence() -> None:
    corpus = LocalCorpus([
        CorpusDocument(
            "voice",
            "Voice cloning",
            "local://voice",
            "Voice cloning systems can recreate a speaker voice using seconds of recorded audio.",
        ),
    ])

    result = run_investigation(
        "submission.txt",
        "txt",
        "Voice cloning systems can recreate a speaker voice using seconds of recorded audio.",
        corpus,
    )

    agents = [event["agent"] for event in result["audit_log"]]
    assert agents == [
        "Workflow",
        "Decomposer Agent",
        "Web Investigator Agent",
        "Source Investigator Agent",
        "Embedding Retriever",
        "Candidate Merger",
        "Evidence Verifier Agent",
        "False Positive Filter Agent",
        "Citation Checker Agent",
        "Forensic Judge Agent",
    ]
    assert result["candidate_count"] == 1
    assert result["strong_matches"] == 1
    assert result["matches"][0]["match_type"] == "EXACT_OR_LEXICAL_OVERLAP"


def test_workflow_downgrades_generic_language() -> None:
    corpus = LocalCorpus([
        CorpusDocument("generic", "Generic", "local://generic", "In conclusion, the results show useful findings."),
    ])

    result = run_investigation("submission.txt", "txt", "In conclusion, the results show useful findings.", corpus)

    assert result["false_positives"] == 1
    assert result["matches"] == []


def test_workflow_does_not_confirm_weak_retrieval_overlap() -> None:
    corpus = LocalCorpus([
        CorpusDocument("voice", "Voice", "local://voice", "These synthetic voice tools increase impersonation scams."),
    ])

    result = run_investigation("submission.txt", "txt", "The policy makes public services more inclusive and representative.", corpus)

    assert result["matches"] == []
