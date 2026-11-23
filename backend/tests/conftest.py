import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""
os.environ["SEMATRACE_WRITE_REQUESTS_PER_MINUTE"] = (
    "0"  # dedicated limiter tests exercise quotas
)
os.environ["SEMATRACE_ALLOWED_HOSTS"] = "localhost,127.0.0.1,testserver,backend"

import pytest


@pytest.fixture(autouse=True)
def isolate_source_repository(monkeypatch, tmp_path):
    # API source writes must never populate the developer's persistent corpus.
    monkeypatch.setenv("SEMATRACE_CORPUS_DB_PATH", str(tmp_path / "sources.sqlite3"))
    monkeypatch.setenv("SEMATRACE_DB_PATH", str(tmp_path / "analyses.sqlite3"))


@pytest.fixture
def provision_local_index():
    """API fixtures must explicitly provision the local index required by V3."""

    def build(corpus):
        from app.retrieval.generations import GenerationStore, IndexSpec

        return GenerationStore(corpus.retrieval_index_path).build(
            corpus, IndexSpec.local(), api_key=""
        )

    return build
