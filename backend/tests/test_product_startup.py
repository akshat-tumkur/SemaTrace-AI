"""Native configuration and research packaging need no private model to start."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_product_starts_with_missing_optional_artifact_and_no_provider_keys(tmp_path):
    front = tmp_path / "frontend"
    (front / "assets").mkdir(parents=True)
    (front / "index.html").write_text("<!doctype html><title>Startup fixture</title>")
    environment = dict(
        os.environ,
        SEMATRACE_FRONTEND_DIST=str(front),
        SEMATRACE_DB_PATH=str(tmp_path / "reports.sqlite3"),
        SEMATRACE_CORPUS_DB_PATH=str(tmp_path / "sources.sqlite3"),
        SEMATRACE_REVIEW_MODEL_DIR=str(tmp_path / "absent-private-model"),
        OPENAI_API_KEY="",
        TAVILY_API_KEY="",
    )
    code = """
from fastapi.testclient import TestClient
from product_server import configure
from app.privacy import LocalPrivacy
app=configure(8037)
assert {'http://127.0.0.1:8037','http://localhost:8037'}<=LocalPrivacy(None).origins
with TestClient(app) as client:
    assert client.get('/').status_code==200
    assert client.get('/api/health').status_code==200
    status=client.get('/api/product/status').json()
    assert status['cloud_review_available'] is False
    assert status['web_search_available'] is False
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script", ["build_roberta_notebook.py", "build_practical_roberta_notebook.py"]
)
def test_research_builder_explains_missing_private_inputs_without_training(
    tmp_path, script
):
    root = Path(__file__).resolve().parents[2]
    args = [
        sys.executable,
        str(root / "scripts" / script),
        "--legacy-head",
        str(tmp_path / "absent.npz"),
    ]
    if "practical" in script:
        args += ["--selection", str(tmp_path / "absent.json")]
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "deliberately not distributed" in result.stderr
    assert not list(tmp_path.iterdir())
