"""Vector API setup compatibility and credential-safe failures; no live calls."""

import importlib.util
import json
from pathlib import Path
import sys

from dotenv import dotenv_values
import pytest

from app.cloud.client import CloudUnavailable
from app.cloud.pinecone import API_VERSION, Pinecone


@pytest.fixture
def setup_cli(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/cloud_admin.py"
    spec = importlib.util.spec_from_file_location("pinecone_setup_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = "fixture-private-key-do-not-print"
    env = tmp_path / "private.env"
    env.write_text(
        f"PINECONE_API_KEY={key}\nPINECONE_INDEX_NAME=sematrace-minilm\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PINECONE_API_KEY", key)
    monkeypatch.setenv("PINECONE_INDEX_NAME", "sematrace-minilm")
    monkeypatch.setattr(
        sys,
        "argv",
        ["cloud_admin", "--env-file", str(env), "create-index", "--confirm-starter"],
    )
    return module, env, key


def index_model():
    return dict(
        name="sematrace-minilm",
        dimension=384,
        metric="cosine",
        host="sematrace-minilm-fixture.svc.us-east-1.pinecone.io",
        status={"ready": True, "state": "Ready"},
    )


def test_create_vector_index_uses_compatible_api_and_preserves_private_key(
    setup_cli, monkeypatch, capsys
):
    module, env, key = setup_cli
    calls = []

    def request(method, url, **options):
        calls.append((method, options))
        assert options["headers"]["X-Pinecone-Api-Version"] == "2026-04"
        return (
            (404, b"{}")
            if method == "GET"
            else (201, json.dumps(index_model()).encode())
        )

    monkeypatch.setattr("app.cloud.client.request", request)
    module.main()
    assert [m for m, _ in calls] == ["GET", "POST"]
    assert calls[1][1]["json"] == dict(
        name="sematrace-minilm",
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    )
    saved = dotenv_values(env)
    assert saved["PINECONE_API_KEY"] == key
    assert saved["PINECONE_INDEX_HOST"] == index_model()["host"]
    assert key not in capsys.readouterr().out


def test_existing_index_is_reused_without_another_create(setup_cli, monkeypatch):
    module, _, _ = setup_cli

    def request(method, url, **options):
        assert method == "GET"
        return 200, json.dumps(index_model()).encode()

    monkeypatch.setattr("app.cloud.client.request", request)
    module.main()


@pytest.mark.parametrize(
    "status,body",
    [
        (401, {"error": {"code": "UNAUTHENTICATED", "message": "Invalid key {key}"}}),
        (422, "vector_type: unknown field; echoed key {key}"),
    ],
)
def test_failure_explains_http_status_and_redacts_key(
    setup_cli, monkeypatch, status, body
):
    module, env, key = setup_cli
    before = env.read_bytes()
    raw = (
        (json.dumps(body) if isinstance(body, dict) else body)
        .replace("{key}", key)
        .encode()
    )

    def request(method, url, **options):
        if status == 422 and method == "GET":
            return 404, b"{}"
        return status, raw

    monkeypatch.setattr("app.cloud.client.request", request)
    with pytest.raises(CloudUnavailable) as error:
        module.main()
    assert f"HTTP {status}" in str(error.value)
    assert "[REDACTED]" in str(error.value) and key not in str(error.value)
    assert env.read_bytes() == before


def test_runtime_and_setup_share_vector_api_pin(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "fixture-key")
    monkeypatch.setenv("PINECONE_INDEX_HOST", "fixture.svc.pinecone.io")
    assert (
        Pinecone("00000000-0000-0000-0000-000000000001").headers[
            "X-Pinecone-Api-Version"
        ]
        == API_VERSION
        == "2026-04"
    )
