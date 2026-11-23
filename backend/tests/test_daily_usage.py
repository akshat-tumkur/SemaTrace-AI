"""Daily usage reads use real, isolated PostgreSQL and mocked provider auth."""

from contextlib import contextmanager
from datetime import timedelta, timezone
import json
import time

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from test_cloud import cloud, TEXT  # noqa: F401
from app.cloud import records
from app.cloud.client import connect
from app.main import app


def headers(cloud, index=0):  # noqa: F811
    return {
        "Authorization": "Bearer " + cloud["services"].token(cloud["owners"][index])
    }


def test_empty_usage_is_read_only_and_disabled_features_are_explicit(
    cloud, monkeypatch  # noqa: F811
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/usage", headers=headers(cloud))
    assert response.status_code == 200
    assert "no-store" in response.headers["Cache-Control"]
    data = response.json()
    assert data["usage"]["analysis"] == dict(
        used=0,
        limit=10,
        remaining=10,
        global_remaining=30,
        available=10,
        enabled=True,
        status="available",
        message="",
    )
    assert data["usage"]["reference"]["available"] == 3
    assert data["usage"]["ai"]["status"] == "disabled"
    assert data["usage"]["ai"]["available"] == 0
    with connect() as c:
        assert c.execute("SELECT count(*) AS n FROM st_usage").fetchone()["n"] == 0


def test_usage_requires_auth_and_never_returns_ids_or_other_accounts_usage(cloud):  # noqa: F811
    a, b = cloud["owners"]
    # Auth middleware handles both identity providers through the same principal.
    cloud["services"].tokens.update({"password-fixture": a, "google-fixture": b})
    records.reserve(a, "analysis", 10, 30)
    with TestClient(app) as client:
        assert client.get("/api/usage").status_code == 401
        assert (
            client.get(
                "/api/usage", headers={"Authorization": "Bearer invalid"}
            ).status_code
            == 401
        )
        first = client.get(
            "/api/usage?user_id=" + b,
            headers={"Authorization": "Bearer password-fixture"},
        ).json()
        second = client.get(
            "/api/usage", headers={"Authorization": "Bearer google-fixture"}
        ).json()
    assert first["usage"]["analysis"]["used"] == 1
    assert second["usage"]["analysis"]["used"] == 0
    assert (
        first["usage"]["analysis"]["global_remaining"]
        == second["usage"]["analysis"]["global_remaining"]
        == 29
    )
    for identifier in [a, b, "__global__", "fixture-service", "fixture-key"]:
        assert identifier not in json.dumps([first, second])


def test_previous_exhausted_day_does_not_reduce_today_and_new_reservations_update_counts(
    cloud,  # noqa: F811
):
    owner = cloud["owners"][0]
    with connect() as c:
        c.execute(
            "INSERT INTO st_usage(subject,day,operation,used) VALUES(%s,CURRENT_DATE-1,'analysis',10),('__global__',CURRENT_DATE-1,'analysis',30)",
            (owner,),
        )
    assert records.daily_usage(owner)["usage"]["analysis"]["used"] == 0
    records.reserve_options(owner)
    today = records.daily_usage(owner)
    assert today["usage"]["analysis"]["used"] == 1
    assert today["usage"]["analysis"]["remaining"] == 9
    with connect() as c:
        assert (
            c.execute(
                "SELECT used FROM st_usage WHERE subject=%s AND day=CURRENT_DATE-1",
                (owner,),
            ).fetchone()["used"]
            == 10
        )


def test_shared_exhaustion_is_clamped_and_enforcement_still_rejects(cloud, monkeypatch):  # noqa: F811
    owner = cloud["owners"][0]
    with connect() as c:
        c.execute(
            "INSERT INTO st_usage(subject,operation,used) VALUES(%s,'analysis',2),('__global__','analysis',30)",
            (owner,),
        )
    row = records.daily_usage(owner)["usage"]["analysis"]
    assert row["remaining"] == 8 and row["global_remaining"] == row["available"] == 0
    assert row["status"] == "global_exhausted"
    with pytest.raises(HTTPException) as error:
        records.reserve_options(owner)
    assert error.value.status_code == 429
    monkeypatch.setenv("SEMATRACE_ANALYSIS_USER_DAILY_LIMIT", "1")
    row = records.daily_usage(owner)["usage"]["analysis"]
    assert (
        row["used"] == 2 and row["remaining"] == 0 and row["status"] == "user_exhausted"
    )


def test_reset_is_next_midnight_in_database_timezone_not_browser_or_python_time(
    cloud, monkeypatch  # noqa: F811
):
    original = records.connect

    @contextmanager
    def in_kolkata():
        with original() as c:
            c.execute("SET LOCAL TIME ZONE 'Asia/Kolkata'")
            yield c

    monkeypatch.setattr(records, "connect", in_kolkata)
    data = records.daily_usage(cloud["owners"][0])
    assert data["timezone"] == "Asia/Kolkata"
    reset = data["reset_at"].astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert (reset.hour, reset.minute, reset.second) == (0, 0, 0)
    assert reset.date() == data["day"] + timedelta(days=1)
    assert 0 < (data["reset_at"] - data["server_time"]).total_seconds() <= 86400


def test_ai_and_upload_availability_uses_same_configured_budgets(cloud, monkeypatch):  # noqa: F811
    owner = cloud["owners"][0]
    monkeypatch.setenv("SEMATRACE_AI_USER_DAILY_LIMIT", "2")
    monkeypatch.setenv("SEMATRACE_AI_GLOBAL_DAILY_LIMIT", "4")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    records.reserve_options(owner, operation="ai")
    row = records.daily_usage(owner)["usage"]["ai"]
    assert (row["used"], row["limit"], row["available"], row["global_remaining"]) == (
        1,
        2,
        1,
        3,
    )
    monkeypatch.delenv("OPENAI_API_KEY")
    assert records.daily_usage(owner)["usage"]["ai"]["status"] == "disabled"
    monkeypatch.setenv("SEMATRACE_SOURCE_UPLOAD_ENABLED", "false")
    assert records.daily_usage(owner)["usage"]["reference"]["status"] == "disabled"


def test_failed_reserved_analysis_and_reference_upload_remain_visible(cloud):  # noqa: F811
    cloud["services"].fail_upsert = True
    h = headers(cloud)
    with TestClient(app) as client:
        failed = client.post(
            "/api/sources",
            headers=h,
            data={"title": "Failure fixture", "url": "https://example.invalid/failed"},
            files={"file": ("reference.txt", TEXT.encode())},
        )
        assert failed.status_code == 503
        started = client.post(
            "/api/analysis-jobs", headers=h, files={"file": ("empty.txt", b"")}
        )
        assert started.status_code == 202
        job_id = started.json()["job_id"]
        for _ in range(100):
            job = client.get("/api/analysis-jobs/" + job_id, headers=h).json()
            if job["status"] == "failed":
                break
            time.sleep(0.02)
        assert job["status"] == "failed"
        data = client.get("/api/usage", headers=h).json()["usage"]
    assert data["analysis"]["used"] == 1 and data["reference"]["used"] == 1
