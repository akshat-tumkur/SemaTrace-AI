"""Small HTTPS/Postgres clients with explicit timeouts and bounded connections."""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import os
from threading import BoundedSemaphore
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from app.corpus.models import CorpusUnavailable


class CloudUnavailable(CorpusUnavailable):
    pass


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise CloudUnavailable("Missing server configuration: " + name)
    return value


def owner_id(value):
    if not value:
        raise CloudUnavailable("Authenticated account required")
    return str(UUID(str(value)))


def supabase_public_key():
    return os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip() or required(
        "SUPABASE_ANON_KEY"
    )


def supabase_secret_key():
    return os.getenv("SUPABASE_SECRET_KEY", "").strip() or required(
        "SUPABASE_SERVICE_ROLE_KEY"
    )


def endpoint(name, domain):
    raw = required(name).rstrip("/")
    if "://" not in raw:
        raw = "https://" + raw
    u = urlsplit(raw)
    testing = os.getenv("SEMATRACE_ALLOW_LOCAL_CLOUD_ENDPOINTS") == "true"
    local = testing and u.hostname in {"localhost", "127.0.0.1"}
    if u.username or u.password or u.query or u.fragment or u.path not in {"", "/"}:
        raise CloudUnavailable("Invalid endpoint: " + name)
    if not local and (u.scheme != "https" or not (u.hostname or "").endswith(domain)):
        raise CloudUnavailable("HTTPS provider endpoint required: " + name)
    return raw


@lru_cache(maxsize=1)
def http_client():
    return httpx.Client(
        trust_env=False,
        follow_redirects=False,
        timeout=20,
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
    )


connection_slots = BoundedSemaphore(4)


@contextmanager
def connect():
    import psycopg
    from psycopg.rows import dict_row

    url = required("SUPABASE_DB_URL")
    local = os.getenv("SEMATRACE_ALLOW_LOCAL_CLOUD_ENDPOINTS") == "true" and urlsplit(
        url
    ).hostname in {"localhost", "127.0.0.1", "sematrace-cloud-postgres"}
    if not connection_slots.acquire(timeout=10):
        raise CloudUnavailable("Database connection limit reached")
    try:
        with psycopg.connect(
            url,
            connect_timeout=10,
            prepare_threshold=None,
            sslmode="disable" if local else "require",
            row_factory=dict_row,
            options="-c statement_timeout=30000 -c idle_in_transaction_session_timeout=30000",
        ) as c:
            # Transaction poolers may ignore startup options. Set limits inside
            # the transaction that owns our queries, including on Supavisor.
            c.execute(
                "SET LOCAL statement_timeout='30s'; SET LOCAL idle_in_transaction_session_timeout='30s'",
                prepare=False,
            )
            yield c
    except psycopg.Error:
        raise CloudUnavailable("Cloud database temporarily unavailable") from None
    finally:
        connection_slots.release()


def request(method, url, *, headers, max_bytes=12 * 1024 * 1024, **kwargs):
    try:
        with http_client().stream(method, url, headers=headers, **kwargs) as r:
            body = bytearray()
            for part in r.iter_bytes():
                body.extend(part)
                if len(body) > max_bytes:
                    raise CloudUnavailable("Provider response exceeded limit")
            return r.status_code, bytes(body)
    except httpx.HTTPError:
        raise CloudUnavailable("Cloud provider temporarily unavailable") from None


class Blobs:
    def __init__(self, owner):
        self.owner = owner_id(owner)
        self.base = endpoint("SUPABASE_URL", ".supabase.co")
        self.bucket = os.getenv("SUPABASE_STORAGE_BUCKET", "sematrace-private")
        if not self.bucket.replace("-", "").isalnum():
            raise CloudUnavailable("Invalid private storage bucket")
        self.key = supabase_secret_key()

    def headers(self):
        headers = {"apikey": self.key}
        if not self.key.startswith("sb_secret_"):
            headers["Authorization"] = "Bearer " + self.key
        return headers

    def _url(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid blob fingerprint")
        return f"{self.base}/storage/v1/object/{self.bucket}/{self.owner}/blobs/{key}"

    def put(self, raw):
        if len(raw) > 10 * 1024 * 1024:
            raise ValueError("Storage object exceeds upload limit")
        key = hashlib.sha256(raw).hexdigest()
        status, _ = request(
            "POST",
            self._url(key),
            headers={
                **self.headers(),
                "Content-Type": "application/octet-stream",
                "x-upsert": "false",
            },
            content=raw,
        )
        if status not in {200, 201}:
            # Idempotency is accepted only if the existing object is identical.
            if self.get(key) != raw:
                raise CloudUnavailable("Immutable blob upload failed")
        return key

    def get(self, key):
        status, raw = request(
            "GET",
            self._url(key),
            headers=self.headers(),
        )
        if status != 200 or hashlib.sha256(raw).hexdigest() != key:
            raise CloudUnavailable("Cloud source object integrity failure")
        return raw
