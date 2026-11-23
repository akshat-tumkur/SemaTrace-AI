"""Bring-your-own MiniLM vectors; no index creation or paid inference calls."""

import json
import math
import os
import time
from urllib.parse import urlsplit

from app.cloud.client import CloudUnavailable, endpoint, owner_id, request, required
from app.retrieval.generations import GenerationStore, IndexSpec
from app.cloud.libraries import GLOBAL_LIBRARY, GLOBAL_NAMESPACE

# This adapter uses the Vectors API, whose control-plane schema changed in
# 2026-07. Keep creation, description and data operations on one supported pin.
API_VERSION = "2026-04"


def provider_error(raw, secret):
    """Keep actionable provider errors while redacting the request credential."""
    try:
        value = json.loads(raw)
        error = value.get("error", value) if isinstance(value, dict) else value
        if isinstance(error, dict):
            message = ": ".join(
                str(error[k]) for k in ("code", "message") if k in error
            )
        else:
            message = str(error)
    except (ValueError, TypeError):
        message = raw.decode("utf-8", errors="replace")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return " ".join(message.split())[:700] or "No provider error details returned"


class Pinecone:
    def __init__(self, owner):
        key = owner_id(owner)
        self.namespace = GLOBAL_NAMESPACE if key == GLOBAL_LIBRARY else "user-" + key
        self.host = endpoint("PINECONE_INDEX_HOST", ".pinecone.io")
        self.space = GenerationStore.key(IndexSpec.local())
        self.headers = {
            "Api-Key": required("PINECONE_API_KEY"),
            "Content-Type": "application/json",
            "X-Pinecone-Api-Version": API_VERSION,
        }

    def call(self, path, payload):
        status, raw = request(
            "POST", self.host + path, headers=self.headers, json=payload
        )
        if status != 200:
            raise CloudUnavailable(
                "Vector service unavailable or free-tier quota reached"
            )
        try:
            return json.loads(raw)
        except ValueError:
            raise CloudUnavailable("Invalid vector service response") from None

    def validate(self):
        name = required("PINECONE_INDEX_NAME")
        if not name.replace("-", "").isalnum():
            raise CloudUnavailable("Invalid index name")
        status, raw = request(
            "GET", "https://api.pinecone.io/indexes/" + name, headers=self.headers
        )
        if status != 200:
            raise CloudUnavailable("Unable to verify configured vector index")
        try:
            data = json.loads(raw)
            host = data["host"].removeprefix("https://").rstrip("/")
            if (
                data["dimension"] != 384
                or data["metric"] != "cosine"
                or data.get("embed")
                or host != urlsplit(self.host).netloc
                or data.get("status", {}).get("ready") is not True
            ):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise CloudUnavailable(
                "Pinecone index must match the configured host, 384 dimensions and cosine metric"
            ) from None
        return data

    def fetch(self, ids):
        if not ids:
            return {}
        if len(ids) > 100:
            raise ValueError("Vector fetch batch exceeds limit")
        status, raw = request(
            "GET",
            self.host + "/vectors/fetch",
            headers=self.headers,
            params=[("namespace", self.namespace)] + [("ids", x) for x in ids],
        )
        if status != 200:
            raise CloudUnavailable("Unable to verify indexed vectors")
        try:
            return json.loads(raw)["vectors"]
        except (ValueError, KeyError, TypeError):
            raise CloudUnavailable("Invalid vector fetch response") from None

    def upsert(self, records):
        if len(records) > 64:
            raise ValueError("Vector upsert batch exceeds limit")
        for r in records:
            if len(r["values"]) != 384 or any(
                not math.isfinite(v) for v in r["values"]
            ):
                raise ValueError("Invalid MiniLM vector")
        self.call("/vectors/upsert", {"namespace": self.namespace, "vectors": records})
        # Pinecone is eventually consistent. Activate a source only after this
        # bounded read-after-write check has observed every record's identity.
        expected = {r["id"]: r["metadata"] for r in records}
        for attempt in range(8):
            found = self.fetch(list(expected))
            if all(found.get(k, {}).get("metadata") == v for k, v in expected.items()):
                return
            if attempt < 7:
                time.sleep(min(1 + attempt, 3))
        raise CloudUnavailable(
            "Vector publication is pending; retry source ingestion later"
        )

    def query(self, vector, versions):
        if not versions:
            return []
        limit = int(os.getenv("SEMATRACE_PINECONE_TOP_K", "256"))
        if not 8 <= limit <= 1000:
            raise CloudUnavailable("Pinecone passage budget must be 8–1000")
        data = self.call(
            "/query",
            dict(
                namespace=self.namespace,
                vector=vector,
                topK=limit,
                includeValues=False,
                includeMetadata=True,
                filter={
                    "$and": [
                        {"space": {"$eq": self.space}},
                        {"version_id": {"$in": list(versions)}},
                    ]
                },
            ),
        )
        matches = data.get("matches")
        if not isinstance(matches, list) or len(matches) > limit:
            raise CloudUnavailable("Invalid vector query response")
        return matches

    def delete_source(self, source_id):
        self.call(
            "/vectors/delete",
            {"namespace": self.namespace, "filter": {"source_id": {"$eq": source_id}}},
        )

    def delete_version(self, source_id, version_id):
        self.call(
            "/vectors/delete",
            {
                "namespace": self.namespace,
                "filter": {
                    "$and": [
                        {"source_id": {"$eq": source_id}},
                        {"version_id": {"$eq": version_id}},
                    ]
                },
            },
        )
