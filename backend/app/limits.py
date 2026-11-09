"""Small single-process beta limits, not distributed rate limiting or auth."""

from __future__ import annotations

from collections import OrderedDict
import os
from threading import BoundedSemaphore, Lock
import time

from starlette.responses import JSONResponse

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_BODY_SIZE = MAX_FILE_SIZE + 256 * 1024  # multipart metadata allowance
MAX_TEXT_CHARS = 50_000
MAX_ANALYSIS_UNITS = 1000
analysis_slot = BoundedSemaphore(1)


class BodyTooLarge(Exception):
    pass


class RequestLimits:
    def __init__(self, app, *, requests_per_minute=None):
        self.app = app
        self.limit = (
            int(os.getenv("SEMATRACE_WRITE_REQUESTS_PER_MINUTE", "10"))
            if requests_per_minute is None
            else requests_per_minute
        )
        self.clients = OrderedDict()
        self.lock = Lock()

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        similarity_write = path.startswith("/api/analysis/") and path.endswith(
            "/similarity"
        )
        semantic_write = path.startswith("/api/analysis/") and path.endswith(
            (
                "/semantic/rerank",
                "/semantic/investigate",
                "/ai-review",
                "/cloud-investigate",
            )
        )
        deletion_write = scope.get("method") == "DELETE" and path.startswith(
            "/api/analysis/"
        )
        if (
            scope["type"] != "http"
            or (scope["method"] != "POST" and not deletion_write)
            or (
                scope["path"]
                not in {"/api/analyze", "/api/sources", "/api/analysis-jobs"}
                and not similarity_write
                and not semantic_write
                and not deletion_write
            )
        ):
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid content length."}, 400)(
                scope, receive, send
            )
        if declared > MAX_BODY_SIZE:
            return await JSONResponse(
                {"detail": "Upload exceeds the request limit."}, 413
            )(scope, receive, send)
        # Do not trust forwarded IP headers. A trusted proxy needs its own limit.
        client = (scope.get("client") or ("unknown", 0))[0]
        now = time.monotonic()
        with self.lock:
            start, count = self.clients.pop(client, (now, 0))
            if now - start >= 60:
                start, count = now, 0
            allowed = self.limit <= 0 or count < self.limit
            self.clients[client] = (start, count + 1)
            while len(self.clients) > 4096:
                self.clients.popitem(last=False)
        if not allowed:
            return await JSONResponse(
                {"detail": "Too many uploads; retry after one minute."},
                429,
                headers={"Retry-After": "60"},
            )(scope, receive, send)
        total = 0
        started = False
        overflow = False

        async def bounded_receive():
            nonlocal total, overflow
            message = await receive()
            total += len(message.get("body", b""))
            if total > MAX_BODY_SIZE:
                overflow = True
                raise BodyTooLarge()
            return message

        async def observed_send(message):
            nonlocal started
            # FastAPI may turn a receive exception into a generic parse error.
            # Preserve 413 even in that path; never send two response starts.
            if overflow:
                if not started:
                    started = True
                    await JSONResponse(
                        {"detail": "Upload exceeds the request limit."}, 413
                    )(scope, receive, send)
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, observed_send)
        except BodyTooLarge:
            if started:
                raise
            await JSONResponse({"detail": "Upload exceeds the request limit."}, 413)(
                scope, receive, send
            )
