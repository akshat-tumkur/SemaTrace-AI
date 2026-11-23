"""Validate bearer tokens with the configured Supabase Auth project, fail closed."""

from collections import OrderedDict
from contextvars import ContextVar
import hashlib
import base64
import json
from threading import Lock
import time

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from app.cloud import enabled
from app.cloud.client import (
    CloudUnavailable,
    endpoint,
    owner_id,
    request,
    supabase_public_key,
)

principal = ContextVar("sematrace_authenticated_owner", default=None)
_verified = OrderedDict()
_lock = Lock()


def current_owner():
    return principal.get()


def verify_token(token):
    if not token or len(token) > 8192:
        return None
    base = endpoint("SUPABASE_URL", ".supabase.co")
    fingerprint = hashlib.sha256((base + "\0" + token).encode()).hexdigest()
    now = time.monotonic()
    with _lock:
        cached = _verified.get(fingerprint)
        if cached and now < cached[1]:
            return cached[0]
    status, raw = request(
        "GET",
        base + "/auth/v1/user",
        headers={
            "apikey": supabase_public_key(),
            "Authorization": "Bearer " + token,
        },
        max_bytes=128 * 1024,
    )
    if status in {401, 403}:
        return None
    if status != 200:
        raise CloudUnavailable("Authentication service unavailable")
    try:
        user = json.loads(raw)
        if user.get("is_anonymous") or user.get("banned_until"):
            return None
        identifier = owner_id(user["id"])
    except (ValueError, KeyError, TypeError):
        return None
    # Do not cache raw tokens. A short TTL bounds logout/revocation staleness.
    try:
        payload = token.split(".")[1]
        expires = float(
            json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))[
                "exp"
            ]
        )
        ttl = max(0, min(5, expires - time.time()))
    except (ValueError, KeyError, IndexError, TypeError):
        ttl = 0
    with _lock:
        _verified[fingerprint] = (identifier, now + ttl)
        _verified.move_to_end(fingerprint)
        while len(_verified) > 128:
            _verified.popitem(last=False)
    return identifier


class CloudAuthentication:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not enabled():
            return await self.app(scope, receive, send)
        public = {"/api/health", "/api/ready", "/api/product/status"}
        if (
            scope["method"] == "OPTIONS"
            or scope.get("path") in public
            or not scope.get("path", "").startswith("/api/")
        ):
            return await self.app(scope, receive, send)
        authorization = (
            dict(scope["headers"]).get(b"authorization", b"").decode("latin1")
        )
        owner = None
        try:
            if authorization.startswith("Bearer "):
                owner = await run_in_threadpool(verify_token, authorization[7:])
        except CloudUnavailable:
            return await JSONResponse(
                {"detail": "Authentication service temporarily unavailable."}, 503
            )(scope, receive, send)
        if not owner:
            return await JSONResponse(
                {"detail": "Sign in to access your documents."}, 401
            )(scope, receive, send)
        scope["sematrace_owner"] = owner
        token = principal.set(owner)
        memory_guard = False
        try:
            path = scope.get("path", "")
            guarded_read = scope["method"] == "GET" and (
                path.startswith("/api/analysis/")
                or path.startswith("/api/corpus/versions/")
                or path.endswith("/upload")
            )
            guarded_recalculation = scope["method"] == "POST" and path.endswith(
                "/similarity"
            )
            if guarded_read or guarded_recalculation:
                # Report JSON/PDF can be large. Serialize those reads with the
                # existing analysis/indexing slot on a 512 MB single instance.
                from app.limits import analysis_slot

                memory_guard = analysis_slot.acquire(blocking=False)
                if not memory_guard:
                    return await JSONResponse(
                        {
                            "detail": "The service is processing another document. Retry shortly."
                        },
                        429,
                        headers={"Retry-After": "3"},
                    )(scope, receive, send)
            await self.app(scope, receive, send)
        finally:
            if memory_guard:
                from app.limits import analysis_slot

                analysis_slot.release()
            principal.reset(token)
