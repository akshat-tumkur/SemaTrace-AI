"""Local beta browser-origin checks and private-response headers."""

import os
from starlette.responses import JSONResponse


class LocalPrivacy:
    def __init__(self, app):
        self.app = app
        self.origins = {
            s.strip()
            for s in os.getenv("SEMATRACE_CORS_ORIGINS", "http://localhost:5173").split(
                ","
            )
            if s.strip()
        }

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        origin = headers.get(b"origin", b"").decode("latin1")
        if (
            scope["method"] in {"POST", "PUT", "PATCH", "DELETE"}
            and origin
            and origin not in self.origins
        ):
            return await JSONResponse(
                {"detail": "This browser origin is not allowed to change local data."},
                403,
            )(scope, receive, send)

        async def private_send(message):
            if message["type"] == "http.response.start":
                existing = list(message.get("headers", []))
                existing = [
                    (k, v)
                    for k, v in existing
                    if k.lower()
                    not in {
                        b"cache-control",
                        b"x-content-type-options",
                        b"referrer-policy",
                    }
                ]
                message = dict(
                    message,
                    headers=existing
                    + [
                        (b"cache-control", b"no-store"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                    ],
                )
            await send(message)

        await self.app(scope, receive, private_send)
