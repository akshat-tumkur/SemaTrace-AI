"""Read the frozen transfer fixture across Git LF/Windows CRLF checkouts."""

import hashlib, json

FROZEN_SHA = "debf0ae5430aed784dca6736a76b3bb197c104a1b1b945b4a86d3758d6343276"


def load_fixture(path):
    raw = path.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if hashlib.sha256(canonical).hexdigest() != FROZEN_SHA:
        raise ValueError("Frozen source-passage fixture changed")
    return json.loads(raw), FROZEN_SHA
