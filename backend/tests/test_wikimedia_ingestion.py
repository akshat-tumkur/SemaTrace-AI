import io


import json


from unittest.mock import Mock


from urllib.error import HTTPError


from urllib.parse import parse_qs, urlparse


import pytest


from app.corpus.wikimedia import (
    AcquisitionError,
    BudgetExceeded,
    ProseParser,
    WikipediaClient,
    EXTRACTOR_VERSION,
    validate_entry,
)


from app.corpus.cli import ingest_manifest, export_lock


from app.storage.source_corpus import SourceCorpusStore, digest


RIGHTS = {
    "query": {
        "rightsinfo": {
            "url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en",
            "text": "Creative Commons Attribution-Share Alike 4.0",
        }
    }
}


HTML = "<div><p>The Cedar reservoir stores winter rainfall in covered tanks for the dry season.</p></div>"


ENTRY = dict(
    title="Reservoir",
    page_id=123,
    revision_id=456,
    modified_at="2026-11-01T00:00:00Z",
    url="https://en.wikipedia.org/wiki/Reservoir",
    version_url="https://en.wikipedia.org/w/index.php?oldid=456",
    attribution_url="https://en.wikipedia.org/wiki/Reservoir",
    license_url="https://creativecommons.org/licenses/by-sa/4.0/deed.en",
    license_name="Creative Commons Attribution-Share Alike 4.0",
)


class Response:
    def __init__(self, value, headers=None):
        self.raw = json.dumps(value).encode() if not isinstance(value, bytes) else value
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.raw[:limit]


def client(values, **kwargs):
    queue = iter(values)
    calls = []
    sleeps = []

    def open_(req, **kw):
        calls.append(req)
        value = next(queue)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, Response) else Response(value)

    c = WikipediaClient(opener=open_, sleep=sleeps.append, clock=lambda: 0, **kwargs)
    c.calls = calls
    c.sleeps = sleeps
    return c


def article(html=HTML, **changes):
    return {
        "parse": {
            **dict(title="Reservoir", pageid=123, revid=456, text=html),
            **changes,
        }
    }


def manifest():
    return dict(
        format_version=1,
        collection="wikipedia_en",
        extractor_version=EXTRACTOR_VERSION,
        entries=[dict(ENTRY)],
    )


def test_unicode_inline_text_and_citations_preserve_qualifiers():
    p = ProseParser()
    p.feed(
        '<p>The <b>gate</b> opens <i>only if</i> the temperature is below 0 °C.<sup class="reference">[7]</sup></p><p>H<sub>2</sub>O stays here.</p>'
    )
    text, details = p.result()
    assert (
        text
        == "The gate opens only if the temperature is below 0 °C.\n\nH2O stays here."
    )
    assert details["paragraphs_indexed"] == 2


@pytest.mark.parametrize(
    "embedded",
    [
        "<math>x=3</math>",
        '<span class="mwe-math-element">formula</span>',
        '<span class="metadata">not</span>',
        '<img src="https://external.invalid/x" />',
    ],
)
def test_unsupported_embedded_content_omits_whole_paragraph(embedded):
    p = ProseParser()
    p.feed(
        f"<p>A claim depends on {embedded} rather than a different quantity.</p><p>A complete second paragraph remains available without omitted information.</p>"
    )
    text, details = p.result()
    assert "A claim" not in text and details["unsupported_paragraphs_omitted"] == 1


def test_navigation_tables_scripts_and_images_are_not_indexed_or_fetched():
    p = ProseParser()
    p.feed(
        '<script>bad()</script><table><tr><td><p>Table data.</p></td></tr></table><div class="navbox"><p>Navigation.</p></div><p>Actual visible prose is preserved.</p>'
    )
    text, details = p.result()
    assert text == "Actual visible prose is preserved."
    assert details["excluded_blocks"] >= 3


def test_attribution_notices_preserved_and_restrictive_notices_block():
    p = ProseParser()
    p.feed(
        '<div class="attribution">Includes public domain text from an identified government publication.</div><p>Actual prose.</p>'
    )
    assert "government publication" in p.result()[1]["attribution_notices"]
    p = ProseParser()
    p.feed(
        '<div class="copyright">Non-free text used with permission.</div><p>Actual prose.</p>'
    )
    with pytest.raises(AcquisitionError):
        p.result()


def test_resolve_pins_real_identity_and_does_not_fetch_content():
    page = dict(
        ENTRY,
        pageid=123,
        ns=0,
        canonicalurl=ENTRY["url"],
        revisions=[dict(revid=456, timestamp=ENTRY["modified_at"])],
    )
    c = client([RIGHTS, {"query": {"pages": [page]}}])
    m = c.resolve(["Reservoir"])
    assert m["entries"][0]["revision_id"] == 456 and len(c.calls) == 2
    assert "SemaTraceCorpusBot" in c.calls[0].headers["User-agent"]
    assert c.sleeps == [1.0]


@pytest.mark.parametrize(
    "change", [dict(pageid=999), dict(revid=999), dict(title="Other")]
)
def test_wrong_revision_or_title_never_becomes_ingested_source(change):
    with pytest.raises(AcquisitionError, match="revision_identity"):
        client([article(**change)]).article(ENTRY)


def test_pinned_plaintext_mismatch_fails_without_repair():
    with pytest.raises(AcquisitionError, match="checksum"):
        client([article()]).article({**ENTRY, "text_sha256": "0" * 64})


@pytest.mark.parametrize(
    "change",
    [
        dict(url="https://example.org/wiki/Reservoir"),
        dict(version_url="https://en.wikipedia.org/w/index.php?oldid=999"),
        dict(attribution_url="https://example.org/source"),
        dict(license_url="https://example.org/free-license"),
        dict(revision_id=True),
    ],
)
def test_manifest_cannot_invent_urls_licenses_or_revision_bindings(change):
    with pytest.raises(AcquisitionError):
        validate_entry({**ENTRY, **change})


def test_licensing_failure_never_downloads_article():
    c = client([{"query": {"rightsinfo": {"url": "https://example.org/uncertain"}}}])
    with pytest.raises(AcquisitionError, match="license"):
        c.resolve(["Reservoir"])
    assert len(c.calls) == 1


def test_retry_after_and_bounded_physical_attempts():
    error = HTTPError(
        "https://en.wikipedia.org/w/api.php",
        429,
        "rate limited",
        {"Retry-After": "2"},
        io.BytesIO(b"unneeded body"),
    )
    c = client([error, RIGHTS])
    assert c.rights()["url"] == ENTRY["license_url"]
    assert c.requests == 2 and 2 in c.sleeps
    c = client([TimeoutError(), TimeoutError(), TimeoutError()])
    with pytest.raises(AcquisitionError, match="attempts_exhausted"):
        c.rights()
    assert c.requests == 3


def test_large_retry_after_is_deferred_not_ignored():
    c = client([HTTPError("url", 429, "wait", {"Retry-After": "120"}, io.BytesIO())])
    with pytest.raises(AcquisitionError, match="deferred"):
        c.rights()
    assert c.requests == 1 and not c.sleeps


def test_byte_and_request_caps_stop_transport():
    c = client([Response(b"x" * 100)], max_response_bytes=40)
    with pytest.raises(BudgetExceeded):
        c.get(dict(action="query"))
    assert c.bytes_read == 40
    c = client([TimeoutError()], max_requests=1)
    with pytest.raises(BudgetExceeded):
        c.rights()
    assert len(c.calls) == 1


def test_ingestion_failure_resume_and_cached_rerun_make_no_unnecessary_download(
    tmp_path,
):
    store = SourceCorpusStore(tmp_path / "db")
    missing = HTTPError("url", 404, "missing", {}, io.BytesIO())
    first = ingest_manifest(store, manifest(), client([RIGHTS, missing]))
    assert first["status"] == "PARTIAL" and store.stats()["metadata_only_sources"] == 1
    resumed = ingest_manifest(
        store, manifest(), client([RIGHTS, article()]), resume=first["run_id"]
    )
    assert resumed["status"] == "COMPLETE" and store.stats()["unique_passages"] == 1
    assert (
        resumed["metrics"]["http_requests"] == 4
        and len(resumed["metrics"]["invocations"]) == 2
    )
    cached_client = client([])
    cached = ingest_manifest(store, manifest(), cached_client)
    assert cached["status"] == "COMPLETE" and cached_client.requests == 0
    exported = export_lock(store, manifest())
    assert (
        exported["indexed_sources"] == 1
        and len(exported["entries"][0]["text_sha256"]) == 64
    )
    assert exported["entries"][0]["content_status"] == "INDEXED_PROSE"


def test_resume_cannot_change_frozen_limits(tmp_path):
    store = SourceCorpusStore(tmp_path / "db")
    first = ingest_manifest(store, manifest(), client([RIGHTS, article()]))
    with pytest.raises(ValueError, match="configuration"):
        ingest_manifest(store, manifest(), client([]), limit=1, resume=first["run_id"])
