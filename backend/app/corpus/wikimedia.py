"""Bounded, revision-pinned Wikipedia prose acquisition. No model dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, unquote
from urllib.request import Request, HTTPRedirectHandler, build_opener

from app.corpus.models import SourceMetadata
from app.storage.source_corpus import digest

EXTRACTOR_VERSION = "wikimedia-prose-html-v1"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SemaTraceCorpusBot/0.1 (https://github.com/akshat-tumkur/SemaTrace-AI)"
LICENSE_PREFIX = "https://creativecommons.org/licenses/by-sa/4.0/"


class AcquisitionError(RuntimeError):
    pass


class BudgetExceeded(AcquisitionError):
    pass


def validate_entry(entry):
    if (
        type(entry["revision_id"]) is not int
        or type(entry["page_id"]) is not int
        or min(entry["revision_id"], entry["page_id"]) < 1
    ):
        raise AcquisitionError("invalid_revision_identity")
    url = urlparse(entry["url"])
    if (
        url.scheme != "https"
        or url.hostname != "en.wikipedia.org"
        or not url.path.startswith("/wiki/")
        or unquote(url.path[6:]).replace("_", " ") != entry["title"]
        or url.query
        or url.fragment
    ):
        raise AcquisitionError("invalid_canonical_article_url")
    if entry["attribution_url"] != entry["url"] or entry[
        "version_url"
    ] != "https://en.wikipedia.org/w/index.php?oldid=" + str(entry["revision_id"]):
        raise AcquisitionError("invalid_attribution_or_revision_url")
    if not entry["license_url"].startswith(LICENSE_PREFIX):
        raise AcquisitionError("unverified_manifest_license")


class SameHostRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if (
            urlparse(newurl).scheme != "https"
            or urlparse(newurl).hostname != "en.wikipedia.org"
        ):
            raise AcquisitionError("unexpected_redirect_host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WikipediaClient:
    def __init__(
        self,
        *,
        opener=None,
        sleep=time.sleep,
        clock=time.monotonic,
        min_interval=1.0,
        max_response_bytes=2 * 1024 * 1024,
        max_total_bytes=25 * 1024 * 1024,
        max_requests=64,
        retries=2,
    ):
        if (
            min_interval < 1
            or min(max_response_bytes, max_total_bytes, max_requests) < 1
            or not 0 <= retries <= 2
        ):
            raise ValueError("invalid_acquisition_limits")
        self.open = opener or build_opener(SameHostRedirect()).open
        self.sleep, self.clock = sleep, clock
        self.interval = min_interval
        self.last = None
        self.max_response, self.max_total, self.max_requests, self.retries = (
            max_response_bytes,
            max_total_bytes,
            max_requests,
            retries,
        )
        self.bytes_read = self.requests = 0

    def get(self, parameters):
        request = Request(
            API
            + "?"
            + urlencode(
                {
                    "format": "json",
                    "formatversion": 2,
                    "maxlag": 5,
                    "uselang": "en",
                    **parameters,
                }
            ),
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        for attempt in range(self.retries + 1):
            if self.requests >= self.max_requests or self.bytes_read >= self.max_total:
                raise BudgetExceeded("acquisition_budget_exhausted")
            if self.last is not None:
                self.sleep(max(0, self.interval - (self.clock() - self.last)))
            self.last = self.clock()
            self.requests += 1
            retry_after = None
            try:
                with self.open(request, timeout=20) as response:
                    limit = min(self.max_response, self.max_total - self.bytes_read)
                    length = response.headers.get("Content-Length")
                    if length and int(length) >= limit:
                        raise BudgetExceeded("response_size_limit")
                    if (
                        response.headers.get("Content-Encoding", "identity")
                        != "identity"
                    ):
                        raise AcquisitionError("unexpected_content_encoding")
                    raw = response.read(limit)
                    self.bytes_read += len(raw)
                    if len(raw) >= limit:
                        raise BudgetExceeded("response_size_limit")
                    obj = json.loads(raw)
                    retry_after = response.headers.get("Retry-After")
                if obj.get("error", {}).get("code") in {"maxlag", "ratelimited"}:
                    raise URLError("provider_retryable_error")
                if "error" in obj:
                    raise AcquisitionError(
                        "provider_" + str(obj["error"].get("code", "error"))
                    )
                return obj, raw
            except HTTPError as e:
                if e.code not in {408, 429, 500, 502, 503, 504}:
                    raise AcquisitionError("HTTP_" + str(e.code)) from e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                e.close()
            except (URLError, TimeoutError, OSError):
                pass
            if attempt == self.retries:
                raise AcquisitionError("transient_attempts_exhausted")
            wait = 2**attempt
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    try:
                        wait = (
                            parsedate_to_datetime(retry_after)
                            - datetime.now(timezone.utc)
                        ).total_seconds()
                    except (ValueError, TypeError):
                        raise AcquisitionError("unsupported_retry_after")
            if wait > 10:
                raise AcquisitionError("retry_deferred_by_provider")
            self.sleep(max(0, wait))

    def rights(self):
        obj, _ = self.get(dict(action="query", meta="siteinfo", siprop="rightsinfo"))
        rights = obj["query"]["rightsinfo"]
        if not rights.get("url", "").startswith(LICENSE_PREFIX):
            raise AcquisitionError("unverified_site_license")
        return rights

    def resolve(self, titles):
        if not titles or len(titles) > 50:
            raise ValueError("resolve_one_to_fifty_titles")
        rights = self.rights()
        obj, _ = self.get(
            dict(
                action="query",
                prop="info|revisions",
                inprop="url",
                rvprop="ids|timestamp",
                redirects=1,
                titles="|".join(titles),
            )
        )
        entries = []
        failures = []
        for page in obj["query"]["pages"]:
            if page.get("missing") or page.get("ns") != 0 or not page.get("revisions"):
                failures.append(
                    dict(title=page.get("title"), error="missing_or_non_article")
                )
                continue
            rev = page["revisions"][0]
            url = page["canonicalurl"]
            if urlparse(url).hostname != "en.wikipedia.org":
                raise AcquisitionError("invalid_canonical_host")
            entries.append(
                dict(
                    title=page["title"],
                    page_id=page["pageid"],
                    revision_id=rev["revid"],
                    modified_at=rev["timestamp"],
                    url=url,
                    version_url="https://en.wikipedia.org/w/index.php?oldid="
                    + str(rev["revid"]),
                    attribution_url=url,
                    license_url=rights["url"],
                    license_name=rights["text"],
                )
            )
        entries.sort(key=lambda e: (e["title"], e["page_id"]))
        return dict(
            format_version=1,
            collection="wikipedia_en",
            extractor_version=EXTRACTOR_VERSION,
            entries=entries,
            resolve_failures=failures,
            license_basis="Wikimedia site rights and reuse terms; prose only; attribution notices preserved; no non-text licenses inferred.",
            terms_url="https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
        )

    def article(self, entry):
        validate_entry(entry)
        obj, raw = self.get(
            dict(
                action="parse",
                oldid=entry["revision_id"],
                prop="text|revid|displaytitle",
            )
        )
        p = obj["parse"]
        if (
            p["revid"] != entry["revision_id"]
            or p["pageid"] != entry["page_id"]
            or p["title"] != entry["title"]
        ):
            raise AcquisitionError("revision_identity_mismatch")
        html = p["text"] if isinstance(p["text"], str) else p["text"]["*"]
        parser = ProseParser()
        parser.feed(html)
        parser.close()
        text, details = parser.result()
        if entry.get("text_sha256") and digest(text) != entry["text_sha256"]:
            raise AcquisitionError("pinned_text_checksum_mismatch")
        return (
            SourceMetadata(
                collection="wikipedia_en",
                external_id=str(entry["page_id"]),
                title=entry["title"],
                url=entry["url"],
                revision=str(entry["revision_id"]),
                extractor_version=EXTRACTOR_VERSION,
                authors=("Wikipedia contributors",),
                modified_at=entry["modified_at"],
                license_url=entry["license_url"],
                license_name=entry["license_name"],
                rights_status="WIKIMEDIA_DEFAULT_LICENSE",
                attribution_url=entry["attribution_url"],
                version_url=entry["version_url"],
                attribution_text="Wikipedia contributors; extracted prose adapted by whitespace normalization and omission of non-prose blocks. See article history and original notices.",
                extraction_details=details,
            ),
            text,
            raw,
        )


class ProseParser(HTMLParser):
    """Only prose paragraphs; no script execution, assets, tables or math loss."""

    VOID = {
        "br",
        "hr",
        "img",
        "meta",
        "link",
        "input",
        "source",
        "wbr",
        "area",
        "base",
        "embed",
        "param",
        "track",
        "col",
    }
    OMIT_TAGS = {"script", "style", "table", "figure", "nav", "aside", "math"}
    OMIT_CLASSES = {
        "infobox",
        "navbox",
        "toc",
        "reflist",
        "references",
        "noprint",
        "metadata",
        "hatnote",
        "ambox",
        "sidebar",
        "mw-empty-elt",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.current = None
        self.unsupported = False
        self.seen = 0
        self.omitted = 0
        self.excluded = 0
        self.notices = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        inherited = self.stack[-1][1] if self.stack else False
        skip = (
            inherited
            or tag in self.OMIT_TAGS
            or bool(classes & self.OMIT_CLASSES)
            or (tag == "sup" and "reference" in classes)
        )
        if skip and not inherited:
            self.excluded += 1
        if self.current is not None and (
            tag in {"math", "img", "table"} or "mwe-math-element" in classes
        ):
            self.unsupported = True
        if (
            self.current is not None
            and skip
            and not inherited
            and not (tag == "sup" and "reference" in classes)
        ):
            self.unsupported = True
        if tag == "p" and not skip:
            if self.current is not None:
                self.finish()
            self.current = []
            self.unsupported = False
            self.seen += 1
        if tag == "br" and self.current is not None and not skip:
            self.current.append(" ")
        notice = any("attribution" in c or "copyright" in c for c in classes)
        if tag not in self.VOID:
            self.stack.append(
                (tag, skip, notice or bool(self.stack and self.stack[-1][2]))
            )

    def handle_data(self, data):
        if self.stack and self.stack[-1][2]:
            self.notices.append(data)
        if self.current is not None and not (self.stack and self.stack[-1][1]):
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            self.finish()
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def finish(self):
        text = " ".join("".join(self.current).split())
        if self.unsupported:
            self.omitted += 1
        elif text:
            self.parts.append(text)
        self.current = None

    def result(self):
        if self.current is not None:
            self.finish()
        notices = " ".join(" ".join(self.notices).split())
        if re.search(
            r"all rights reserved|non-free text|used with permission", notices, re.I
        ):
            raise AcquisitionError("additional_rights_review_required")
        text = "\n\n".join(self.parts)
        if not text.strip():
            raise AcquisitionError("no_supported_prose")
        return text, dict(
            scope="prose_paragraphs_only",
            extractor_version=EXTRACTOR_VERSION,
            paragraphs_seen=self.seen,
            paragraphs_indexed=len(self.parts),
            unsupported_paragraphs_omitted=self.omitted,
            excluded_blocks=self.excluded,
            attribution_notices=notices,
            normalization="HTML entities decoded; paragraph whitespace collapsed; blank lines separate paragraphs. Offsets refer to this plaintext, not HTML.",
        )
