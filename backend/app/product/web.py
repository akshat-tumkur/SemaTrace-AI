"""Two-query/three-source public-web discovery with pinned public-IP fetches."""

import hashlib, http.client, ipaddress, json, socket, ssl, os
from urllib.parse import urlsplit, urljoin
import httpx
from app.corpus.wikimedia import ProseParser
from app.corpus.models import SourceMetadata
from app.parsing.sentences import split_into_units
from app.retrieval.generations import GenerationStore, IndexSpec


class UnsafeSource(ValueError):
    pass


MAX_BYTES = 2 * 1024 * 1024


def public_target(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise UnsafeSource("Unsupported source URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in (80, 443):
        raise UnsafeSource("Unsupported source port")
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    ips = list(dict.fromkeys(item[4][0] for item in addresses))
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise UnsafeSource("Private or non-public source address refused")
    return parsed, port, ips[0]


def fetch_public(url):
    for _ in range(4):
        parsed, port, ip = public_target(url)
        # Connect to the checked IP directly; preserve original TLS SNI/Host.
        connection = http.client.HTTPConnection(ip, port, timeout=12)
        if parsed.scheme == "https":
            connection.sock = ssl.create_default_context().wrap_socket(
                socket.create_connection((ip, port), timeout=12),
                server_hostname=parsed.hostname,
            )
        try:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            connection.request(
                "GET",
                path,
                headers={
                    "Host": parsed.hostname,
                    "User-Agent": "SemaTraceSourceReview/1.0",
                    "Accept": "text/html,application/pdf;q=0.8",
                    "Accept-Encoding": "identity",
                },
            )
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise UnsafeSource("Source redirect has no location")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise UnsafeSource("Source is not publicly readable")
            if int(response.getheader("Content-Length", "0")) > MAX_BYTES:
                raise UnsafeSource("Source response too large")
            if response.getheader("Content-Encoding", "identity") != "identity":
                raise UnsafeSource("Unsupported compressed source response")
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise UnsafeSource("Source response too large")
            mime = response.getheader("Content-Type", "").lower()
            if "text/html" in mime:
                parser = ProseParser()
                parser.feed(raw.decode("utf-8", errors="replace"))
                text, details = parser.result()
            elif "application/pdf" in mime:
                from app.parsing.document import parse_document

                text = parse_document("public-source.pdf", raw).text
                details = {"scope": "extracted_pdf_text"}
            else:
                raise UnsafeSource("Source has no supported text content")
            details.update(
                captured_characters=min(50000, len(text)),
                truncated=len(text) > 50000,
                offset_basis="captured extracted text",
            )
            return url, text[:50000], raw, details
        finally:
            connection.close()
    raise UnsafeSource("Source redirect limit reached")


def provider_extract(url, key):
    # Some publishers refuse direct automated fetches. Keep provenance explicit
    # when using the search provider's extracted snapshot instead.
    public_target(url)
    with httpx.Client(timeout=25, trust_env=False, follow_redirects=False) as client:
        response = client.post(
            "https://api.tavily.com/extract",
            headers={"Authorization": "Bearer " + key},
            json={"urls": [url], "extract_depth": "basic", "format": "text"},
        )
        response.raise_for_status()
        data = response.json()
    rows = data.get("results", [])
    if not rows:
        raise UnsafeSource("Provider did not capture source text")
    row = rows[0]
    final = row.get("url", url)
    public_target(final)
    text = row.get("raw_content", "")
    if not isinstance(text, str) or len(text.split()) < 30:
        raise UnsafeSource("Provider source text unavailable")
    raw = json.dumps({"url": final, "raw_content": text}, ensure_ascii=False).encode()
    if len(raw) > MAX_BYTES:
        raise UnsafeSource("Provider source response too large")
    return (
        final,
        text[:50000],
        raw,
        {
            "scope": "provider_extracted_text",
            "provider": "Tavily extract",
            "publisher_page_independently_fetched": False,
            "truncated": len(text) > 50000,
            "offset_basis": "captured provider-extracted Unicode text",
        },
    )


def discover(text, store, progress=None):
    result = {
        "status": "complete",
        "queries": [],
        "sources": [],
        "warnings": [],
        "max_search_requests": 2,
        "max_sources": 3,
        "extract_calls": 0,
        "scope": "Public web discovery is bounded and does not search the entire internet.",
    }
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return dict(
            result,
            status="unavailable",
            warnings=[
                "Web search is not configured; indexed sources were still checked."
            ],
        )
    units = [
        u.text
        for u in split_into_units(text, preserve_urls=True)
        if 12 <= len(u.text.split()) <= 70 and "http" not in u.text
    ]
    queries = (
        list(dict.fromkeys([units[i][:250] for i in sorted({0, len(units) // 2})]))
        if units
        else []
    )
    urls = []
    with httpx.Client(timeout=20, trust_env=False) as client:
        for query in queries:
            try:
                response = client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": "Bearer " + key},
                    json={
                        "query": query,
                        "search_depth": "basic",
                        "max_results": 3,
                        "include_answer": False,
                        "include_raw_content": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                result["queries"].append(query)
                for item in data.get("results", []):
                    if item.get("url") and item["url"] not in [r[0] for r in urls]:
                        urls.append(
                            (item["url"], str(item.get("title") or item["url"])[:500])
                        )
            except (httpx.HTTPError, ValueError):
                result["warnings"].append("One public-web search could not complete.")
    for url, title in urls[:6]:
        if len(result["sources"]) >= 3:
            break
        try:
            if progress:
                progress(
                    "Capturing public source evidence", 20 + len(result["sources"]) * 4
                )
            try:
                final, text, raw, details = fetch_public(url)
            except Exception:
                # Recheck public addressing before the optional provider call.
                if result["extract_calls"] >= 3:
                    raise UnsafeSource("Provider extraction budget reached")
                result["extract_calls"] += 1
                final, text, raw, details = provider_extract(url, key)
            if len(text.split()) < 30:
                raise UnsafeSource("Too little source prose")
            metadata = SourceMetadata(
                collection="web_private_snapshot",
                external_id=hashlib.sha256(final.encode()).hexdigest(),
                title=title,
                url=final,
                revision="response-sha256:" + hashlib.sha256(raw).hexdigest(),
                extractor_version="tavily-extract-snapshot-v1"
                if details.get("provider")
                else "public-prose-snapshot-v1",
                rights_status="UNVERIFIED",
                attribution_url=final,
                attribution_text=(
                    "Tavily-extracted source snapshot; publisher page was not independently fetched. "
                    if details.get("provider")
                    else "Directly captured public page. "
                )
                + "Retained privately for source comparison. Publisher rights are unverified; no redistribution clearance is claimed.",
                extraction_details=details,
            )
            saved = store.ingest(metadata, text, raw)
            result["sources"].append(dict(saved, title=title, url=final))
        except Exception:
            # No remote bodies, keys or submission text in errors/logs.
            result["warnings"].append(
                "A candidate page could not be safely captured as source evidence."
            )
    if result["sources"]:
        if progress:
            progress("Indexing captured public sources", 34)
        corpus = store.snapshot()
        generation, _ = GenerationStore(corpus.retrieval_index_path).build(
            corpus, IndexSpec.local(), api_key="", batch_size=32
        )
        result["generation_id"] = generation.id
    else:
        result["status"] = "partial"
        result["warnings"].append(
            "No additional web source was captured; the indexed library was still analyzed."
        )
    return result
