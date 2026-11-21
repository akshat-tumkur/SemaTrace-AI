"""Offline product contracts, malformed uploads and presentation safety."""

from io import BytesIO


from contextlib import closing


import json


from zipfile import ZipFile, ZIP_DEFLATED


from unittest.mock import MagicMock


from docx import Document


from fastapi import HTTPException


from fastapi.testclient import TestClient


import pymupdf as fitz


import pytest


from app.compat.evidence import annotate_evidence


from app.limits import RequestLimits, MAX_BODY_SIZE, analysis_slot


from app.parsing.document import parse_document


from app.storage.database import AnalysisStore


from app.retrieval.bm25 import CorpusDocument, LocalCorpus




@pytest.mark.parametrize(
    "method,verdict",
    [
        ("deterministic", None),
        ("deterministic_fallback", None),
        ("llm", "POSSIBLE_PARAPHRASE"),
    ],
)
def test_neither_lexical_overlap_nor_literal_llm_quote_creates_verified_semantic_truth(
    method, verdict
):
    c = dict(
        submission_text="An index maps documents to terms.",
        source_text="An index maps terms to documents.",
        deterministic_outcome="ACCEPT",
        verification_method=method,
        final_match_status="MATCH",
        llm_verdict=verdict,
        llm_verification=dict(
            verdict=verdict, supporting_spans=[], contradictions=[], confidence="HIGH"
        ),
    )
    annotate_evidence(c)
    assert c["evidence_assessment"]["semantic_support"]["status"] == "UNVERIFIED"
    assert not c["evidence_assessment"]["semantic_support"]["independently_validated"]


def test_partial_support_with_reported_conflicts_remains_unknown_without_changing_retention():
    c = dict(
        submission_text="The gate opens.",
        source_text="The gate opens on Tuesdays.",
        llm_verdict="PARTIAL_OVERLAP",
        final_match_status="PARTIAL_MATCH",
        deterministic_outcome="REVIEW",
        verification_method="llm",
        llm_verification={
            "verdict": "PARTIAL_OVERLAP",
            "confidence": "HIGH",
            "supporting_spans": [],
            "contradictions": ["A condition differs."],
        },
    )
    annotate_evidence(c)
    assert c["evidence_assessment"]["semantic_support"]["status"] == "UNKNOWN"
    assert c["final_match_status"] == "PARTIAL_MATCH"


@pytest.mark.parametrize("kind", ["txt", "pdf", "docx"])
def test_real_parsers_upload_report_and_reopened_storage(
    kind, tmp_path, monkeypatch, provision_local_index
):
    from app import main

    text = "The calibrated laboratory sensor recorded seventeen volts throughout the controlled experiment."
    if kind == "txt":
        raw = text.encode()
    elif kind == "pdf":
        with fitz.open() as doc:
            doc.new_page().insert_text((35, 60), text, fontsize=9)
            raw = doc.tobytes()
    else:
        doc = Document()
        doc.add_paragraph(text)
        buf = BytesIO()
        doc.save(buf)
        raw = buf.getvalue()
    monkeypatch.setattr(
        main,
        "local_corpus",
        LocalCorpus(
            [CorpusDocument("sensor", "Sensor", "https://example.org/source", text)]
        ),
    )
    path = tmp_path / "analyses.sqlite3"
    monkeypatch.setattr(main, "analysis_store", AnalysisStore(path))
    provision_local_index(main.get_source_store().snapshot())
    with TestClient(main.app) as client:
        response = client.post("/api/analyze", files={"file": ("sample." + kind, raw)})
        assert response.status_code == 202, response.text
        result = response.json()
        assert result["document"]["text"] == text
        assert result["units"][0]["start_char"] == 0 and result["units"][0][
            "end_char"
        ] == len(text)
        assert result["matches"][0]["source_text"] == text.rstrip(".")
        assert result["verification"]["metrics"]["llm_calls"] == 0
        monkeypatch.setattr(main, "analysis_store", AnalysisStore(path))
        saved = client.get("/api/analysis/" + result["analysis_id"]).json()
        report = client.get("/api/analysis/" + result["analysis_id"] + "/report").json()
        assert saved["document"] == report["document"] == result["document"]
        assert saved["units"] == report["units"] == result["units"]


@pytest.mark.parametrize(
    "name,raw,status",
    [
        ("invalid.pdf", b"not pdf", 422),
        ("invalid.docx", b"not zip", 422),
        ("script.exe", b"binary", 415),
        ("empty.txt", b" ", 422),
        ("binary.txt", b"hello\x00world", 422),
        ("big.txt", b"x" * 50001, 413),
    ],
    ids=["bad-pdf", "bad-docx", "unsupported", "empty", "binary", "text-limit"],
)
def test_malformed_and_bounded_parser_inputs(name, raw, status):
    with pytest.raises(HTTPException) as error:
        parse_document(name, raw)
    assert error.value.status_code == status


def test_docx_expansion_and_pdf_page_limits():
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as archive:
        archive.writestr("bomb.xml", b"x" * (32 * 1024 * 1024 + 1))
    with pytest.raises(HTTPException) as e:
        parse_document("bomb.docx", buf.getvalue())
    assert e.value.status_code == 413
    with fitz.open() as doc:
        for _ in range(101):
            doc.new_page()
        raw = doc.tobytes()
    with pytest.raises(HTTPException) as e:
        parse_document("pages.pdf", raw)
    assert e.value.status_code == 413


def test_write_rate_limit_and_declared_body_limit():
    from starlette.responses import JSONResponse

    async def inner(scope, receive, send):
        await JSONResponse({"ok": True})(scope, receive, send)

    with closing(TestClient(RequestLimits(inner, requests_per_minute=1))) as client:
        assert client.post("/api/analyze").status_code == 200
        assert client.post("/api/analyze").status_code == 429
        assert (
            client.post(
                "/api/analyze", headers={"Content-Length": str(MAX_BODY_SIZE + 1)}
            ).status_code
            == 413
        )


def test_chunked_oversize_body_cannot_bypass_limit():
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    async def inner(scope, receive, send):
        await Request(scope, receive).body()
        await JSONResponse({"ok": True})(scope, receive, send)

    with closing(TestClient(RequestLimits(inner, requests_per_minute=0))) as client:
        assert (
            client.post(
                "/api/analyze", content=iter([b"x" * (MAX_BODY_SIZE + 1)])
            ).status_code
            == 413
        )


def test_investigation_busy_and_source_upload_control(monkeypatch):
    from app import main

    with TestClient(main.app) as client:
        assert analysis_slot.acquire(blocking=False)
        try:
            assert (
                client.post(
                    "/api/analyze", files={"file": ("one.txt", b"sample")}
                ).status_code
                == 429
            )
        finally:
            analysis_slot.release()
        monkeypatch.setenv("SEMATRACE_SOURCE_UPLOAD_ENABLED", "false")
        assert (
            client.post(
                "/api/sources",
                files={"file": ("one.txt", b"sample")},
                data={"title": "Sample", "url": "https://example.org"},
            ).status_code
            == 403
        )


def test_real_multipart_stream_limit_and_unit_limit():
    from app import main

    with TestClient(main.app) as client:
        header = b'--beta\r\nContent-Disposition: form-data; name="file"; filename="large.txt"\r\nContent-Type: text/plain\r\n\r\n'
        body = iter([header, b"x" * (MAX_BODY_SIZE + 1), b"\r\n--beta--\r\n"])
        response = client.post(
            "/api/analyze",
            content=body,
            headers={"Content-Type": "multipart/form-data; boundary=beta"},
        )
        assert response.status_code == 413
        from app.limits import MAX_ANALYSIS_UNITS

        text = "Measure pressure. " * (MAX_ANALYSIS_UNITS + 1)
        response = client.post(
            "/api/analyze", files={"file": ("units.txt", text.encode())}
        )
        assert response.status_code == 413
        assert (
            "1,001" in response.json()["detail"]
            and "1,000" in response.json()["detail"]
        )
