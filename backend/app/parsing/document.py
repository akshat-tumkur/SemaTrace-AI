from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import re

from docx import Document
from fastapi import HTTPException
from fitz import open as open_pdf
from app.limits import MAX_TEXT_CHARS


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    text: str
    file_type: str
    original_text: str = ""
    normalized_starts: tuple[int, ...] = ()
    normalized_ends: tuple[int, ...] = ()


def parse_document(filename: str, content: bytes) -> ParsedDocument:
    filename = (
        "".join(
            c for c in filename.replace("\\", "/").rsplit("/", 1)[-1] if ord(c) >= 32
        )[-255:]
        or "document.txt"
    )
    extension = Path(filename).suffix.lower()
    if extension == ".txt":
        text = _decode_text_file(content)
    elif extension == ".pdf":
        text = _parse_pdf(content)
    elif extension == ".docx":
        text = _parse_docx(content)
    else:
        raise HTTPException(
            status_code=415, detail="Supported files are .txt, .pdf, and .docx."
        )

    if len(text) > MAX_TEXT_CHARS * 4:
        raise HTTPException(
            413, "Original extracted text exceeds the processing limit."
        )
    normalized, starts, ends = normalize_with_map(text)
    if len(normalized) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413, detail="Extracted text exceeds the 50,000 character limit."
        )
    if any(ord(c) < 32 and c not in "\n\r\t" for c in normalized):
        raise HTTPException(
            status_code=422,
            detail="The document contains unsupported binary/control data.",
        )
    if not normalized:
        raise HTTPException(status_code=422, detail="The uploaded document is empty.")
    return ParsedDocument(
        filename=filename,
        text=normalized,
        file_type=extension[1:],
        original_text=text,
        normalized_starts=tuple(starts),
        normalized_ends=tuple(ends),
    )


def _parse_pdf(content: bytes) -> str:
    try:
        with open_pdf(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise HTTPException(422, "Password-protected PDFs are not supported.")
            if len(document) > 100:
                raise HTTPException(413, "PDFs are limited to 100 pages.")
            pages = []
            total = 0
            for page in document:
                text = page.get_text()
                total += len(text)
                if total > MAX_TEXT_CHARS * 2:
                    raise HTTPException(
                        413, "PDF extracted text exceeds the processing limit."
                    )
                pages.append(text)
            return "\n".join(pages)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=422, detail="The PDF could not be parsed."
        ) from error


def _parse_docx(content: bytes) -> str:
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            if (
                len(entries) > 512
                or sum(i.file_size for i in entries) > 32 * 1024 * 1024
            ):
                raise HTTPException(
                    413, "DOCX package exceeds the expanded-size or entry limit."
                )
        document = Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=422, detail="The DOCX could not be parsed."
        ) from error


def _normalize_text(text: str) -> str:
    return normalize_with_map(text)[0]


def normalize_with_map(text):
    """Each normalized character maps to a retained original extracted interval."""
    output = []
    starts = []
    ends = []
    offset = 0
    for line in text.splitlines(keepends=True):
        pieces = list(re.finditer(r"\S+", line))
        for i, match in enumerate(pieces):
            a, b = offset + match.start(), offset + match.end()
            if output:
                output.append(" " if i else "\n")
                starts.append(ends[-1])
                ends.append(a)
            output.extend(text[a:b])
            starts.extend(range(a, b))
            ends.extend(range(a + 1, b + 1))
        offset += len(line)
    return "".join(output), starts, ends


def _decode_text_file(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    try:
        return content.decode("cp1252", errors="replace")
    except (LookupError, UnicodeError) as error:
        raise HTTPException(
            status_code=415, detail="The text file encoding is not supported."
        ) from error
