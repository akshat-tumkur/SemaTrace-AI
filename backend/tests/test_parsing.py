from io import BytesIO

from docx import Document

from app.parsing.document import parse_document
from app.parsing.sentences import split_into_units


def test_text_parser_normalizes_whitespace() -> None:
    parsed = parse_document("paper.txt", b"First line.\n\nSecond   line.")

    assert parsed.file_type == "txt"
    assert parsed.text == "First line.\nSecond line."


def test_sentence_units_preserve_offsets() -> None:
    text = "First claim. A second claim?"

    units = split_into_units(text)

    assert [unit.text for unit in units] == ["First claim.", "A second claim?"]
    assert text[units[0].start_char : units[0].end_char] == units[0].text
    assert text[units[1].start_char : units[1].end_char] == units[1].text


def test_docx_parser_extracts_paragraphs() -> None:
    document = Document()
    document.add_paragraph("A paragraph from DOCX.")
    stream = BytesIO()
    document.save(stream)

    parsed = parse_document("paper.docx", stream.getvalue())

    assert parsed.text == "A paragraph from DOCX."
