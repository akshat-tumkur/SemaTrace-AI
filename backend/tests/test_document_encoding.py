from app.parsing.document import parse_document


def test_uploaded_legacy_text_encoding_is_supported() -> None:
    parsed = parse_document("legacy.txt", "Global warming affects climate systems.".encode("cp1252"))

    assert "Global warming" in parsed.text