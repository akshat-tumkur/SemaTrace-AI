from app.parsing.document import normalize_with_map, parse_document


def test_unicode_whitespace_projection_preserves_extracted_occurrence():
    raw = "  Résumé\t  🙂alpha\r\n\r\n  beta   gamma.  "
    normalized, starts, ends = normalize_with_map(raw)
    assert normalized == "Résumé 🙂alpha\nbeta gamma."
    assert len(normalized) == len(starts) == len(ends)
    for i, c in enumerate(normalized):
        original = raw[starts[i] : ends[i]]
        assert original.isspace() if c.isspace() else original == c
    a = normalized.index("🙂")
    b = normalized.index("gamma") + 5
    assert raw[starts[a] : ends[b - 1]] == "🙂alpha\r\n\r\n  beta   gamma"


def test_filename_traversal_is_display_name_only():
    parsed = parse_document(
        "../../folder\\document.txt", b"An authorized test document."
    )
    assert parsed.filename == "document.txt"
    assert parsed.original_text == "An authorized test document."
