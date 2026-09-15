from app.retrieval.queries import query_variants


def test_query_variants_include_short_and_quoted_forms() -> None:
    variants = query_variants("The history of artificial intelligence includes many important milestones and discoveries.")

    assert len(variants) == 3
    assert variants[-1].startswith('"history artificial intelligence includes many important milestones')
