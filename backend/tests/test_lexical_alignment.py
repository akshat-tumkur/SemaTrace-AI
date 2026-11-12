from app.alignment.lexical import align, AlignmentConfig, AlignmentLimit


import pytest


TEXT = "The experiment measured ocean temperatures at several depths during the winter season and compared these observations with historical measurements from nearby stations."


def test_insertions_numbers_and_unicode_offsets_are_grounded():
    source = "Preface. " + TEXT + " End."
    target = (
        "🙂 Résumé. "
        + TEXT.replace("several depths", "three different depths").replace(
            "winter", "2024 winter"
        )
        + " Novel conclusion."
    )
    found = align(target, source)
    assert found and found[0]["category"] == "MODIFIED_REUSE"
    for row in found:
        assert (
            "Novel conclusion"
            not in target[row["submission_start"] : row["submission_end"]]
        )
        for run in row["matched_runs"]:
            assert (
                target[run["submission_start"] : run["submission_end"]].casefold()
                == source[run["source_start"] : run["source_end"]].casefold()
            )


def test_repeated_source_occurrences_and_disjoint_matches_survive():
    second = "Laboratory staff collected samples from the northern coastal region and stored the material in sealed containers before the subsequent chemical analysis began."
    found = align(
        TEXT + " Completely different bridge. " + second,
        TEXT + " " + ("Unrelated context. " * 60) + TEXT + " " + second,
    )
    assert len(found) >= 2
    assert len({row["source_start"] for row in found}) >= 2


def test_unrelated_and_bounded_resource_failure():
    assert (
        align(
            TEXT,
            "Independent philosophical argument about abstract social organizations and cultural history.",
        )
        == []
    )
    with pytest.raises(AlignmentLimit):
        align(TEXT, TEXT * 5, AlignmentConfig(maximum_windows=0))


def test_exact_category_requires_literal_equality_not_only_casefolded_words():
    assert align(TEXT, TEXT)[0]["category"] == "EXACT_REUSE"
    assert align(TEXT.lower(), TEXT)[0]["category"] == "MODIFIED_REUSE"
