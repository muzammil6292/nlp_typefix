from src.edit_distance import levenshtein


def test_levenshtein_basic():
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_empty():
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "") == 0
