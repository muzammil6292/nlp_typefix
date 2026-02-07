import pytest
from src.ngram import tokenize, generate_ngrams, ngram_counts


def test_tokenize():
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_generate_ngrams_basic():
    tokens = ["a", "b", "c", "d"]
    assert generate_ngrams(tokens, 2) == [("a", "b"), ("b", "c"), ("c", "d")]


def test_generate_ngrams_pad():
    tokens = ["a", "b"]
    ngrams = generate_ngrams(tokens, 3, pad=True)
    assert ngrams[0] == ("<s>", "<s>", "a")


def test_ngram_counts():
    text = "a b a"
    counts = ngram_counts(text, 1)
    assert counts[("a",)] == 2
