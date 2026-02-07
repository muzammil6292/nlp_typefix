import re
from collections import Counter
from typing import List, Tuple, Iterable

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Simple whitespace + word character tokenizer (lowercases)."""
    return TOKEN_RE.findall(text.lower())


def generate_ngrams(tokens: Iterable[str], n: int, pad: bool = False) -> List[Tuple[str, ...]]:
    """Generate n-grams (list of tuples) from a sequence of tokens."""
    toks = list(tokens)
    if n <= 0:
        raise ValueError("n must be >= 1")
    if pad:
        start = ["<s>"] * (n - 1)
        end = ["</s>"] * (n - 1)
        toks = start + toks + end
    if len(toks) < n:
        return []
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def ngram_counts(text: str, n: int, pad: bool = False) -> Counter:
    tokens = tokenize(text)
    ngrams = generate_ngrams(tokens, n, pad=pad)
    return Counter(ngrams)


def ngram_probabilities(text: str, n: int, pad: bool = False) -> dict:
    counts = ngram_counts(text, n, pad)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {ng: cnt / total for ng, cnt in counts.items()}
