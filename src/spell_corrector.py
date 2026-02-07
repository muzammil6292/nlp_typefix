from collections import Counter
from typing import Set, Iterable, List, Tuple

from .edit_distance import levenshtein
from .noisy_channel import channel_probability


class SpellCorrector:
    """A simple frequency-based spell corrector using edit-distance candidate generation."""

    def __init__(self, wordlist: Iterable[str], freqs: Counter = None):
        self.WORDS: Set[str] = set(w.lower() for w in wordlist)
        self.freqs = freqs or Counter({w: 1 for w in self.WORDS})

    def known(self, words: Iterable[str]) -> Set[str]:
        return {w for w in words if w in self.WORDS}

    def edits1(self, word: str) -> Set[str]:
        """Return all strings one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + (R[1:] if len(R) else '') for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> Set[str]:
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def candidates(self, word: str):
        word = word.lower()
        if word in self.WORDS:
            return {word}
        c1 = self.known(self.edits1(word))
        if c1:
            return c1
        c2 = self.known(self.edits2(word))
        return c2 or {word}

    def prior_prob(self, w: str) -> float:
        """Return a unigram prior probability P(w) estimated from frequencies with simple smoothing."""
        total = sum(self.freqs.values())
        if total == 0:
            return 1.0 / max(1, len(self.WORDS))
        # add-one smoothing
        return (self.freqs.get(w, 0) + 1) / (total + len(self.WORDS))

    def correction(self, word: str) -> str:
        cand = self.candidates(word)
        # choose highest frequency (prior)
        return max(cand, key=lambda w: self.freqs.get(w, 0))

    def noisy_channel_correction(self, word: str, error_rate: float = 0.1) -> str:
        """Return the best correction under a simple noisy-channel model: max_c P(c) * P(w|c)."""
        word = word.lower()
        cand = self.candidates(word)
        scored = ((c, self.prior_prob(c) * channel_probability(word, c, error_rate)) for c in cand)
        best = max(scored, key=lambda t: t[1])[0]
        return best

    def noisy_channel_candidates(self, word: str, k: int = 5, error_rate: float = 0.1) -> List[Tuple[str, float]]:
        """Return top-k candidate corrections with their score (P(c)*P(w|c)), sorted descending."""
        word = word.lower()
        cand = self.candidates(word)
        scored = sorted(
            ((c, self.prior_prob(c) * channel_probability(word, c, error_rate)) for c in cand),
            key=lambda t: -t[1],
        )
        return scored[:k]

    def most_similar_by_distance(self, word: str, k: int = 5):
        """Return up to `k` words from the wordlist closest by edit distance (ties by frequency)."""
        word = word.lower()
        scored = sorted(
            ((w, levenshtein(word, w), -self.freqs.get(w, 0)) for w in self.WORDS),
            key=lambda t: (t[1], t[2]),
        )
        return [w for w, _, _ in scored[:k]]
