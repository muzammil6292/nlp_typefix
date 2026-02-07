from typing import Iterable

from .edit_distance import levenshtein


def channel_probability(observed: str, candidate: str, error_rate: float = 0.1) -> float:
    """Simple noisy-channel model: probability of observing `observed` given `candidate`.

    We use a very small model based on Levenshtein distance: assume each edit has
    independent probability `error_rate`, so P(observed|candidate) ~ error_rate^d when d>0,
    and 1.0 when d==0. This is simple but effective for demonstration and can be replaced
    with confusion matrices or learned models later.
    """
    d = levenshtein(observed, candidate)
    if d == 0:
        return 1.0
    # Treat probability as decreasing exponentially with distance
    return error_rate ** d


def top_candidates_by_channel(observed: str, candidates: Iterable[str], error_rate: float = 0.1):
    """Yield candidates sorted by channel probability (descending)."""
    scored = sorted(((c, channel_probability(observed, c, error_rate)) for c in candidates), key=lambda x: -x[1])
    return [c for c, _ in scored]
