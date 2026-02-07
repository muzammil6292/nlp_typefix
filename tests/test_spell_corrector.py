from collections import Counter
from src.spell_corrector import SpellCorrector


def test_correction_and_candidates():
    words = ["spell", "spelling", "correct", "natural", "n-gram"]
    freqs = Counter({"spell": 10, "spelling": 5, "correct": 7, "natural": 3, "n-gram": 1})
    sc = SpellCorrector(words, freqs)

    assert sc.correction("spel") == "spell"
    assert sc.correction("natural") == "natural"

    sims = sc.most_similar_by_distance("spel", k=3)
    assert "spell" in sims


def test_noisy_channel_correction():
    words = ["spell", "spelling", "correct", "natural", "ngram"]
    freqs = Counter({"spell": 10, "spelling": 5, "correct": 7, "natural": 3, "ngram": 1})
    sc = SpellCorrector(words, freqs)

    # 'spel' should prefer 'spell' under noisy-channel scoring
    assert sc.noisy_channel_correction("spel") == "spell"
    # ensure noisy channel returns ranked candidates
    ranked = sc.noisy_channel_candidates("natral", k=3)
    assert ranked[0][0] == "natural"
