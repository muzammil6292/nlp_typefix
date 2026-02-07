import os
import sys
from collections import Counter

# Ensure project root is on sys.path so `src` package can be imported when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ngram import tokenize, generate_ngrams, ngram_counts
from src.spell_corrector import SpellCorrector


def load_corpus(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    corpus = load_corpus('data/sample_corpus.txt')
    tokens = tokenize(corpus)
    print('Tokens sample:', tokens[:20])

    # n-grams
    print('\nUnigrams (top 5):')
    uni_counts = ngram_counts(corpus, 1)
    for ng, cnt in uni_counts.most_common(5):
        print(ng, cnt)

    print('\nBigrams (top 5):')
    bi_counts = ngram_counts(corpus, 2)
    for ng, cnt in bi_counts.most_common(5):
        print(ng, cnt)

    # Spell corrector
    wordlist = [w.strip() for w in open('data/words.txt', encoding='utf-8')]
    freqs = Counter(token for token in tokens)
    sc = SpellCorrector(wordlist, freqs)

    for test in ['spel', 'correction', 'n-gram', 'natral']:
        print(f"\nInput: {test}")
        print('Correction:', sc.correction(test))
        print('Closest (by distance):', sc.most_similar_by_distance(test, k=5))


if __name__ == '__main__':
    main()
