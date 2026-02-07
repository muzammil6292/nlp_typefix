# N-gram & Edit Distance Mini Project 🔧

A compact Python mini-project demonstrating n-gram extraction and Levenshtein edit distance with a simple spell-corrector.

## Features ✅
- Generate n-grams (unigram, bigram, trigram)
- Count and compute n-gram frequencies
- Compute Levenshtein edit distance
- Simple frequency-based spell corrector using edit-distance candidates
- Optional noisy-channel correction model (P(c)*P(w|c)) for probabilistic ranking
- Unit tests with `pytest`

## Quickstart ⚡
1. Create and activate a virtual environment

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dev requirements

   ```bash
   pip install -r requirements.txt
   ```

3. Run example

   ```bash
   python examples/demo.py
   ```

4. Run tests

   ```bash
   pytest -q
   ```

## Project layout
- `src/` - core modules (`ngram`, `edit_distance`, `spell_corrector`)
- `data/` - sample corpus and word list
- `examples/` - usage demo
- `tests/` - unit tests

---

💡 **Tip:** The code is dependency-light and intended for learning and experimentation.
# nlp_typefix
