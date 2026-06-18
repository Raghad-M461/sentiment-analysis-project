"""

What this script does:
  1. Loads GloVe 50d vectors .
  2. For 10 sentiment-relevant words, prints the top-5 most similar words.
  3. Runs a few word-analogy tests
  4.  Builds sentence vectors by averaging word embeddings, then runs 5-fold cross-validation against the TF-IDF baseline.

"""

import argparse
import csv
import os
import sys

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# ── so we can import preprocessing.py from the repo root ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import preprocessing as pp

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_GLOVE = os.path.join(os.path.dirname(__file__), "..", "glove_data", "glove.6B.50d.txt")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "embeddings", "sentiment_dataset_enriched.csv")
DIM = 50   # must match the file you load

# ── words we care about for sentiment ─────────────────────────────────────────
SENTIMENT_WORDS = [
    "good", "bad", "terrible", "excellent",
    "great", "service", "awful", "happy", "poor", "not",
]


# ==============================================================================
# 1.  LOAD GLOVE
# ==============================================================================
def load_glove(path: str) -> KeyedVectors:
    print(f"Loading GloVe from: {path}")
    kv = KeyedVectors.load_word2vec_format(path, no_header=True, binary=False)
    print(f"  Loaded {len(kv):,} word vectors (dim={kv.vector_size})\n")
    return kv


# ==============================================================================
# 2.  SIMILARITY
# ==============================================================================
def print_similarities(kv: KeyedVectors, words: list[str], topn: int = 5) -> None:
    print("=" * 65)
    print("PART A — Top-5 most similar words (cosine similarity)")
    print("=" * 65)
    for word in words:
        if word not in kv:
            print(f"  {word:12s}  [NOT IN VOCAB]")
            continue
        sims = kv.most_similar(word, topn=topn)
        row = "  ".join(f"{w}({s:.3f})" for w, s in sims)
        print(f"  {word:12s}  →  {row}")
    print()


# ==============================================================================
# 3.  ANALOGIES  (a → b  ::  c → ?)
# ==============================================================================
def print_analogies(kv: KeyedVectors) -> None:
    print("=" * 65)
    print("PART B — Word analogies  (a → b  ::  c → ?)")
    print("         method: vector(b) - vector(a) + vector(c)")
    print("=" * 65)

    triples = [
        # (a,      b,        c,        note)
        ("good",  "better", "bad",    "comparative degree"),
        ("good",  "best",   "bad",    "superlative degree — trickier"),
        ("happy", "happiest","sad",   "superlative, emotional words"),
        ("man",   "king",   "woman",  "classic gender-role analogy"),
    ]

    for a, b, c, note in triples:
        results = kv.most_similar(positive=[b, c], negative=[a], topn=3)
        top_word, top_score = results[0]
        others = ", ".join(f"{w}({s:.3f})" for w, s in results[1:])
        status = "✓" if top_word not in (a, b, c) else "?"
        print(f"  {status} '{a}' → '{b}'  ::  '{c}' → '{top_word}' ({top_score:.3f})")
        print(f"    [{note}]  runners-up: {others}")
    print()


# ==============================================================================
# 4.  SPOT-CHECK COSINE SIMILARITIES
# ==============================================================================
def print_cosine_checks(kv: KeyedVectors) -> None:
    print("=" * 65)
    print("PART C — Cosine-similarity spot checks")
    print("=" * 65)
    pairs = [
        ("excellent", "great",    "should be HIGH  — both positive"),
        ("bad",       "terrible", "should be HIGH  — both negative"),
        ("excellent", "terrible", "should be LOW   — opposite poles"),
        ("bad",       "good",     "NOTE: often HIGH because same contexts"),
        ("not",       "good",     "key negation pair — what do we find?"),
        ("not",       "bad",      "key negation pair"),
    ]
    for a, b, comment in pairs:
        sim = kv.similarity(a, b)
        print(f"  cos({a:12s}, {b:12s}) = {sim:.4f}   # {comment}")
    print()


# ==============================================================================
# 5.  SENTENCE-VECTOR BASELINE COMPARISON  
# ==============================================================================
def run_baseline_comparison(kv: KeyedVectors) -> None:
    print("=" * 65)
    print("PART D — Baseline comparison (5-fold CV)")
    print("=" * 65)

    rows = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))
    texts_raw = [r["text"] for r in rows]
    y = [r["label"] for r in rows]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── TF-IDF baseline (with negation marking — same as existing pipeline) ──
    X_tfidf = [pp.preprocess(t) for t in texts_raw]
    tfidf_model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    tfidf_scores = cross_val_score(tfidf_model, X_tfidf, y, cv=cv)

   
    def sent_vec(text: str) -> np.ndarray:
        tokens = pp.preprocess(text, handle_negation=False, return_tokens=True)
        vecs = [kv[tok] for tok in tokens if tok in kv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(DIM)

    X_emb = np.array([sent_vec(t) for t in texts_raw])
    emb_scores = cross_val_score(LogisticRegression(max_iter=1000), X_emb, y, cv=cv)

    oov, total = 0, 0
    for t in texts_raw:
        toks = pp.preprocess(t, handle_negation=False, return_tokens=True)
        total += len(toks)
        oov += sum(1 for tok in toks if tok not in kv)

    # ── Report ───────────────────────────────────────────────────────────────
    t_mean  = round(tfidf_scores.mean() * 100, 1)
    t_std   = round(tfidf_scores.std()  * 100, 1)
    e_mean  = round(emb_scores.mean()   * 100, 1)
    e_std   = round(emb_scores.std()    * 100, 1)
    delta   = round(e_mean - t_mean, 1)

    print(f"  Dataset   : {len(texts_raw)} samples, 3 classes (Pos/Neg/Neutral)")
    print(f"  OOV rate  : {oov}/{total} tokens ({round(100*oov/total,1)}%)")
    print()
    print(f"  {'Method':<28s}  {'Mean Acc':>8s}  {'Std':>6s}  {'Fold scores'}")
    print(f"  {'-'*70}")
    tfidf_fold_str = "  ".join(f"{s*100:.1f}%" for s in tfidf_scores)
    emb_fold_str   = "  ".join(f"{s*100:.1f}%" for s in emb_scores)
    print(f"  {'TF-IDF + LogReg (baseline)':<28s}  {t_mean:>7.1f}%  {t_std:>5.1f}%  {tfidf_fold_str}")
    print(f"  {'GloVe-avg + LogReg':<28s}  {e_mean:>7.1f}%  {e_std:>5.1f}%  {emb_fold_str}")
    print()
    print(f"  Δ accuracy (emb − tfidf): {delta:+.1f}%")
    print()
    if delta > 0:
        print("  Embedding average OUTPERFORMS TF-IDF here. Why?")
        print("  Dense vectors let the classifier generalise across synonyms")
        print("  ('terrible' and 'horrible' sit near each other in vector space),")
        print("  whereas TF-IDF treated them as completely separate sparse features.")
    else:
        print("  TF-IDF matches or beats embedding-average here.")

    # ── Negation check ───────────────────────────────────────────────────────
    print()
    print("  Negation sanity check (model trained on full dataset):")
    tfidf_model.fit(X_tfidf, y)
    from sklearn.linear_model import LogisticRegression as LR
    emb_model_full = LR(max_iter=1000).fit(X_emb, y)

    neg_tests = [
        ("The product is not good",         "Negative"),
        ("The service was not bad at all",   "Positive"),
        ("Not happy with the support",       "Negative"),
        ("Not the worst experience",         "Positive"),   # double negation — hard
    ]
    print(f"  {'Sentence':<38s}  {'Expected':>10s}  {'TF-IDF':>8s}  {'Emb-avg':>8s}")
    print(f"  {'-'*80}")
    for text, expected in neg_tests:
        t_pred  = tfidf_model.predict([pp.preprocess(text)])[0]
        e_pred  = emb_model_full.predict([sent_vec(text)])[0]
        t_mark  = "✓" if t_pred == expected else "✗"
        e_mark  = "✓" if e_pred == expected else "✗"
        print(f"  {text:<38s}  {expected:>10s}  {t_mark} {t_pred:<6s}  {e_mark} {e_pred}")
    print()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", default=DEFAULT_GLOVE,
                        help="Path to glove.6B.50d.txt")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the CV comparison (faster)")
    args = parser.parse_args()

    kv = load_glove(args.glove)

    print_similarities(kv, SENTIMENT_WORDS)
    print_analogies(kv)
    print_cosine_checks(kv)

    if not args.skip_baseline:
        run_baseline_comparison(kv)


if __name__ == "__main__":
    main()
