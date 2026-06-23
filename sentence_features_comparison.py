"""
sentence_features_comparison.py
Branch: feature/contextual-embeddings

Compares three feature representations on the same dataset
with the same classifier and the same 5-fold CV split:

  1. TF-IDF (baseline from Week 1)
  2. Averaged GloVe vectors (Week 3)  -- requires --glove path
  3. Sentence embeddings via all-MiniLM-L6-v2 (this week)

Usage:
    # full comparison (needs GloVe file + internet for HF model)
    python sentence_features_comparison.py --glove glove_data/glove.6B.50d.txt

    # sentence embeddings + TF-IDF only (no GloVe file needed)
    python sentence_features_comparison.py --skip-glove
"""

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

# ── dataset ──────────────────────────────────────────────────────────────────
# the script looks for the enriched 210-sample CSV first.
# if it is not there it falls back to the 80-sample hardcoded data.
DATASET_PATH = os.path.join("embeddings", "sentiment_dataset_enriched.csv")

HARDCODED_TEXTS = [
    'A fantastic quality overall.', 'A friendly update overall.',
    'A great website overall.', 'A helpful delivery overall.',
    'A perfect service overall.', 'A reliable service overall.',
    'A wonderful onboarding overall.', 'An impressive product overall.',
    'Honestly the dashboard is great.', 'Honestly the service is impressive.',
    'Honestly the setup process is perfect.', 'I found the delivery really impressive.',
    'I found the experience really outstanding.', 'I found the platform really great.',
    'I found the service really wonderful.', 'I found the tool really impressive.',
    'I found the update really reliable.', 'Such an impressive platform.',
    'Such an impressive support team.', 'Such a reliable service.',
    'The dashboard felt reliable to me.', 'The experience felt helpful to me.',
    'The experience was friendly.', 'The interface felt great to me.',
    'The interface was perfect.', 'The price felt perfect to me.',
    'The price was excellent.', 'The quality felt reliable to me.',
    'The service felt outstanding to me.', 'The service was friendly.',
    'The software was excellent.', 'The update felt wonderful to me.',
    'Their dashboard was reliable.', 'Their delivery was smooth.',
    'Their experience was outstanding.', 'Their onboarding was friendly.',
    'Their team was smooth.', 'Their tool was impressive.',
    'Their update was helpful.', 'Their update was reliable.',
    'A broken onboarding overall.', 'A confusing platform overall.',
    'A disappointing product overall.', 'A frustrating dashboard overall.',
    'A frustrating delivery overall.', 'A frustrating website overall.',
    'A poor quality overall.', 'A poor website overall.',
    'A rude tool overall.', 'A slow delivery overall.',
    'A slow experience overall.', 'An unreliable app overall.',
    'Honestly the customer service is rude.', 'Honestly the dashboard is unreliable.',
    'Honestly the software is frustrating.', 'Honestly the tool is unreliable.',
    'I found the customer service really frustrating.', 'I found the dashboard really rude.',
    'I found the delivery really terrible.', 'I found the onboarding really horrible.',
    'I found the price really broken.', 'I found the price really horrible.',
    'I found the update really broken.', 'Such a disappointing software.',
    'Such a frustrating dashboard.', 'Such a rude dashboard.',
    'Such a rude software.', 'Such a slow quality.',
    'Such a slow website.', 'Such a terrible product.',
    'Such an unreliable tool.', 'The interface felt disappointing to me.',
    'The platform was frustrating.', 'The service felt useless to me.',
    'The support team felt terrible to me.', 'The tool felt frustrating to me.',
    'The website was horrible.', 'Their customer service was broken.',
    'Their experience was poor.', 'Their product was awful.',
]
HARDCODED_LABELS = ["Positive"] * 40 + ["Negative"] * 40


def load_dataset():
    if os.path.exists(DATASET_PATH):
        rows = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))
        texts  = [r["text"]  for r in rows]
        labels = [r["label"] for r in rows]
        print(f"loaded enriched dataset: {len(texts)} samples from {DATASET_PATH}")
    else:
        texts  = HARDCODED_TEXTS
        labels = HARDCODED_LABELS
        print(f"enriched CSV not found — using hardcoded 80-sample dataset")
    return texts, labels


# ── evaluation helper ─────────────────────────────────────────────────────────
def run_cv(X, y, model, cv):
    """Run 5-fold CV and return acc, precision, recall, F1 as percentages."""
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        "acc":  round(results["test_accuracy"].mean()        * 100, 1),
        "prec": round(results["test_precision_macro"].mean() * 100, 1),
        "rec":  round(results["test_recall_macro"].mean()    * 100, 1),
        "f1":   round(results["test_f1_macro"].mean()        * 100, 1),
    }


# ── feature methods ──────────────────────────────────────────────────────────
def tfidf_features(texts):
    """Returns (X, model) for sklearn pipeline."""
    return texts, make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))


def glove_features(texts, glove_path):
    """Average GloVe word vectors per sentence."""
    from gensim.models import KeyedVectors
    print(f"\nloading GloVe from {glove_path} ...")
    kv = KeyedVectors.load_word2vec_format(glove_path, no_header=True, binary=False)
    DIM = kv.vector_size

    def sentence_vec(text):
        tokens = text.lower().split()
        vecs = [kv[t] for t in tokens if t in kv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(DIM)

    X = np.array([sentence_vec(t) for t in texts])
    model = LogisticRegression(max_iter=1000)
    return X, model


def sentence_embedding_features(texts):
    """Encode sentences with all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer
    print("\nloading all-MiniLM-L6-v2 ...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    X = st_model.encode(texts, show_progress_bar=True)
    model = LogisticRegression(max_iter=1000)
    return X, model


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glove", default=None,
                        help="Path to glove.6B.50d.txt (skip GloVe row if omitted)")
    parser.add_argument("--skip-glove", action="store_true",
                        help="Skip the GloVe row (faster, no file needed)")
    args = parser.parse_args()

    texts, labels = load_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    # 1. TF-IDF
    print("\nrunning TF-IDF ...")
    X_tfidf, m_tfidf = tfidf_features(texts)
    rows.append(("TF-IDF (baseline)", run_cv(X_tfidf, labels, m_tfidf, cv)))

    # 2. Averaged GloVe
    if not args.skip_glove and args.glove:
        print("\nrunning averaged GloVe ...")
        X_glove, m_glove = glove_features(texts, args.glove)
        rows.append(("Averaged GloVe (Week 3)", run_cv(X_glove, labels, m_glove, cv)))
    elif not args.skip_glove:
        print("\nskipping GloVe (no --glove path given)")
        print("  to include it: python sentence_features_comparison.py --glove glove_data/glove.6B.50d.txt")

    # 3. Sentence embeddings
    print("\nrunning sentence embeddings ...")
    X_sent, m_sent = sentence_embedding_features(texts)
    rows.append(("Sentence embeddings (MiniLM)", run_cv(X_sent, labels, m_sent, cv)))

    # ── print results table ──────────────────────────────────────────────────
    print("\n")
    print("=" * 72)
    print("COMPARISON TABLE — 5-fold CV, same classifier (Logistic Regression)")
    print("=" * 72)
    print(f"  {'Features':<30s}  {'Accuracy':>9s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}")
    print(f"  {'-'*68}")
    for name, m in rows:
        print(f"  {name:<30s}  {m['acc']:>8.1f}%  {m['prec']:>8.1f}%  {m['rec']:>8.1f}%  {m['f1']:>8.1f}%")
    print("=" * 72)

    # ── interpretation ───────────────────────────────────────────────────────
    if len(rows) >= 2:
        best_name = max(rows, key=lambda r: r[1]["f1"])[0]
        print(f"\nbest F1: {best_name}")
        sent_row = next((r for r in rows if "MiniLM" in r[0]), None)
        tfidf_row = next((r for r in rows if "TF-IDF" in r[0]), None)
        if sent_row and tfidf_row:
            delta = sent_row[1]["f1"] - tfidf_row[1]["f1"]
            if delta > 0:
                print(f"sentence embeddings beat TF-IDF by {delta:+.1f}% F1")
                print("the pretrained contextual representations generalise better than sparse counts")
            elif delta < 0:
                print(f"TF-IDF beats sentence embeddings by {-delta:.1f}% F1")
                print("on this small dataset, sparse exact-match features outperform dense vectors")
                print("this is a valid result — see sentence-features.md for the interpretation")
            else:
                print("both methods tied on F1")


if __name__ == "__main__":
    main()
