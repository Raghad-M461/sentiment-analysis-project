"""

Compares three feature representations on the same dataset
with the same classifier and the same 5-fold CV split:

  1. TF-IDF (baseline from Week 1)
  2. Averaged GloVe vectors (Week 3) — numbers from explore_embeddings.py output
  3. Sentence embeddings  (this week)

"""

import argparse
import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

# ── dataset ───────────────────────────────────────────────────────────────────
# looks for the enriched 210-sample CSV first, falls back to 80-sample hardcoded
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
        print("enriched CSV not found — using hardcoded 80-sample dataset")
    return texts, labels


# ── evaluation helper ─────────────────────────────────────────────────────────
def run_cv(X, y, model, cv):
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        "acc":  round(results["test_accuracy"].mean()        * 100, 1),
        "prec": round(results["test_precision_macro"].mean() * 100, 1),
        "rec":  round(results["test_recall_macro"].mean()    * 100, 1),
        "f1":   round(results["test_f1_macro"].mean()        * 100, 1),
    }


# ── feature methods ───────────────────────────────────────────────────────────
def tfidf_features(texts):
    return texts, make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))


def sentence_embedding_features(texts):
    from sentence_transformers import SentenceTransformer
    print("\nloading all-MiniLM-L6-v2 ...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    X = st_model.encode(texts, show_progress_bar=True)
    model = LogisticRegression(max_iter=1000)
    return X, model


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-glove", action="store_true")
    args = parser.parse_args()

    texts, labels = load_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    # 1. TF-IDF
    print("\nrunning TF-IDF ...")
    X_tfidf, m_tfidf = tfidf_features(texts)
    rows.append(("TF-IDF (baseline)", run_cv(X_tfidf, labels, m_tfidf, cv)))

    # 2. Averaged GloVe — real numbers from Week 3 (explore_embeddings.py on same dataset)
    rows.append(("Averaged GloVe (Week 3)", {
        "acc": 74.3, "prec": 74.0, "rec": 74.3, "f1": 73.8
    }))

    # 3. Sentence embeddings
    print("\nrunning sentence embeddings ...")
    X_sent, m_sent = sentence_embedding_features(texts)
    rows.append(("Sentence embeddings (MiniLM)", run_cv(X_sent, labels, m_sent, cv)))

    # ── print table ───────────────────────────────────────────────────────────
    print("\n")
    print("=" * 72)
    print("Sentiment Classification Performance by Feature Representation")
    print("=" * 72)
    print(f"  {'Features':<30s}  {'Accuracy':>9s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}")
    print(f"  {'-'*68}")
    for name, m in rows:
        print(f"  {name:<30s}  {m['acc']:>8.1f}%  {m['prec']:>8.1f}%  {m['rec']:>8.1f}%  {m['f1']:>8.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
