"""
error_analysis.py
Branch: feature/contextual-embeddings

Extracts misclassified reviews from TF-IDF and sentence embedding models,
compares them, and saves results to analysis/errors.csv.

Usage:
    python error_analysis.py

Output:
    analysis/errors_tfidf.csv       — all TF-IDF misclassifications
    analysis/errors_embeddings.csv  — all embedding misclassifications
    analysis/errors_comparison.csv  — side-by-side: which model got it wrong
"""

import csv
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

DATASET_PATH = os.path.join("embeddings", "sentiment_dataset_enriched.csv")
OUTPUT_DIR   = "analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_dataset():
    rows   = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))
    texts  = [r["text"]  for r in rows]
    labels = [r["label"] for r in rows]
    print(f"loaded {len(texts)} samples from {DATASET_PATH}")
    return texts, labels


def collect_errors_tfidf(texts, labels, cv):
    errors = []
    for train_idx, test_idx in cv.split(texts, labels):
        X_train = [texts[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test  = [texts[i] for i in test_idx]
        y_test  = [labels[i] for i in test_idx]

        model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        for i, (true, pred) in enumerate(zip(y_test, preds)):
            if true != pred:
                errors.append({"text": X_test[i], "true_label": true, "predicted": pred})
    return errors


def collect_errors_embeddings(texts, labels, cv):
    from sentence_transformers import SentenceTransformer
    print("loading all-MiniLM-L6-v2 ...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    X_emb    = st_model.encode(texts, show_progress_bar=False)

    errors   = []
    correct  = []
    for train_idx, test_idx in cv.split(texts, labels):
        X_train = X_emb[train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test  = X_emb[test_idx]
        y_test  = [labels[i] for i in test_idx]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        for i, (true, pred) in enumerate(zip(y_test, preds)):
            entry = {"text": texts[test_idx[i]], "true_label": true, "predicted": pred}
            if true != pred:
                errors.append(entry)
            else:
                correct.append(entry)
    return errors, correct


def save_csv(rows, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows → {path}")


def main():
    texts, labels = load_dataset()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── TF-IDF errors ─────────────────────────────────────────────────────
    print("\nrunning TF-IDF error collection ...")
    tfidf_errors = collect_errors_tfidf(texts, labels, cv)
    save_csv(tfidf_errors, os.path.join(OUTPUT_DIR, "errors_tfidf.csv"),
             ["text", "true_label", "predicted"])

    tfidf_error_texts = {e["text"]: e for e in tfidf_errors}

    # ── Embedding errors ───────────────────────────────────────────────────
    print("\nrunning embedding error collection ...")
    emb_errors, emb_correct = collect_errors_embeddings(texts, labels, cv)
    save_csv(emb_errors, os.path.join(OUTPUT_DIR, "errors_embeddings.csv"),
             ["text", "true_label", "predicted"])

    emb_error_texts   = {e["text"] for e in emb_errors}
    emb_correct_texts = {e["text"] for e in emb_correct}
    emb_pred_lookup   = {e["text"]: e["predicted"] for e in emb_errors}

    # ── Comparison ────────────────────────────────────────────────────────
    comparison = []

    # TF-IDF wrong, embeddings right
    for text, err in tfidf_error_texts.items():
        if text in emb_correct_texts:
            comparison.append({
                "text":        text,
                "true_label":  err["true_label"],
                "tfidf_pred":  err["predicted"],
                "emb_pred":    err["true_label"],  # correct
                "winner":      "embeddings",
            })

    # Embeddings wrong, TF-IDF right
    for text in emb_error_texts:
        if text not in tfidf_error_texts:
            comparison.append({
                "text":       text,
                "true_label": next(e["true_label"] for e in emb_errors if e["text"] == text),
                "tfidf_pred": next(e["true_label"] for e in emb_errors if e["text"] == text),
                "emb_pred":   emb_pred_lookup[text],
                "winner":     "tfidf",
            })

    # Both wrong
    for text in emb_error_texts:
        if text in tfidf_error_texts:
            comparison.append({
                "text":       text,
                "true_label": tfidf_error_texts[text]["true_label"],
                "tfidf_pred": tfidf_error_texts[text]["predicted"],
                "emb_pred":   emb_pred_lookup[text],
                "winner":     "neither",
            })

    save_csv(comparison, os.path.join(OUTPUT_DIR, "errors_comparison.csv"),
             ["text", "true_label", "tfidf_pred", "emb_pred", "winner"])

    # ── Summary ────────────────────────────────────────────────────────────
    emb_wins    = sum(1 for r in comparison if r["winner"] == "embeddings")
    tfidf_wins  = sum(1 for r in comparison if r["winner"] == "tfidf")
    both_wrong  = sum(1 for r in comparison if r["winner"] == "neither")

    print(f"\n{'='*55}")
    print("ERROR COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  Total TF-IDF errors:       {len(tfidf_errors)}")
    print(f"  Total embedding errors:    {len(emb_errors)}")
    print(f"  TF-IDF wrong, emb right:   {emb_wins}")
    print(f"  Emb wrong, TF-IDF right:   {tfidf_wins}")
    print(f"  Both wrong:                {both_wrong}")
    print(f"{'='*55}")

    print("\nTF-IDF WRONG, EMBEDDINGS RIGHT (showing first 6):")
    shown = [r for r in comparison if r["winner"] == "embeddings"][:6]
    for r in shown:
        print(f"  true={r['true_label']:9s}  tfidf={r['tfidf_pred']:9s}  | {r['text']}")

    print("\nEMBEDDINGS WRONG, TF-IDF RIGHT (showing first 3):")
    shown2 = [r for r in comparison if r["winner"] == "tfidf"][:3]
    for r in shown2:
        print(f"  true={r['true_label']:9s}  emb={r['emb_pred']:9s}  | {r['text']}")


if __name__ == "__main__":
    main()
