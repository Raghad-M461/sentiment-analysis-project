

import csv
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

DATASET_PATH = os.path.join("embeddings", "sentiment_dataset_enriched.csv")
MODEL_DIR    = os.path.join("app", "model")
ENCODER_NAME = "all-MiniLM-L6-v2"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    rows   = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))
    texts  = [r["text"]  for r in rows]
    labels = [r["label"] for r in rows]
    print(f"loaded {len(texts)} samples from {DATASET_PATH}")

    print(f"loading encoder: {ENCODER_NAME} ...")
    encoder = SentenceTransformer(ENCODER_NAME)
    print("encoding sentences ...")
    X = encoder.encode(texts, show_progress_bar=True)
    print(f"embeddings shape: {X.shape}")

    print("training Logistic Regression on full dataset ...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    print(f"training done. classes: {clf.classes_}")

    clf_path    = os.path.join(MODEL_DIR, "logreg_classifier.joblib")
    labels_path = os.path.join(MODEL_DIR, "label_classes.joblib")

    joblib.dump(clf,          clf_path)
    joblib.dump(clf.classes_, labels_path)

    print(f"\nsaved classifier  → {clf_path}")
    print(f"saved class names → {labels_path}")
    print("\nmodel ready. run:  python predict.py \"your sentence here\"")


if __name__ == "__main__":
    main()
