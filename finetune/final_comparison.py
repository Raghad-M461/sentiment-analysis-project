from __future__ import annotations


import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
FINETUNE = ROOT / "finetune"
TRAIN_FILE = ROOT / "data" / "train.csv"
if not TRAIN_FILE.exists():
    TRAIN_FILE = ROOT / "train.csv"
TEST_FILE = ROOT / "data" / "test_frozen.csv"
if not TEST_FILE.exists():
    TEST_FILE = ROOT / "test_frozen.csv"
DISTILBERT_RESULTS = FINETUNE / "evaluation_results.json"
LABELS = ["Negative", "Neutral", "Positive"]
NEGATION_TESTS = [
    "not good", "not bad", "not bad at all", "not terrible", "not amazing",
    "The service was not good.", "The product was not bad.",
    "I do not like this.", "I am not disappointed.",
    "This is not the best experience.",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in {path.name}: {unknown}")
    return df


def folder_size_mb(folder: Path) -> float:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1048576


def average_latency_ms(predict_one, texts: list[str]) -> float:
    samples = texts[: min(30, len(texts))]
    for text in samples[:5]:
        predict_one(text)
    times = []
    for text in samples:
        start = time.perf_counter()
        predict_one(text)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


def metric_row(name, y_true, y_pred, size_mb, latency_ms):
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "negative_recall": float(recalls[0]),
        "neutral_recall": float(recalls[1]),
        "positive_recall": float(recalls[2]),
        "model_size_mb": float(size_mb),
        "average_inference_latency_ms": float(latency_ms),
    }


def main():
    FINETUNE.mkdir(parents=True, exist_ok=True)
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)
    x_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    x_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()
    results, negation_rows = [], []

    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    tfidf.fit(x_train, y_train)
    tfidf_pred = tfidf.predict(x_test).tolist()
    with tempfile.TemporaryDirectory() as temp:
        model_file = Path(temp) / "tfidf.joblib"
        joblib.dump(tfidf, model_file)
        tfidf_size = model_file.stat().st_size / 1048576
    tfidf_latency = average_latency_ms(lambda t: tfidf.predict([t])[0], x_test)
    results.append(metric_row("TF-IDF + Logistic Regression", y_test, tfidf_pred, tfidf_size, tfidf_latency))
    for sentence in NEGATION_TESTS:
        negation_rows.append({"sentence": sentence, "model": "TF-IDF + Logistic Regression", "prediction": tfidf.predict([sentence])[0]})

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    train_embeddings = encoder.encode(x_train, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    test_embeddings = encoder.encode(x_test, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    classifier = LogisticRegression(max_iter=2000, random_state=42)
    classifier.fit(train_embeddings, y_train)
    minilm_pred = classifier.predict(test_embeddings).tolist()
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        encoder_dir = temp_path / "encoder"
        classifier_file = temp_path / "classifier.joblib"
        encoder.save(str(encoder_dir))
        joblib.dump(classifier, classifier_file)
        minilm_size = folder_size_mb(encoder_dir) + classifier_file.stat().st_size / 1048576

    def predict_minilm(text):
        emb = encoder.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return classifier.predict(emb)[0]

    minilm_latency = average_latency_ms(predict_minilm, x_test)
    results.append(metric_row("MiniLM embeddings + Logistic Regression", y_test, minilm_pred, minilm_size, minilm_latency))
    for sentence in NEGATION_TESTS:
        negation_rows.append({"sentence": sentence, "model": "MiniLM embeddings + Logistic Regression", "prediction": predict_minilm(sentence)})

    if not DISTILBERT_RESULTS.exists():
        raise FileNotFoundError("Run finetune/evaluate_finetuned.py before this script.")
    distilbert = json.loads(DISTILBERT_RESULTS.read_text(encoding="utf-8"))
    recalls = distilbert["per_class_recall"]
    results.append({
        "model": "Fine-tuned DistilBERT",
        "accuracy": float(distilbert["accuracy"]),
        "macro_f1": float(distilbert["macro_f1"]),
        "negative_recall": float(recalls["Negative"]),
        "neutral_recall": float(recalls["Neutral"]),
        "positive_recall": float(recalls["Positive"]),
        "model_size_mb": float(distilbert["model_size_mb"]),
        "average_inference_latency_ms": float(distilbert["average_inference_latency_ms"]),
    })
    for row in distilbert.get("negation_tests", []):
        negation_rows.append({"sentence": row["sentence"], "model": "Fine-tuned DistilBERT", "prediction": row["prediction"]})

    results_df = pd.DataFrame(results)
    negation_df = pd.DataFrame(negation_rows)
    results_df.to_csv(FINETUNE / "final_comparison_results.csv", index=False)
    negation_df.to_csv(FINETUNE / "negation_comparison.csv", index=False)
    (FINETUNE / "final_comparison_results.json").write_text(
        json.dumps({"models": results, "negation_tests": negation_rows}, indent=2),
        encoding="utf-8",
    )
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
