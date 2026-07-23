

from __future__ import annotations

import json
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = PROJECT_ROOT / "finetune"
TRAIN_FILE = PROJECT_ROOT / "train.csv"
TEST_FILE = (
    PROJECT_ROOT / "data" / "test_frozen.csv"
    if (PROJECT_ROOT / "data" / "test_frozen.csv").exists()
    else PROJECT_ROOT / "test_frozen.csv"
)
DISTILBERT_DIR = FINETUNE_DIR / "best_checkpoint"
DOCUMENTED_RESULTS_FILE = FINETUNE_DIR / "results.md"

RESULTS_CSV = FINETUNE_DIR / "final_comparison_results.csv"
RESULTS_JSON = FINETUNE_DIR / "final_comparison_results.json"
NEGATION_CSV = FINETUNE_DIR / "negation_comparison.csv"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

NEGATION_TESTS = [
    "not good",
    "not bad",
    "not bad at all",
    "not terrible",
    "not amazing",
    "The service was not good.",
    "The product was not bad.",
    "I do not like this.",
    "I am not disappointed.",
    "This is not the best experience.",
]


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    frame = pd.read_csv(file_path)
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{file_path.name} is missing columns: {sorted(missing)}")

    frame = frame[["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[frame["text"] != ""].reset_index(drop=True)

    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in {file_path.name}: {unknown}")

    return frame


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    recalls = recall_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "negative_recall": float(recalls[0]),
        "neutral_recall": float(recalls[1]),
        "positive_recall": float(recalls[2]),
    }


def average_latency_ms(predict_one: Callable[[str], str], texts: list[str]) -> float:
    samples = texts[: min(30, len(texts))]
    if not samples:
        raise ValueError("The frozen test set is empty.")

    for text in samples[:5]:
        predict_one(text)

    timings = []
    for text in samples:
        start = time.perf_counter()
        predict_one(text)
        timings.append((time.perf_counter() - start) * 1000)

    return float(np.mean(timings))


def directory_size_mb(directory: Path) -> float:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) / (1024**2)


def file_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024**2)


def evaluate_tfidf(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    print("\n[1/3] Training and evaluating TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    train_features = vectorizer.fit_transform(train["text"])
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(train_features, train["label"])

    predictions = model.predict(vectorizer.transform(test["text"])).tolist()
    metrics = classification_metrics(test["label"].tolist(), predictions)

    def predict_one(text: str) -> str:
        return str(model.predict(vectorizer.transform([text]))[0])

    metrics["average_latency_ms"] = average_latency_ms(predict_one, test["text"].tolist())

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact = Path(temp_dir) / "tfidf_logistic_regression.pkl"
        with artifact.open("wb") as file:
            pickle.dump({"vectorizer": vectorizer, "classifier": model}, file)
        metrics["model_size_mb"] = file_size_mb(artifact)

    metrics["evaluation_source"] = "live_same_frozen_test"
    negation_predictions = [predict_one(text) for text in NEGATION_TESTS]
    return metrics, negation_predictions


def evaluate_minilm(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    print("\n[2/3] Loading MiniLM, training Logistic Regression, and evaluating...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is missing. Run: pip install -r requirements.txt"
        ) from exc

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    train_embeddings = encoder.encode(
        train["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    test_embeddings = encoder.encode(
        test["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    classifier = LogisticRegression(max_iter=2000, random_state=42)
    classifier.fit(train_embeddings, train["label"])
    predictions = classifier.predict(test_embeddings).tolist()
    metrics = classification_metrics(test["label"].tolist(), predictions)

    def predict_one(text: str) -> str:
        embedding = encoder.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return str(classifier.predict(embedding)[0])

    metrics["average_latency_ms"] = average_latency_ms(predict_one, test["text"].tolist())

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        encoder_dir = temp_path / "minilm_encoder"
        encoder.save(str(encoder_dir))
        classifier_file = temp_path / "minilm_classifier.pkl"
        with classifier_file.open("wb") as file:
            pickle.dump(classifier, file)
        metrics["model_size_mb"] = directory_size_mb(temp_path)

    metrics["evaluation_source"] = "live_same_frozen_test"
    negation_predictions = [predict_one(text) for text in NEGATION_TESTS]
    return metrics, negation_predictions


def synchronize(device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch
        torch.cuda.synchronize()


def evaluate_distilbert_live(test: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    print("\n[3/3] Loading and evaluating fine-tuned DistilBERT...")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers or torch is missing. Run: pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict_batch(texts: list[str]) -> list[str]:
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        ids = torch.argmax(logits, dim=-1).cpu().tolist()
        return [ID_TO_LABEL[int(index)] for index in ids]

    predictions: list[str] = []
    texts = test["text"].tolist()
    for start in range(0, len(texts), 16):
        predictions.extend(predict_batch(texts[start : start + 16]))

    metrics = classification_metrics(test["label"].tolist(), predictions)

    def predict_one(text: str) -> str:
        encoded = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        synchronize(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        synchronize(device)
        return ID_TO_LABEL[int(torch.argmax(logits, dim=-1).item())]

    metrics["average_latency_ms"] = average_latency_ms(predict_one, texts)
    metrics["model_size_mb"] = directory_size_mb(DISTILBERT_DIR)
    metrics["evaluation_source"] = "live_same_frozen_test"
    metrics["device"] = str(device)
    negation_predictions = [predict_one(text) for text in NEGATION_TESTS]
    return metrics, negation_predictions


def documented_distilbert_results() -> tuple[dict[str, Any], list[str]]:
    """Use existing documented DistilBERT results if its checkpoint is absent."""
    if not DOCUMENTED_RESULTS_FILE.exists():
        raise FileNotFoundError(
            "DistilBERT checkpoint and finetune/results.md are both missing. "
            "Run: python finetune/train_distilbert.py"
        )

    text = DOCUMENTED_RESULTS_FILE.read_text(encoding="utf-8")

    def extract(pattern: str, name: str) -> float:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not read {name} from {DOCUMENTED_RESULTS_FILE}")
        return float(match.group(1))

    metrics = {
        "accuracy": extract(r"\|\s*Accuracy\s*\|\s*([0-9.]+)", "accuracy"),
        "macro_f1": extract(r"\|\s*Macro-F1\s*\|\s*([0-9.]+)", "macro-F1"),
        "average_latency_ms": extract(
            r"\|\s*Average Inference Latency\s*\|\s*([0-9.]+)", "latency"
        ),
        "model_size_mb": extract(r"\|\s*Model Size\s*\|\s*([0-9.]+)", "model size"),
        "negative_recall": extract(r"Negative\s+[0-9.]+\s+([0-9.]+)", "negative recall"),
        "neutral_recall": extract(r"Neutral\s+[0-9.]+\s+([0-9.]+)", "neutral recall"),
        "positive_recall": extract(r"Positive\s+[0-9.]+\s+([0-9.]+)", "positive recall"),
        "evaluation_source": "documented_existing_frozen_test_results",
    }

    negation_predictions = []
    for sentence in NEGATION_TESTS:
        pattern = rf"\|\s*{re.escape(sentence)}\s*\|\s*(Negative|Neutral|Positive)\s*\|"
        match = re.search(pattern, text)
        negation_predictions.append(match.group(1) if match else "Not documented")

    print(
        "\n[3/3] DistilBERT checkpoint not found. "
        "Using the existing frozen-test results in finetune/results.md."
    )
    return metrics, negation_predictions


def rounded_row(model_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model_name,
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "negative_recall": round(float(metrics["negative_recall"]), 4),
        "neutral_recall": round(float(metrics["neutral_recall"]), 4),
        "positive_recall": round(float(metrics["positive_recall"]), 4),
        "model_size_mb": round(float(metrics["model_size_mb"]), 2),
        "average_latency_ms": round(float(metrics["average_latency_ms"]), 2),
        "evaluation_source": metrics["evaluation_source"],
    }


def main() -> None:
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    train = load_data(TRAIN_FILE)
    test = load_data(TEST_FILE)

    print("=" * 72)
    print("FINAL MODEL COMPARISON")
    print(f"Training examples: {len(train)}")
    print(f"Frozen test examples: {len(test)}")
    print(f"Frozen test file: {TEST_FILE}")
    print("=" * 72)

    tfidf_metrics, tfidf_negation = evaluate_tfidf(train, test)
    minilm_metrics, minilm_negation = evaluate_minilm(train, test)

    if DISTILBERT_DIR.exists() and (DISTILBERT_DIR / "config.json").exists():
        distilbert_metrics, distilbert_negation = evaluate_distilbert_live(test)
    else:
        distilbert_metrics, distilbert_negation = documented_distilbert_results()

    rows = [
        rounded_row("TF-IDF + Logistic Regression", tfidf_metrics),
        rounded_row("MiniLM + Logistic Regression", minilm_metrics),
        rounded_row("Fine-tuned DistilBERT", distilbert_metrics),
    ]
    results_frame = pd.DataFrame(rows)
    results_frame.to_csv(RESULTS_CSV, index=False)

    negation_frame = pd.DataFrame(
        {
            "sentence": NEGATION_TESTS,
            "tfidf_prediction": tfidf_negation,
            "minilm_prediction": minilm_negation,
            "distilbert_prediction": distilbert_negation,
        }
    )
    negation_frame.to_csv(NEGATION_CSV, index=False)

    payload = {
        "train_file": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
        "test_file": str(TEST_FILE.relative_to(PROJECT_ROOT)),
        "train_examples": len(train),
        "test_examples": len(test),
        "models": rows,
        "negation_tests": negation_frame.to_dict(orient="records"),
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(results_frame.to_string(index=False))
    print("=" * 72)
    print(f"\nCreated: {RESULTS_CSV}")
    print(f"Created: {RESULTS_JSON}")
    print(f"Created: {NEGATION_CSV}")

    if distilbert_metrics["evaluation_source"].startswith("documented"):
        print(
            "\nNote: TF-IDF and MiniLM were evaluated live now. "
            "DistilBERT used the already documented frozen-test results because "
            "finetune/best_checkpoint was not present."
        )


if __name__ == "__main__":
    main()
