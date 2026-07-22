

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = PROJECT_ROOT / "finetune"
MODEL_DIR = FINETUNE_DIR / "best_checkpoint"

TEST_FILE = (
    PROJECT_ROOT / "data" / "test_frozen.csv"
    if (PROJECT_ROOT / "data" / "test_frozen.csv").exists()
    else PROJECT_ROOT / "test_frozen.csv"
)

REPORT_FILE = FINETUNE_DIR / "classification_report.txt"
CONFUSION_MATRIX_FILE = FINETUNE_DIR / "confusion_matrix.png"
JSON_RESULTS_FILE = FINETUNE_DIR / "evaluation_results.json"
MARKDOWN_RESULTS_FILE = FINETUNE_DIR / "results.md"

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


def validate_inputs() -> None:
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            "Frozen test set not found. Expected data/test_frozen.csv "
            "or test_frozen.csv."
        )

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {MODEL_DIR}\n"
            "Run python finetune/train_distilbert.py first."
        )

    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"The saved model is incomplete: {MODEL_DIR}/config.json is missing."
        )


def load_test_data() -> pd.DataFrame:
    dataframe = pd.read_csv(TEST_FILE)

    required_columns = {"text", "label"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{TEST_FILE.name} is missing columns: {sorted(missing)}"
        )

    dataframe = dataframe[["text", "label"]].copy()
    dataframe = dataframe.dropna(subset=["text", "label"])
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe["label"] = dataframe["label"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"] != ""].reset_index(drop=True)

    unknown = sorted(set(dataframe["label"]) - set(LABELS))
    if unknown:
        raise ValueError(
            f"Unknown labels: {unknown}. Expected labels: {LABELS}"
        )

    return dataframe


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def predict_texts(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[list[int], list[float]]:
    prediction_ids: list[int] = []
    confidences: list[float] = []

    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)

        prediction_ids.extend(torch.argmax(probabilities, dim=-1).cpu().tolist())
        confidences.extend(torch.max(probabilities, dim=-1).values.cpu().tolist())

    return prediction_ids, confidences


def measure_average_latency_ms(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> float:
    sample_texts = texts[: min(30, len(texts))]
    if not sample_texts:
        raise ValueError("The frozen test set is empty.")

    # Warm-up requests are excluded.
    for text in sample_texts[:5]:
        encoded = tokenizer(
            text, truncation=True, max_length=128, return_tensors="pt"
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            model(**encoded)
        synchronize(device)

    times_ms: list[float] = []
    for text in sample_texts:
        encoded = tokenizer(
            text, truncation=True, max_length=128, return_tensors="pt"
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            model(**encoded)
        synchronize(device)

        times_ms.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times_ms))


def calculate_model_size_mb(directory: Path) -> float:
    total_bytes = sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    )
    return total_bytes / (1024 * 1024)


def save_confusion_matrix(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(LABELS)))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    display.plot(ax=axis, values_format="d", cmap="Blues", colorbar=False)
    axis.set_title("Fine-Tuned DistilBERT — Frozen Test Set")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_FILE, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return matrix


def negation_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Sentence | Prediction | Confidence |",
        "|---|---:|---:|",
    ]
    for row in rows:
        sentence = str(row["sentence"]).replace("|", r"\|")
        lines.append(
            f"| {sentence} | {row['prediction']} | "
            f"{row['confidence']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    validate_inputs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Frozen test file: {TEST_FILE}")

    test_dataframe = load_test_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    texts = test_dataframe["text"].tolist()
    true_labels = test_dataframe["label"].tolist()
    true_ids = [LABEL_TO_ID[label] for label in true_labels]

    predicted_ids, confidences = predict_texts(
        texts, tokenizer, model, device
    )
    predicted_labels = [ID_TO_LABEL[index] for index in predicted_ids]

    report_text = classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
    REPORT_FILE.write_text(report_text, encoding="utf-8")

    matrix = save_confusion_matrix(true_ids, predicted_ids)
    accuracy = float(accuracy_score(true_ids, predicted_ids))
    macro_f1 = float(
        f1_score(true_ids, predicted_ids, average="macro", zero_division=0)
    )

    recall_values = recall_score(
        true_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )
    per_class_recall = {
        label: float(value)
        for label, value in zip(LABELS, recall_values)
    }

    average_latency_ms = measure_average_latency_ms(
        texts, tokenizer, model, device
    )
    model_size_mb = calculate_model_size_mb(MODEL_DIR)

    negation_ids, negation_confidences = predict_texts(
        NEGATION_TESTS, tokenizer, model, device, batch_size=8
    )
    negation_results = [
        {
            "sentence": sentence,
            "prediction": ID_TO_LABEL[prediction_id],
            "confidence": float(confidence),
        }
        for sentence, prediction_id, confidence in zip(
            NEGATION_TESTS, negation_ids, negation_confidences
        )
    ]

    results = {
        "model_directory": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
        "test_file": str(TEST_FILE.relative_to(PROJECT_ROOT)),
        "test_examples": len(test_dataframe),
        "device": str(device),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "average_inference_latency_ms": average_latency_ms,
        "model_size_mb": model_size_mb,
        "confusion_matrix": matrix.tolist(),
        "negation_tests": negation_results,
        "predictions": [
            {
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": float(confidence),
            }
            for text, true_label, predicted_label, confidence in zip(
                texts, true_labels, predicted_labels, confidences
            )
        ],
    }

    JSON_RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    discussion = (
        "The negation examples provide a focused check of whether the "
        "fine-tuned model follows meaning changes caused by words such as "
        "\"not\". Improvement should only be claimed after comparing these "
        "outputs with the same examples from the baseline model. Without that "
        "baseline comparison, the results describe the fine-tuned model's "
        "behavior but do not independently prove improvement."
    )

    markdown = f"""# Fine-Tuning Evaluation Results

## Evaluation Setup

The best saved DistilBERT checkpoint was evaluated once on the frozen test set
`{TEST_FILE.relative_to(PROJECT_ROOT)}`. The frozen test set contained
**{len(test_dataframe)} examples** and was not used during training or checkpoint
selection.

## Main Metrics

| Metric | Result |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Macro-F1 | {macro_f1:.4f} |
| Negative recall | {per_class_recall["Negative"]:.4f} |
| Neutral recall | {per_class_recall["Neutral"]:.4f} |
| Positive recall | {per_class_recall["Positive"]:.4f} |
| Average inference latency | {average_latency_ms:.2f} ms |
| Saved model size | {model_size_mb:.2f} MB |

## Classification Report

```text
{report_text.rstrip()}
```

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

Matrix values:

```text
{matrix.tolist()}
```

Label order: Negative, Neutral, Positive.

## Negation Tests

{negation_markdown(negation_results)}

## Negation-Handling Discussion

{discussion}
"""

    MARKDOWN_RESULTS_FILE.write_text(markdown, encoding="utf-8")

    print("\nClassification report")
    print(report_text)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Average latency: {average_latency_ms:.2f} ms")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Confusion matrix: {matrix.tolist()}")

    print("\nGenerated files:")
    for path in [
        REPORT_FILE,
        CONFUSION_MATRIX_FILE,
        JSON_RESULTS_FILE,
        MARKDOWN_RESULTS_FILE,
    ]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
 
