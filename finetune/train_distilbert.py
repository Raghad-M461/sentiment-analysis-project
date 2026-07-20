from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "train.csv"
VALIDATION_FILE = PROJECT_ROOT / "validation.csv"

FINETUNE_DIR = PROJECT_ROOT / "finetune"
OUTPUT_DIR = FINETUNE_DIR / "training_output"
BEST_MODEL_DIR = FINETUNE_DIR / "best_checkpoint"
METRICS_FILE = FINETUNE_DIR / "epoch_metrics.json"


LABEL_TO_ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def load_split(file_path: Path) -> Dataset:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {file_path}\n"
            "Make sure train.csv and validation.csv are in the repository root."
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe[["text", "label"]].copy()
    dataframe = dataframe.dropna(subset=["text", "label"])

    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe["label"] = dataframe["label"].astype(str).str.strip()

    dataframe = dataframe[dataframe["text"] != ""]

    unknown_labels = set(dataframe["label"]) - set(LABEL_TO_ID)

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found in {file_path.name}: "
            f"{sorted(unknown_labels)}\n"
            f"Expected labels: {sorted(LABEL_TO_ID)}"
        )

    dataframe["labels"] = dataframe["label"].map(LABEL_TO_ID)

    dataframe = dataframe[["text", "labels"]].reset_index(drop=True)

    return Dataset.from_pandas(
        dataframe,
        preserve_index=False,
    )


def compute_metrics(evaluation_prediction):

    logits, labels = evaluation_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        )
    }


def main():
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DistilBERT Fine-Tuning")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training will run on CPU.")

    print(f"\nTraining file: {TRAIN_FILE}")
    print(f"Validation file: {VALIDATION_FILE}")

    train_dataset = load_split(TRAIN_FILE)
    validation_dataset = load_split(VALIDATION_FILE)

    print(f"\nTraining examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    tokenized_train = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    tokenized_validation = validation_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=4,
        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,
        weight_decay=0.01,

        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1
            )
        ],
    )

    print("\nStarting training...\n")

    train_result = trainer.train()

    trainer.save_model(str(BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(BEST_MODEL_DIR))

    log_history = trainer.state.log_history

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            log_history,
            file,
            indent=2,
        )

    final_validation_results = trainer.evaluate()

    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)

    print(
        f"Best checkpoint: "
        f"{trainer.state.best_model_checkpoint}"
    )
    print(
        f"Best validation loss: "
        f"{trainer.state.best_metric}"
    )
    print(
        f"Final training loss: "
        f"{train_result.training_loss:.4f}"
    )
    print(
        f"Final validation loss: "
        f"{final_validation_results['eval_loss']:.4f}"
    )
    print(
        f"Final validation accuracy: "
        f"{final_validation_results['eval_accuracy']:.4f}"
    )
    print(f"Saved best model to: {BEST_MODEL_DIR}")
    print(f"Saved metrics to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
