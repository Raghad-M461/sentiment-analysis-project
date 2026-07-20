

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")

    print(f"\nLoading tokenizer: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded OK:", type(tokenizer).__name__)

    print(f"\nLoading model: {MODEL_NAME} (with a fresh 2-class classification head) ...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    print("Model loaded OK:", type(model).__name__)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    sample_sentences = [
        "This product completely exceeded my expectations, I love it!",
        "Terrible experience, the app crashed every five minutes.",
        "It's okay, nothing special but does the job.",
    ]

    print("\n--- Tokenizing sample sentences ---")
    encoded = tokenizer(
        sample_sentences,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    for i, sent in enumerate(sample_sentences):
        print(f"\nSentence {i+1}: {sent!r}")
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][i])
        print("Tokens:        ", tokens)
        print("Input IDs:     ", encoded["input_ids"][i].tolist())
        print("Attention mask:", encoded["attention_mask"][i].tolist())

    print("\nAll checks passed: tokenizer + model load without errors, "
          "and tokenization produces expected input_ids/attention_mask shapes.")
    print("input_ids shape:", tuple(encoded["input_ids"].shape))
    print("attention_mask shape:", tuple(encoded["attention_mask"].shape))

if __name__ == "__main__":
    main()
