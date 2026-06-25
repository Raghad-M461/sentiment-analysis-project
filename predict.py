"""
predict.py
Branch: feature/contextual-embeddings

Reusable prediction function and CLI for sentiment analysis.
Loads the saved model from app/model/ and returns a label + confidence score.

Usage (CLI):
    python predict.py "your sentence here"
    python predict.py "not bad at all"

Usage (import):
    from predict import predict_sentiment
    result = predict_sentiment("The product is great")
    print(result)  # {"label": "Positive", "confidence": 0.91}

Requirements:
    - Run train_and_save.py first to generate the model files in app/model/
    - sentence-transformers must be installed
"""

import os
import sys
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR    = os.path.join("app", "model")
CLF_PATH     = os.path.join(MODEL_DIR, "logreg_classifier.joblib")
ENCODER_NAME = "all-MiniLM-L6-v2"

# ── module-level cache so the model loads once per process ─────────────────
_encoder = None
_clf     = None


def _load_model():
    """Load encoder and classifier once and cache them."""
    global _encoder, _clf

    if _clf is None:
        if not os.path.exists(CLF_PATH):
            raise FileNotFoundError(
                f"model not found at {CLF_PATH}\n"
                "run:  python train_and_save.py"
            )
        _clf = joblib.load(CLF_PATH)

    if _encoder is None:
        _encoder = SentenceTransformer(ENCODER_NAME)


def predict_sentiment(text: str) -> dict:
    """
    Predict the sentiment of a piece of text.

    Args:
        text: raw input string (no preprocessing needed)

    Returns:
        dict with keys:
            "label"      — predicted class (Positive / Negative / Neutral)
            "confidence" — probability of the predicted class (0.0 to 1.0)
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    _load_model()

    # encode the input sentence into a 384-dim vector
    vector = _encoder.encode([text])

    # get class probabilities from the logistic regression
    proba   = _clf.predict_proba(vector)[0]
    label   = _clf.classes_[proba.argmax()]
    confidence = float(proba.max())

    return {"label": label, "confidence": round(confidence, 4)}


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("usage: python predict.py \"your sentence here\"")
        print("example: python predict.py \"not bad at all\"")
        sys.exit(1)

    text   = " ".join(sys.argv[1:])
    result = predict_sentiment(text)

    print(f"\nInput      : {text}")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
