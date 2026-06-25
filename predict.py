
import os
import sys
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR    = os.path.join("app", "model")
CLF_PATH     = os.path.join(MODEL_DIR, "logreg_classifier.joblib")
ENCODER_NAME = "all-MiniLM-L6-v2"

_encoder = None
_clf     = None


def _load_model():
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
    
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    _load_model()

    vector = _encoder.encode([text])

    proba   = _clf.predict_proba(vector)[0]
    label   = _clf.classes_[proba.argmax()]
    confidence = float(proba.max())

    return {"label": label, "confidence": round(confidence, 4)}


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
