"""
test_predictions.py
Branch: feature/contextual-embeddings

Tests the predict_sentiment function on 5 hand-written sentences.
Includes at least one negation case and one mixed-sentiment case.

Usage:
    python test_predictions.py

Requires train_and_save.py to have been run first.
"""

from predict import predict_sentiment

TEST_SENTENCES = [
    # (sentence, note)
    (
        "The product quality is outstanding and the delivery was fast",
        "clear positive — strong sentiment words",
    ),
    (
        "This is not good at all, very disappointed with the service",
        "negation case — 'not good' plus explicit negative word",
    ),
    (
        "The app works fine but the customer support could be better",
        "mixed sentiment — positive function, negative support",
    ),
    (
        "Absolutely terrible experience, would not recommend to anyone",
        "clear negative — strong negative vocabulary",
    ),
    (
        "Not bad, actually quite impressed with how well it performs",
        "negation of negative — 'not bad' used as mild positive",
    ),
]


def main():
    print("=" * 65)
    print("SENTIMENT PREDICTION TEST — 5 sentences")
    print("model: all-MiniLM-L6-v2 + Logistic Regression")
    print("=" * 65)

    for sentence, note in TEST_SENTENCES:
        result = predict_sentiment(sentence)
        label  = result["label"]
        conf   = result["confidence"]

        print(f"\n[{label:9s}  {conf:.0%}]  {sentence}")
        print(f"  note: {note}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
