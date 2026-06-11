
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import preprocessing as pp


DATASET_PATH = "evaluation/sentiment_dataset_enriched.csv"

rows = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))

X = [pp.preprocess(r["text"]) for r in rows]
y = [r["label"] for r in rows]

model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

model.fit(X, y)

tests = [
    ("The product is not good", "Negative"),
    ("The service was not bad at all", "Positive"),
    ("Not happy with the support", "Negative"),
    ("Not the worst experience", "Positive"),
]

print(f"{'Sentence':35s} {'Expected':10s} {'Predicted':10s} Result")
print("-" * 75)

for text, expected in tests:
    processed_text = pp.preprocess(text)
    prediction = model.predict([processed_text])[0]
    result = "OK" if prediction == expected else "WRONG"

    print(f"{text:35s} {expected:10s} {prediction:10s} {result}")
    print("Preprocessed:", processed_text)
    print()
