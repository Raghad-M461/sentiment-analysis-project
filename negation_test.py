"""Task 8 Part 3 - negation failure investigation."""
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import preprocessing as pp

rows = list(csv.DictReader(open("sentiment_dataset_enriched.csv")))
X = [pp.preprocess(r["text"]) for r in rows]
y = [r["label"] for r in rows]

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X, y)

tests = [
    ("The product is not good", "Negative"),
    ("The service was not bad at all", "Positive"),
    ("Not happy with the support", "Negative"),
    ("Not the worst experience", "Positive"),
]
print(f"{'sentence':32s} {'expected':9s} {'predicted':9s} {'preprocessed'}")
print("-"*90)
for text, expected in tests:
    proc = pp.preprocess(text)
    p = model.predict([proc])[0]
    mark = "OK" if p == expected else "WRONG"
    print(f"{text:32s} {expected:9s} {p:9s} [{mark}]  {proc!r}")
