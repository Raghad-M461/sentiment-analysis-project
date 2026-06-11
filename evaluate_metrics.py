"""Task 8 """

import csv
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import preprocessing as pp


DATASET_PATH = "evaluation/sentiment_dataset_enriched.csv"

rows = list(csv.DictReader(open(DATASET_PATH, encoding="utf-8")))

X = [pp.preprocess(r["text"]) for r in rows]
y = [r["label"] for r in rows]

classes = ["Positive", "Negative", "Neutral"]

model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy = cross_val_score(model, X, y, cv=cv).mean()
predictions = cross_val_predict(model, X, y, cv=cv)

print("Accuracy:", round(accuracy * 100, 1), "%\n")

report = classification_report(y, predictions, labels=classes, digits=3)
print(report)

cm = confusion_matrix(y, predictions, labels=classes)
print("Confusion Matrix:")
print(cm)

os.makedirs("evaluation", exist_ok=True)

with open("evaluation/classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Accuracy: " + str(round(accuracy * 100, 1)) + "%\n\n")
    f.write(report)

plt.figure(figsize=(6.2, 5.2))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix - Enriched 3-Class Model")
plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png", dpi=150)

print("\nSaved:")
print("evaluation/classification_report.txt")
print("evaluation/confusion_matrix.png")
