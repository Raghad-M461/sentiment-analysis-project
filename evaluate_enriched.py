import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import preprocessing as pp

# load the enriched dataset
rows = list(csv.DictReader(open("data/sentiment_dataset_enriched.csv")))
X = [pp.preprocess(r["text"]) for r in rows]
y = [r["label"] for r in rows]
classes = ["Positive", "Negative", "Neutral"]

# same model and 5-fold setup as before
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv)
pred = cross_val_predict(model, X, y, cv=cv)

acc = accuracy_score(y, pred)
p, r, f, s = precision_recall_fscore_support(y, pred, labels=classes)
mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro")
cm = confusion_matrix(y, pred, labels=classes).tolist()

print("dataset size:", len(rows))
print("cross-validation accuracy:", round(scores.mean() * 100, 1), "%")
print("fold scores:", [round(v * 100) for v in scores])
print()
print("per class:")
for i, c in enumerate(classes):
    print(" ", c, "- precision", round(p[i], 3),
          "recall", round(r[i], 3), "f1", round(f[i], 3))
print()
print("macro f1:", round(mf, 3))
print("accuracy:", round(acc, 3))
print()
print("confusion matrix (rows = true, cols = pred):", classes)
for row in cm:
    print(" ", list(row))

os.makedirs("evaluation", exist_ok=True)
with open("evaluation/results_enriched.txt", "w") as fh:
    fh.write("Enriched dataset evaluation\n")
    fh.write("dataset size: " + str(len(rows)) + "\n")
    fh.write("accuracy: " + str(round(acc * 100, 1)) + "%\n")
    fh.write("macro f1: " + str(round(mf, 3)) + "\n\n")
    for i, c in enumerate(classes):
        fh.write(c + " precision " + str(round(p[i], 3)) +
                 " recall " + str(round(r[i], 3)) +
                 " f1 " + str(round(f[i], 3)) + "\n")
    fh.write("\nconfusion matrix " + str(classes) + "\n")
    for row in cm:
        fh.write(str(list(row)) + "\n")
