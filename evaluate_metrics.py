"""Task 8 - full evaluation metrics for the best model on the enriched dataset.
Produces a classification report and a confusion matrix image, using the same
5-fold cross-validation methodology as the previous task."""
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import preprocessing as pp

rows = list(csv.DictReader(open("sentiment_dataset_enriched.csv")))
X = [pp.preprocess(r["text"]) for r in rows]
y = [r["label"] for r in rows]
classes = ["Positive", "Negative", "Neutral"]

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = cross_val_score(model, X, y, cv=cv).mean()
pred = cross_val_predict(model, X, y, cv=cv)

print("accuracy:", round(acc*100, 1), "%\n")
report = classification_report(y, pred, labels=classes, digits=3)
print(report)
cm = confusion_matrix(y, pred, labels=classes)
print("confusion matrix:\n", cm)

# save text report
os.makedirs("evaluation", exist_ok=True)
with open("evaluation/classification_report.txt", "w") as f:
    f.write("Accuracy: " + str(round(acc*100,1)) + "%\n\n")
    f.write(classification_report(y, pred, labels=classes, digits=3))

# confusion matrix heatmap
plt.figure(figsize=(6.2, 5.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=classes, yticklabels=classes,
            annot_kws={"size": 14, "weight": "bold"})
plt.xlabel("Predicted label", fontsize=11)
plt.ylabel("True label", fontsize=11)
plt.title("Confusion Matrix - Enriched 3-Class Model (n=210)", fontsize=12)
plt.tight_layout()
plt.savefig("evaluation/confusion_matrix.png", dpi=150)
print("\nSaved evaluation/classification_report.txt and evaluation/confusion_matrix.png")
