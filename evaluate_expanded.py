"""
Re-evaluation of the BEST configuration from the previous task
(preprocess() + TF-IDF unigrams + Logistic Regression) on the expanded
3-class dataset. Same methodology as before: 5-fold stratified CV, seed=42.
"""
import csv, importlib.util
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

spec = importlib.util.spec_from_file_location("pp", "scripts/preprocessing.py")
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)

rows = list(csv.DictReader(open("data/sentiment_dataset_expanded.csv")))
texts = [pp.preprocess(r["text"]) for r in rows]      # BEST config, fixed
labels = [r["label"] for r in rows]
classes = ["Positive", "Negative", "Neutral"]

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc = cross_val_score(model, texts, labels, cv=cv, scoring="accuracy")
preds = cross_val_predict(model, texts, labels, cv=cv)

print("="*64)
print("RE-EVALUATION — Expanded 3-class dataset (n=1500)")
print("Config: preprocess() + TF-IDF unigrams + LogisticRegression")
print("="*64)
print(f"5-fold CV accuracy: {acc.mean()*100:.1f}%  (+/- {acc.std()*100:.1f})")
print(f"Folds: {[f'{x*100:.0f}%' for x in acc]}\n")
print(classification_report(labels, preds, labels=classes, digits=3))
print("Confusion matrix (rows=true, cols=pred)  order:", classes)
print(confusion_matrix(labels, preds, labels=classes))

# Save machine-readable results
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
p, r, f, s = precision_recall_fscore_support(labels, preds, labels=classes)
macro = precision_recall_fscore_support(labels, preds, average="macro")
with open("build/evaluation/results_expanded.txt", "w") as fh:
    fh.write("Expanded 3-class re-evaluation (preprocess()+TF-IDF unigrams+LogReg, 5-fold CV, seed=42)\n\n")
    fh.write(f"Accuracy: {accuracy_score(labels,preds)*100:.1f}%  (CV mean {acc.mean()*100:.1f}% +/- {acc.std()*100:.1f})\n")
    fh.write(f"Macro precision/recall/F1: {macro[0]*100:.1f}% / {macro[1]*100:.1f}% / {macro[2]*100:.1f}%\n\n")
    fh.write("Per-class:\n")
    for i,c in enumerate(classes):
        fh.write(f"  {c:9s} P={p[i]*100:5.1f}%  R={r[i]*100:5.1f}%  F1={f[i]*100:5.1f}%  (n={s[i]})\n")
    fh.write("\nConfusion matrix (rows=true, cols=pred) order "+str(classes)+":\n")
    fh.write(str(confusion_matrix(labels, preds, labels=classes)))
print("\nSaved -> build/evaluation/results_expanded.txt")
