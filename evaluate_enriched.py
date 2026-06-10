"""Re-evaluate the BEST config (preprocess()+TF-IDF unigrams+LogReg) on the
Option B enriched 3-class dataset. Same methodology: 5-fold stratified CV, seed 42."""
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import preprocessing as pp

rows = list(csv.DictReader(open("data/sentiment_dataset_enriched.csv")))
X = [pp.preprocess(r["text"]) for r in rows]
y = [r["label"] for r in rows]
classes = ["Positive", "Negative", "Neutral"]
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc = cross_val_score(model, X, y, cv=cv)
pred = cross_val_predict(model, X, y, cv=cv)
print("="*60)
print("RE-EVALUATION — Option B enriched dataset (n=210)")
print("Config: preprocess() + TF-IDF unigrams + LogisticRegression")
print("="*60)
print(f"5-fold CV accuracy: {acc.mean()*100:.1f}% (+/- {acc.std()*100:.1f})")
print(f"Folds: {[f'{x*100:.0f}%' for x in acc]}\n")
print(classification_report(y, pred, labels=classes, digits=3))
print("Confusion matrix (rows=true, cols=pred)", classes)
print(confusion_matrix(y, pred, labels=classes))

p, r, f, s = precision_recall_fscore_support(y, pred, labels=classes)
macro = precision_recall_fscore_support(y, pred, average="macro")
os.makedirs("evaluation", exist_ok=True)
with open("evaluation/results_enriched.txt", "w") as fh:
    fh.write("Option B enriched re-evaluation (preprocess()+TF-IDF unigrams+LogReg, 5-fold CV, seed 42)\n\n")
    fh.write(f"Accuracy: {accuracy_score(y,pred)*100:.1f}% (CV mean {acc.mean()*100:.1f}% +/- {acc.std()*100:.1f})\n")
    fh.write(f"Macro P/R/F1: {macro[0]*100:.1f}% / {macro[1]*100:.1f}% / {macro[2]*100:.1f}%\n\nPer-class:\n")
    for i,c in enumerate(classes):
        fh.write(f"  {c:9s} P={p[i]*100:5.1f}%  R={r[i]*100:5.1f}%  F1={f[i]*100:5.1f}%  (n={s[i]})\n")
    fh.write("\nConfusion matrix (rows=true, cols=pred) "+str(classes)+":\n")
    fh.write(str(confusion_matrix(y, pred, labels=classes)))
print("\nSaved -> evaluation/results_enriched.txt")
