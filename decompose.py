import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import preprocessing as pp
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run(texts, labels, name):
    X = [pp.preprocess(t) for t in texts]
    m = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    s = cross_val_score(m, X, labels, cv=cv)
    print(f"{name:52s} {s.mean()*100:5.1f}%   (chance {100/len(set(labels)):.0f}%)")

rows = list(csv.DictReader(open("data/sentiment_dataset_enriched.csv")))

o = [r for r in rows if r["source"] == "original_manual"]
run([r["text"] for r in o], [r["label"] for r in o],
    "Original 80 only, binary (old setup)")

b = [r for r in rows if r["label"] in ("Positive", "Negative")]
run([r["text"] for r in b], [r["label"] for r in b],
    "Enriched, binary (Pos/Neg incl. challenging)")

run([r["text"] for r in rows], [r["label"] for r in rows],
    "Enriched, 3-class (adds Neutral)")
