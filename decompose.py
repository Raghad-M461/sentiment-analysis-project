import csv, importlib.util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
spec = importlib.util.spec_from_file_location("pp","scripts/preprocessing.py")
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run(texts, labels, name):
    X=[pp.preprocess(t) for t in texts]
    m=make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    s=cross_val_score(m,X,labels,cv=cv)
    base=100/len(set(labels))
    print(f"{name:48s} {s.mean()*100:5.1f}%   (random baseline {base:.0f}%)")

rows=list(csv.DictReader(open("data/sentiment_dataset_expanded.csv")))
# A: real data, BINARY only (drop neutral) -> isolates 'real vs synthetic'
bin_rows=[r for r in rows if r["label"] in ("Positive","Negative")]
run([r["text"] for r in bin_rows],[r["label"] for r in bin_rows],
    "Real data, BINARY (Pos/Neg) - isolates real-vs-synthetic")
# B: full 3-class
run([r["text"] for r in rows],[r["label"] for r in rows],
    "Real data, 3-CLASS (Pos/Neg/Neu) - adds neutral difficulty")
print("\nReference: ORIGINAL synthetic data, binary, same config = 82.5%")
