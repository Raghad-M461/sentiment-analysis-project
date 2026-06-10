"""
Build the expanded 3-class sentiment dataset.

Strategy: KEEP the original 80 hand-authored sentences unchanged, ADD real
customer-review sentences from the Hu & Liu (2004) Customer Review Datasets
(NLTK product_reviews_1 + product_reviews_2), and balance the three classes.
Reproducible via a fixed seed.
"""
import csv, random, re, sys
import nltk
from nltk.corpus import product_reviews_1 as pr1, product_reviews_2 as pr2
from nltk.tokenize.treebank import TreebankWordDetokenizer

SEED = 42
PER_CLASS = 500           # target size per class (balanced)
random.seed(SEED)
detok = TreebankWordDetokenizer()

# ---- 1. Original dataset (unchanged) -------------------------------------
sys.path.insert(0, "scripts")
import ngram_comparison as ngc
original = [(t, lab, "original_manual") for t, lab in zip(ngc.texts, ngc.labels)]
orig_pos = [r for r in original if r[1] == "Positive"]
orig_neg = [r for r in original if r[1] == "Negative"]
print(f"Original kept: {len(orig_pos)} Positive, {len(orig_neg)} Negative, 0 Neutral")

# ---- 2. Extract + clean real sentences from Hu & Liu ----------------------
def clean(tokens):
    txt = detok.detokenize(tokens)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def word_count(s):
    return len(s.split())

pos_real, neg_real, neu_real = [], [], []
seen = set()
for corpus, cname in ((pr1, "huliu_pr1"), (pr2, "huliu_pr2")):
    for review in corpus.reviews():
        prod = review  # provenance handled at corpus level
        for rl in review.review_lines:
            txt = clean(rl.sent)
            key = txt.lower()
            # quality filters: drop junk, too-short, duplicates, mostly-symbol lines
            if word_count(txt) < 4 or word_count(txt) > 40:
                continue
            if not re.search(r"[a-zA-Z]", txt):
                continue
            if key in seen:
                continue
            seen.add(key)
            if rl.features:
                score = sum(int(p) for _, p in rl.features
                            if p.lstrip("+-").isdigit())
                if score > 0:
                    pos_real.append((txt, "Positive", cname))
                elif score < 0:
                    neg_real.append((txt, "Negative", cname))
                # score == 0 with conflicting features -> ambiguous, skip
            else:
                neu_real.append((txt, "Neutral", cname))

print(f"Real available after cleaning: "
      f"{len(pos_real)} Pos, {len(neg_real)} Neg, {len(neu_real)} Neu")

# ---- 3. Sample to balance (originals counted first) -----------------------
def take(real_rows, already):
    need = PER_CLASS - already
    random.shuffle(real_rows)
    return real_rows[:need]

rows = []
rows += orig_pos + take(pos_real, len(orig_pos))
rows += orig_neg + take(neg_real, len(orig_neg))
rows += take(neu_real, 0)            # neutral is all real
random.shuffle(rows)

# ---- 4. Write CSV ---------------------------------------------------------
with open("data/sentiment_dataset_expanded.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "label", "source"])
    w.writerows(rows)

from collections import Counter
dist = Counter(r[1] for r in rows)
src = Counter(r[2] for r in rows)
print(f"\nFINAL expanded dataset: {len(rows)} samples")
print("By class :", dict(dist))
print("By source:", dict(src))
