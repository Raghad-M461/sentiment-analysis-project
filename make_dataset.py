"""
make_dataset.py  —  Option B (Dataset Enrichment)

Builds the enriched 3-class dataset WITHOUT importing any external dataset.
It combines:
  - the original 80 hand-authored sentences (kept unchanged), and
  - 130 manually authored enrichment sentences from enriched_samples.py
    (Neutral class + linguistic diversity + challenging cases).

Result: balanced 70 / 70 / 70 (Positive / Negative / Neutral) = 210 samples.
Deterministic order via a fixed sort; no randomness needed.
"""
import csv
from collections import Counter

import ngram_comparison as ngc          # holds the original 80 sentences
from enriched_samples import ENRICHED   # 130 manual additions

# 1. Original dataset (kept) -> tag source + a 'template' category
original = [
    (t, lab, "original_manual", "template")
    for t, lab in zip(ngc.texts, ngc.labels)
]

# 2. Manual enrichment (authored by hand for this task)
enriched = [(t, lab, "enriched_manual", cat) for t, lab, cat in ENRICHED]

rows = original + enriched
rows.sort(key=lambda r: (r[1], r[2]))   # stable, reproducible order

with open("data/sentiment_dataset_enriched.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "label", "source", "category"])
    w.writerows(rows)

print(f"Total samples: {len(rows)}")
print("By class :", dict(Counter(r[1] for r in rows)))
print("By source:", dict(Counter(r[2] for r in rows)))
