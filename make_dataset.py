
import csv
from collections import Counter

import ngram_comparison as ngc          # holds the original 80 sentences
from enriched_samples import ENRICHED   # 130 manual additions

original = [
    (t, lab, "original_manual", "template")
    for t, lab in zip(ngc.texts, ngc.labels)
]

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
