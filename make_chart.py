import csv
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("data/sentiment_dataset_enriched.csv")))
classes = ["Positive", "Negative", "Neutral"]
orig = defaultdict(int); enr = defaultdict(int)
for r in rows:
    (orig if r["source"] == "original_manual" else enr)[r["label"]] += 1
orig_v = [orig[c] for c in classes]; enr_v = [enr[c] for c in classes]
tot = [orig[c] + enr[c] for c in classes]

fig, ax = plt.subplots(figsize=(8, 5))
c_enr = ["#2e8b57", "#c0392b", "#7f8c8d"]; c_org = ["#a9dfbf", "#f1948a", "#d5d8dc"]
ax.bar(classes, enr_v, color=c_enr, label="Added (manual enrichment)")
ax.bar(classes, orig_v, bottom=enr_v, color=c_org, edgecolor="black", linewidth=0.8,
       label="Original (manual, templated)")
for i, c in enumerate(classes):
    ax.text(i, tot[i] + 1.5, str(tot[i]), ha="center", fontweight="bold")
    if orig[c]:
        ax.text(i, enr_v[i] + orig[c] / 2, f"+{orig[c]} orig", ha="center", va="center", fontsize=8)
ax.set_ylabel("Number of samples")
ax.set_title("Class Distribution — Enriched 3-Class Dataset (Option B, n=210)")
ax.set_ylim(0, max(tot) + 12)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.subplots_adjust(bottom=0.2); plt.tight_layout()
plt.savefig("data/class_distribution.png", dpi=150)
print("Chart saved. Totals:", dict(zip(classes, tot)))
