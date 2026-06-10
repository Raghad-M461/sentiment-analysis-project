import csv
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("data/sentiment_dataset_expanded.csv")))
classes = ["Positive", "Negative", "Neutral"]
# stacked: original vs added-real, per class
orig = defaultdict(int); real = defaultdict(int)
for r in rows:
    (orig if r["source"] == "original_manual" else real)[r["label"]] += 1

orig_vals = [orig[c] for c in classes]
real_vals = [real[c] for c in classes]
totals = [orig[c] + real[c] for c in classes]

fig, ax = plt.subplots(figsize=(8, 5))
colors_real = ["#2e8b57", "#c0392b", "#7f8c8d"]
colors_orig = ["#a9dfbf", "#f1948a", "#d5d8dc"]
b1 = ax.bar(classes, real_vals, color=colors_real, label="Added (Hu & Liu real reviews)")
b2 = ax.bar(classes, orig_vals, bottom=real_vals, color=colors_orig,
            edgecolor="black", linewidth=0.8, label="Original (manual)")

for i, c in enumerate(classes):
    ax.text(i, totals[i] + 8, str(totals[i]), ha="center", fontweight="bold")
    if orig[c]:
        ax.text(i, real_vals[i] + orig[c] / 2, f"+{orig[c]} orig",
                ha="center", va="center", fontsize=8)

ax.set_ylabel("Number of samples")
ax.set_title("Class Distribution — Expanded 3-Class Sentiment Dataset (n=1,500)")
ax.set_ylim(0, max(totals) + 60)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.savefig("data/class_distribution.png", dpi=150)
print("Chart saved. Totals:", dict(zip(classes, totals)))
