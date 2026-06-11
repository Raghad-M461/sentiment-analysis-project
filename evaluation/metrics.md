# Evaluation Metrics

**Project:** sentiment-analysis-project · **Branch:** `feature/evaluation-metrics`
**Task:** deeper evaluation of the best model (beyond accuracy)

---

## Model Configuration

- **Vectorizer:** scikit-learn `TfidfVectorizer` with default settings (unigrams,
  `ngram_range=(1, 1)`), applied after the custom `preprocess()` step.
- **Preprocessing (unchanged best config):** tokenization, lowercasing, negation
  marking (`not good` → `not neg_good`), a sentiment-aware stop-word list that
  keeps words like *not, never, but, very*, and WordNet lemmatization.
- **Classifier:** `LogisticRegression(max_iter=1000)`, default regularisation.
- **Dataset version:** enriched 3-class dataset, `sentiment_dataset_enriched.csv`,
  210 samples, balanced 70 Positive / 70 Negative / 70 Neutral.
- **Evaluation method:** 5-fold stratified cross-validation, `random_state=42`
  (same methodology as the previous task, so results are comparable).

---

## Evaluation Metrics

| Metric | Value |
|---|---|
| Accuracy | 65.7% |
| Macro Precision | 0.655 |
| Macro Recall | 0.657 |
| Macro F1-score | 0.650 |
| Weighted Precision | 0.655 |
| Weighted Recall | 0.657 |
| Weighted F1-score | 0.650 |

### Per-class report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Positive | 0.656 | 0.571 | 0.611 | 70 |
| Negative | 0.644 | 0.543 | 0.589 | 70 |
| Neutral | 0.667 | 0.857 | 0.750 | 70 |

*Note: because the dataset is balanced (70 per class), the macro and weighted
averages are almost identical. They would diverge on an imbalanced dataset.*

### Confusion matrix (rows = true, cols = predicted)

|  | Pred Positive | Pred Negative | Pred Neutral |
|---|---|---|---|
| **True Positive** | 40 | 16 | 14 |
| **True Negative** | 16 | 38 | 16 |
| **True Neutral** | 5 | 5 | 60 |

See `confusion_matrix.png` for the visualization.

---

## Negation Investigation (Part 3)

Four challenging negation sentences were passed through the trained model:

| Sentence | Expected | Predicted | Result | Confusion-matrix cell |
|---|---|---|---|---|
| "The product is not good" | Negative | Negative | correct | True Neg / Pred Neg (diagonal) |
| "The service was not bad at all" | Positive | Positive | correct | True Pos / Pred Pos (diagonal) |
| "Not happy with the support" | Negative | Negative | correct | True Neg / Pred Neg (diagonal) |
| "Not the worst experience" | Positive | Negative | **wrong** | True Pos / Pred Neg (off-diagonal) |

The negation-marking step handled the first three correctly: it turns *not good*
into `not neg_good`, which the model has learned to read as negative, and *not
bad* into `not neg_bad`, read as positive. The failure is **"Not the worst
experience"**, a double negative (litotes) that actually means *acceptable /
positive*. After preprocessing it becomes `not neg_worst neg_experience`. The
word *worst* is such a strong negative signal that, even negated, the bag-of-words
model still leans Negative. This error lands in the True Positive / Predicted
Negative cell, which is the same Positive-vs-Negative confusion that dominates the
matrix.

---

## Findings

- **Strongest-performing class:** **Neutral** (F1 = 0.750, recall 0.857). The
  model finds 60 of 70 Neutral sentences. Neutral text has distinctive surface
  cues (questions, factual words) that make it easy to separate.
- **Weakest-performing class:** **Negative** (F1 = 0.589, recall 0.543). The model
  misses many Negative examples, often labelling them Positive or Neutral.
- **Common failure pattern:** Positive and Negative are confused with each other
  (16 + 16 = 32 mistakes), mostly on the challenging cases — sarcasm, mixed
  sentiment, and double negation. The bag-of-words model cannot use word order, so
  it cannot resolve these.
- **Observed limitations:** simple negation is handled by the `neg_` marking, but
  double negation and implicit sentiment are not; the Neutral class may be
  artificially easy because the authored neutral sentences use obvious cues; and
  the dataset is small (210), so cross-validation is somewhat noisy.
