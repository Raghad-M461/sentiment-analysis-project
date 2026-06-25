# Inference — Notes and Model Choice

**Branch:** `feature/contextual-embeddings`  
**Scripts:** `train_and_save.py`, `predict.py`, `test_predictions.py`

---

## Part 1 — Concepts

### Training code vs inference code

Training code and inference code do completely different jobs and should be
kept separate.

Training code loads the full dataset, fits a model, evaluates it, and saves
the result. It runs once (or occasionally when the dataset changes). It is
allowed to be slow — downloading transformer weights, running cross-validation,
all of that is fine during training.

Inference code takes a single raw input, runs it through a loaded model, and
returns a prediction. It needs to be fast, clean, and isolated from everything
else. It should not touch the dataset, should not retrain anything, and should
not have side effects. The only job of `predict_sentiment` is to load the model
once and return `{"label": ..., "confidence": ...}` every time it is called.

Mixing the two together means every prediction triggers a retraining — which
could take minutes — or worse, means the function cannot be imported cleanly
into another codebase. Keeping them separate is what makes the function reusable.

### Why persist a trained model instead of retraining

Training the sentence embedding classifier on 210 samples takes a few seconds.
But in a real application that number could be 210,000 samples and training could
take hours. Even at small scale, retraining on every prediction is wasteful and
unnecessary — the model does not change between predictions.

`joblib` serialises the trained `LogisticRegression` object to disk.
`predict.py` loads that file once when the module is first used, then caches it
in memory. Every subsequent call to `predict_sentiment` reuses the same loaded
object without touching disk again.

The sentence-transformers encoder is not saved separately because it is a
pretrained model that does not change — it is always loaded the same way via the
library. Only the Logistic Regression weights (trained on this specific dataset)
need to be persisted.

### What a good prediction interface returns

A label alone is not enough. `"Positive"` tells you what the model thinks but
not how certain it is. A confidence score of 0.91 and a confidence score of 0.51
both produce the label `"Positive"`, but those two predictions should be treated
very differently downstream.

Exposing confidence matters because:
- A downstream system can flag low-confidence predictions for human review
- You can set a threshold below which the model says "uncertain" instead of guessing
- It makes the model's uncertainty visible to whoever is using the output

The function returns `{"label": str, "confidence": float}` where confidence is
the probability of the predicted class from Logistic Regression's `predict_proba`.

---

## Model Choice Justification

**Chosen model: sentence embeddings (all-MiniLM-L6-v2) + Logistic Regression**

The comparison table from Task 11 (same dataset, same classifier, same CV split):

| Features | Accuracy | F1 |
|---|---|---|
| TF-IDF (baseline) | 69.5% | 68.2% |
| Averaged GloVe (Week 3) | 74.3% | 73.8% |
| Sentence embeddings (MiniLM) | **82.4%** | **82.1%** |

The sentence embedding model wins by 13.9 F1 points over TF-IDF. More
importantly, the error analysis in Task 12 showed *why* it wins: it correctly
handles negation cases that TF-IDF systematically gets wrong. Negation handling
was the specific weakness tracked from Week 3 onwards, and the sentence embedding
model addresses it architecturally through self-attention.

TF-IDF would have been the right choice if it had genuinely won or if the margin
was small. At 13.9 F1 points difference, the sentence embedding model is clearly
better for this task.

---

## Part 2 — Test Predictions

Trained on full 210-sample dataset using `train_and_save.py`, then tested with
`test_predictions.py`. All results below are real outputs from the GitHub Actions
CI run.

| # | Sentence | Note | Label | Confidence |
|---|---|---|---|---|
| 1 | "The product quality is outstanding and the delivery was fast" | Clear positive | **Positive** | 78% |
| 2 | "This is not good at all, very disappointed with the service" | Negation | **Negative** | 57% |
| 3 | "The app works fine but the customer support could be better" | Mixed sentiment | **Negative** | 54% |
| 4 | "Absolutely terrible experience, would not recommend to anyone" | Clear negative | **Negative** | 53% |
| 5 | "Not bad, actually quite impressed with how well it performs" | Negation of negative | **Positive** | 66% |

### What the results show

**Sentences 2 and 5 are the most important** — both are negation cases that
TF-IDF consistently got wrong in the error analysis:

- Sentence 2: "not good at all" → correctly predicted **Negative** (57%). TF-IDF
  would read "good" and predict Positive. The contextual model attends to "not"
  and shifts the prediction.

- Sentence 5: "not bad, actually quite impressed" → correctly predicted
  **Positive** (66%). TF-IDF might pick up "bad" and predict Negative. The model
  correctly reads "not bad" as a mild positive construction.

**The confidence scores are modest (53–78%)** — this reflects honest uncertainty
on a 3-class problem (Positive / Negative / Neutral) with only 210 training
samples. These scores are not a sign of a broken model; they reflect the genuine
difficulty of short sentiment sentences and the small training set size.

**Sentence 3 (mixed sentiment)** was predicted as Negative with 54% confidence.
The sentence has positive ("works fine") and negative ("could be better") signals.
The model picks the negative side, which is reasonable — "could be better" implies
dissatisfaction. This is a hard case and the low confidence reflects that.

---

## How to run a prediction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and save the model (run once)
python train_and_save.py

# 3. Predict a single sentence
python predict.py "your sentence here"
```

Output:
```
Input      : Not bad, actually quite impressed with how well it performs
Label      : Positive
Confidence : 66.00%
```
