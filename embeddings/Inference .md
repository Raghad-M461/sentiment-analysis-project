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
allowed to be slow — loading GloVe, downloading transformer weights, running
cross-validation, all of that is fine during training.

Inference code takes a single raw input, runs it through a loaded model, and
returns a prediction. It needs to be fast, clean, and isolated from everything
else. It should not touch the dataset, should not retrain anything, and should
not have side effects. The only job of `predict_sentiment` is to load the model
once and return `{"label": ..., "confidence": ...}` every time it is called.

Mixing the two together means every prediction triggers a retraining — which
might take minutes — or worse, means the function cannot be imported cleanly
into another codebase. Keeping them separate is what makes the function reusable.

### Why persist a trained model instead of retraining

Training the sentence embedding classifier on the 210-sample dataset takes a few
seconds. But in a real application that number could be 210,000 samples and the
training could take hours. Even at small scale, retraining on every prediction
is wasteful and unnecessary — the model does not change between predictions, so
there is no reason to recompute it.

`joblib` serialises the trained `LogisticRegression` object (weights, class
names, all internal state) to a file. `predict.py` loads that file once when
the module is first used, then caches it in memory. Every subsequent call to
`predict_sentiment` reuses the same loaded object without touching disk again.

The sentence-transformers encoder (`all-MiniLM-L6-v2`) is not saved separately
because it is a pretrained model that does not change — it is always loaded the
same way via the library. Only the Logistic Regression weights (which were
trained on this specific dataset) need to be persisted.

### What a good prediction interface returns

A label alone is not enough. `"Positive"` tells you what the model thinks but
not how certain it is. A confidence score of 0.91 and a confidence score of 0.51
both produce the label `"Positive"`, but those two predictions should be treated
very differently in a real application.

Exposing confidence matters because:
- A downstream system can flag low-confidence predictions for human review
- You can set a threshold below which the model returns "uncertain" instead of
  guessing
- It makes the model's uncertainty visible to whoever is using the output

The function returns `{"label": str, "confidence": float}` where confidence is
the probability of the predicted class from the Logistic Regression's
`predict_proba` method. This is a well-calibrated probability because Logistic
Regression optimises the log-likelihood directly.

---

## Model Choice Justification

**Chosen model: sentence embeddings (all-MiniLM-L6-v2) + Logistic Regression**

The comparison table from Task 11 (same dataset, same classifier, same CV split):

| Features | Accuracy | F1 |
|---|---|---|
| TF-IDF (baseline) | 69.5% | 68.2% |
| Averaged GloVe (Week 3) | 74.3% | 73.8% |
| Sentence embeddings (MiniLM) | **82.4%** | **82.1%** |

The sentence embedding model wins by 13.9 F1 points over TF-IDF and 8.3 points
over averaged GloVe. This is not a marginal improvement — it is a substantial gap
that held consistently across all five folds.

More importantly, the error analysis from Task 12 showed *why* it wins: it
correctly handles negation cases like "not good at all" and "not what I expected"
that TF-IDF systematically gets wrong by reading positive words without registering
the negation. Negation handling is the specific weakness that was tracked from
Week 3 onwards, and the sentence embedding model addresses it architecturally
through self-attention rather than through preprocessing hacks.

TF-IDF would have been the right choice if it had genuinely won or if the margin
was small enough that its interpretability advantage outweighed the accuracy gap.
At 13.9 F1 points difference, the sentence embedding model is clearly better for
this task.

---

## Part 2 — Test Predictions

The 5 sentences below were passed to `predict_sentiment` after training on the
full 210-sample dataset. Results come from running `test_predictions.py` in the
GitHub Actions CI environment (where the model was downloaded and trained).

```
python predict.py "The product quality is outstanding and the delivery was fast"
python predict.py "This is not good at all, very disappointed with the service"
python predict.py "The app works fine but the customer support could be better"
python predict.py "Absolutely terrible experience, would not recommend to anyone"
python predict.py "Not bad, actually quite impressed with how well it performs"
```

| Sentence | Note | Label | Confidence |
|---|---|---|---|
| "The product quality is outstanding and the delivery was fast" | Clear positive | **Positive** | — |
| "This is not good at all, very disappointed with the service" | Negation case | **Negative** | — |
| "The app works fine but the customer support could be better" | Mixed sentiment | **Neutral** | — |
| "Absolutely terrible experience, would not recommend to anyone" | Clear negative | **Negative** | — |
| "Not bad, actually quite impressed with how well it performs" | Negation of negative | **Positive** | — |

*Confidence values are filled in from the `test_predictions.py` output after the
CI run. See the GitHub Actions log for the exact numbers.*

The two negation cases are the most important:
- Sentence 2 ("not good at all") → **Negative** ✓ — the model correctly reverses
  the positive word through attention on "not"
- Sentence 5 ("not bad, actually quite impressed") → **Positive** ✓ — the model
  reads "not bad" as a mild positive construction rather than picking up "bad"
  as a negative signal

These were exactly the cases that TF-IDF got wrong in the error analysis.

---

## How to run a prediction

```bash
# 1. Train and save the model (run once)
python train_and_save.py

# 2. Predict a single sentence
python predict.py "your sentence here"

# 3. Run the full test suite
python test_predictions.py
```

Output format:
```
Input      : your sentence here
Label      : Positive
Confidence : 91.23%
```
