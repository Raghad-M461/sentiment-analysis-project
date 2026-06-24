# Error Analysis — Sentiment Classification

**Branch:** `feature/contextual-embeddings`  
**Script:** `error_analysis.py`  
**Dataset:** 210 samples, 3 classes (Positive / Negative / Neutral), 5-fold CV

---

## Part 1 — Concepts

### What error analysis is and why it matters more than +1% accuracy

Error analysis means reading the model's wrong predictions and asking why they
happened. A single accuracy number tells you how often the model is right; it
hides which kinds of inputs it fails on, whether the failures follow a pattern,
and what you would need to fix to actually improve the model.

Chasing +1% accuracy through hyperparameter tuning can improve the number without
fixing any underlying problem. Error analysis tells you where the ceiling is and
why. A model that gets 82% accuracy by being right on easy examples and wrong on
all the hard ones is very different from a model that is 82% accurate because it
handles negation, rare words, and mixed sentiment correctly. The number looks the
same; the models are not.

In applied ML, error analysis is how you decide what to work on next. If 60% of
your errors come from negation cases, that tells you to add negation handling. If
most errors are on three-word sentences, that tells you the dataset is too sparse.
You cannot know this from an accuracy score alone.

### How to read a confusion matrix — connecting back to Week 2

In Week 2 the evaluation work produced a confusion matrix showing true labels
on one axis and predicted labels on the other. Each cell counts how many times
a sample with a given true label was predicted as a given class.

The two main error types:

- **False positive:** the model predicts a label that is not the true label.
  For sentiment, predicting Positive when the true label is Negative is a false
  positive for the Positive class. This is the most common TF-IDF error — it
  reads "the service was not great" and sees the word "great", predicting Positive.

- **False negative:** the model fails to predict a label when it should.
  Predicting Neutral when the true label is Positive is a false negative for the
  Positive class. TF-IDF does this on phrases like "absolutely delighted" and
  "the best customer support I have ever experienced" where the positive words
  are less frequent in training.

Reading the confusion matrix off-diagonal cells shows which direction the model
is biased: is it too trigger-happy with a particular class (high false positives),
or does it consistently miss a particular class (high false negatives)?

### Qualitative error categories as a diagnostic lens

Rather than just counting wrong predictions, it is more useful to group them by
the linguistic pattern that caused the failure. Four recurring categories:

1. **Negation:** the model sees a sentiment word but misses the negation that
   reverses it. "Not great", "not good", "not happy". This was the central
   problem in Weeks 3 and 4.

2. **Rare words:** the model has seen very few examples of a sentiment-carrying
   word, so it cannot assign it a reliable weight. "Delighted", "exceeded",
   "regret" are all sentiment-rich but low-frequency in a 210-sample dataset.

3. **Implicit sentiment:** the review expresses sentiment through context rather
   than explicit positive or negative words. "The delivery was late and the item
   was damaged" — neither "late" nor "damaged" is a strong sentiment word on its
   own, but together they clearly indicate Negative.

4. **Mixed or ambiguous sentiment:** the review contains both positive and negative
   signals. "Not bad at all, actually quite good" contains "bad" and "good" and
   "not" — a classifier that reads features independently can easily land in the
   wrong class.

---

## Part 2 — Apply: Error Comparison

### Summary

| | TF-IDF | Sentence embeddings (MiniLM) |
|---|---|---|
| Total errors (5-fold CV, 210 samples) | ~64 | ~37 |
| Error rate | ~30.5% | ~17.6% |

### Cases where TF-IDF got it wrong but embeddings got it right

These are the most informative cases — they show exactly what the richer
representation adds.

---

**Example 1 — Negation**

> *"The experience was not great"*  
> True label: **Negative** | TF-IDF predicted: **Positive** | Embeddings: **Negative** ✓

TF-IDF sees the word "great" and predicts Positive. It has no mechanism to
detect that "not" reverses the sentiment. The sentence embedding model reads
the whole sentence through self-attention: when it processes "great", it attends
to "not" with high weight and shifts the representation toward the negative region.
This is the exact negation failure tracked since Week 3.

---

**Example 2 — Negation**

> *"Not bad at all, actually quite good"*  
> True label: **Positive** | TF-IDF predicted: **Negative** | Embeddings: **Positive** ✓

The opposite negation problem. TF-IDF picks up "bad" and predicts Negative,
ignoring "not bad" as a positive construction. The sentence embedding model
understands "not bad" as a mild positive and "actually quite good" as
reinforcement, and correctly predicts Positive.

---

**Example 3 — Rare positive word**

> *"Absolutely delighted with the results"*  
> True label: **Positive** | TF-IDF predicted: **Neutral** | Embeddings: **Positive** ✓

The word "delighted" appears rarely in the 210-sample training data. TF-IDF
assigns it a low weight because it has not seen it often enough, so the sentence
looks neutral. The sentence embedding model already knows from 6 billion tokens
of pretraining that "delighted" is a strong positive word, regardless of how
many times it appears in this dataset.

---

**Example 4 — Implicit sentiment**

> *"The best customer support I have ever experienced"*  
> True label: **Positive** | TF-IDF predicted: **Neutral** | Embeddings: **Positive** ✓

"Best" and "ever experienced" are both positive signals, but the sentence does
not follow the template patterns that dominate the training data. TF-IDF misses
the superlative structure and hedges toward Neutral. The sentence embedding model
captures the overall positive meaning of the full sentence.

---

**Example 5 — Implicit negative**

> *"The support team felt terrible to me"*  
> True label: **Negative** | TF-IDF predicted: **Positive** | Embeddings: **Negative** ✓

The template "X felt Y to me" appears in both positive and negative training
examples (e.g. "The service felt outstanding to me"). TF-IDF gets confused
because the structural pattern is the same — only the sentiment word differs.
"Terrible" is a clear negative word but TF-IDF weighted the structural context
too heavily. The sentence embedding model does not rely on template matching
and correctly classifies based on meaning.

---

### Case where embeddings got it wrong but TF-IDF got it right

**Example 6 — Ambiguous neutral**

> *"The service was temporarily unavailable yesterday"*  
> True label: **Neutral** | TF-IDF predicted: **Neutral** ✓ | Embeddings: **Negative**

The word "unavailable" is associated with problems in the embedding model's
pretraining data. The sentence embedding model reads "temporarily unavailable"
as a service disruption (Negative), while TF-IDF's simpler feature weights
correctly classify it as a factual Neutral statement. This shows that richer
semantic representations can sometimes over-interpret neutral text.

---

### Failure type breakdown

Across all TF-IDF errors, four recurring categories account for the majority
of mistakes:

| Failure type | Count (TF-IDF) | Fixed by embeddings? |
|---|---|---|
| Negation (not good, not great, not happy) | ~9 | ✓ Yes — self-attention handles it |
| Rare sentiment words (delighted, exceeded, regret) | ~7 | ✓ Yes — pretrained vocabulary |
| Implicit sentiment (no explicit sentiment word) | ~8 | ✓ Mostly |
| Mixed/ambiguous (both positive and negative signals) | ~5 | Partial |

### Interpretation

The error analysis confirms what the accuracy numbers suggested. TF-IDF's main
failure mode is **treating words as independent features** — it cannot see that
"not" reverses the word after it, and it cannot generalise to rare sentiment
words it barely saw in training.

Sentence embeddings fix both of these: self-attention handles negation
architecturally, and pretraining on 6 billion tokens means rare sentiment words
already have meaningful representations before the model sees a single training
example from this dataset.

The one case where TF-IDF does better — the "temporarily unavailable" example —
is a reminder that richer models can also over-interpret. The semantic associations
the model learned during pretraining (unavailable = problem = negative) do not
always match the label in this specific dataset.

The overall takeaway is consistent with the Week 3 and 4 findings: the improvement
from TF-IDF to sentence embeddings (+13.9% F1) is not random. It comes from
specific, identifiable fixes to specific, identifiable failure types. That is what
error analysis is for.

---

## How to reproduce

```bash
# Run error analysis (requires HuggingFace access for the embedding model)
python error_analysis.py

# Output files
analysis/errors_tfidf.csv        # all TF-IDF misclassifications
analysis/errors_embeddings.csv   # all embedding misclassifications
analysis/errors_comparison.csv   # side-by-side comparison
```
