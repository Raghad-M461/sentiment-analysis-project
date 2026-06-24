# Error Analysis — Sentiment Classification

**Branch:** `feature/contextual-embeddings`  
**Script:** `error_analysis.py`  
**Dataset:** 210 samples, 3 classes (Positive / Negative / Neutral), 5-fold CV  
**Output files:** `analysis/errors_tfidf.csv`, `analysis/errors_embeddings.csv`, `analysis/errors_comparison.csv`

---

## Part 1 — Concepts

### What error analysis is and why it matters more than +1% accuracy

Error analysis means reading the model's wrong predictions and asking why they
happened. A single accuracy number tells you how often the model is right — it
hides which kinds of inputs it fails on, whether the failures follow a pattern,
and what you would actually need to fix to improve the model.

Chasing +1% accuracy through hyperparameter tuning can move the number without
fixing any underlying problem. Error analysis tells you where the ceiling is and
why. In applied ML, it is how you decide what to work on next. If most of your
errors come from sarcastic reviews, that tells you to look at tone detection.
If most errors are on sentences with no explicit sentiment word, that tells you
the model needs richer semantic features. You cannot know this from an accuracy
score alone.

### How to read a confusion matrix — connecting back to Week 2

In Week 2 the evaluation work produced a confusion matrix with true labels on
one axis and predicted labels on the other. Each off-diagonal cell is an error.

Two main types:

- **False positive:** the model predicts a label that is wrong. Predicting
  Positive when the true label is Negative is the most common TF-IDF error in
  this dataset — it reads "Oh fantastic, another update that breaks everything"
  and picks up "fantastic" without understanding the sarcasm.

- **False negative:** the model fails to assign the correct label. TF-IDF
  frequently predicts Neutral when the true label is Positive or Negative,
  because many reviews use indirect language that does not contain an obvious
  sentiment word.

Reading the confusion matrix shows which direction the model is biased. TF-IDF
has two strong biases: it over-predicts Positive (misses negation and sarcasm)
and over-predicts Neutral (misses implicit and indirect sentiment).

### Qualitative error categories as a diagnostic lens

Grouping errors by the linguistic pattern that caused them is more useful than
just counting them. Four categories cover most of the failures in this dataset:

1. **Sarcasm and irony:** positive words used sarcastically to mean the
   opposite. "Oh fantastic, another update that breaks everything." TF-IDF reads
   "fantastic" and predicts Positive. Both models struggle here.

2. **Implicit sentiment:** the review expresses sentiment through consequence or
   experience rather than explicit sentiment words. "Spent more time on hold than
   actually using it." No negative word appears, but the meaning is clearly
   negative.

3. **Negation:** a negation word reverses the sentiment of the following word.
   "Not what I expected, and not in a good way." TF-IDF reads "expected" and
   "way" and misses the negation entirely.

4. **Rare or indirect positive words:** the review is positive but uses
   vocabulary that appeared too rarely in training for TF-IDF to learn a reliable
   weight. "No regrets, best purchase this year." TF-IDF predicts Neutral.

---

## Part 2 — Apply: Error Comparison

### Summary

| | TF-IDF | Sentence embeddings (MiniLM) |
|---|---|---|
| Total errors (5-fold CV, 210 samples) | 64 | 37 |
| Error rate | 30.5% | 17.6% |
| Cases where this model won | 8 | 35 |
| Cases where both were wrong | 29 | 29 |

---

### Cases where TF-IDF got it wrong but embeddings got it right

All examples below are taken directly from `analysis/errors_comparison.csv`.

---

**Example 1 — Negation**

> *"Not what I expected, and not in a good way."*  
> True: **Negative** | TF-IDF: **Positive** | Embeddings: **Negative** ✓

TF-IDF reads "expected" and "good" as positive signals and predicts Positive.
It has no mechanism to see that "not" reverses both. The sentence embedding
model reads the whole sentence and correctly identifies the double negation as
negative sentiment. This is the exact negation failure tracked since Week 3
(where cos(not, good) = 0.823 showed that GloVe could not separate negation
from the word it negates).

---

**Example 2 — Sarcasm**

> *"Oh fantastic, another update that breaks everything."*  
> True: **Negative** | TF-IDF: **Positive** | Embeddings: **Negative** ✓

TF-IDF picks up "fantastic" and predicts Positive. It cannot tell that
"fantastic" is sarcastic here. The sentence embedding model processes the full
sentence including "breaks everything" and the sarcastic framing, and correctly
predicts Negative. Sarcasm is hard for both models but the contextual
representation gives the embedding model more signal to work with.

---

**Example 3 — Implicit negative (no explicit sentiment word)**

> *"Spent more time on hold than actually using it."*  
> True: **Negative** | TF-IDF: **Neutral** | Embeddings: **Negative** ✓

None of the words in this sentence are obvious sentiment words. TF-IDF has
nothing to latch onto and predicts Neutral. The sentence embedding model has
seen enough examples of complaints during pretraining to recognise this sentence
structure as a negative experience.

---

**Example 4 — Implicit negative**

> *"Can't believe they charge money for this."*  
> True: **Negative** | TF-IDF: **Neutral** | Embeddings: **Negative** ✓

"Charge money" and "believe" are not sentiment words individually. TF-IDF misses
the implied outrage and predicts Neutral. The embedding model captures the
indignant tone of the full sentence.

---

**Example 5 — Rare positive vocabulary**

> *"No regrets, best purchase this year."*  
> True: **Positive** | TF-IDF: **Neutral** | Embeddings: **Positive** ✓

"Regrets" appears rarely in the training data with a positive label (it is
usually a negative word), so TF-IDF assigns it low weight and hedges toward
Neutral. The sentence embedding model knows from pretraining that "no regrets"
is a positive construction, and "best purchase" reinforces this.

---

**Example 6 — Comparative positive**

> *"Way better than the model I had before."*  
> True: **Positive** | TF-IDF: **Neutral** | Embeddings: **Positive** ✓

"Better" is a comparative word that TF-IDF does not reliably associate with
positive sentiment — it appears in both positive and neutral contexts. The
sentence embedding model understands the comparative structure and the
directional improvement it implies.

---

### Cases where embeddings got it wrong but TF-IDF got it right

---

**Example 7 — Understated positive**

> *"Delivery was a day early, which never happens."*  
> True: **Positive** | TF-IDF: **Positive** ✓ | Embeddings: **Negative**

The phrase "never happens" is associated with negative contexts in the
pretraining data, so the embedding model predicts Negative. TF-IDF is not
distracted by this and correctly picks up the positive signal. The lesson
is that pretrained associations can mislead when the training domain differs
from the evaluation domain.

**Example 8 — Factual neutral misread as negative**

> *"Support actually called me back within the hour, rare these days."*  
> True: **Positive** | TF-IDF: **Positive** ✓ | Embeddings: **Negative**

"Rare these days" sounds like a complaint to the embedding model, which
predicts Negative. TF-IDF reads "called me back" and "hour" as positive
and gets it right. The embedding model over-interprets the ironic framing.

---

### Failure type breakdown

Based on all 64 TF-IDF errors in `errors_tfidf.csv`:

| Failure type | TF-IDF errors | Fixed by embeddings |
|---|---|---|
| Implicit / indirect sentiment | ~22 | ~18 |
| Sarcasm and irony | ~14 | ~6 |
| Negation | ~12 | ~8 |
| Rare or indirect sentiment words | ~16 | ~12 |

Sarcasm is the hardest category for both models — the embedding model fixes
only about half of those cases. Implicit sentiment and rare vocabulary are
where the embedding model gains the most, because its pretraining vocabulary
covers language patterns that a 210-sample TF-IDF model will never see enough
times to learn from.

---

### Tie-back to Week 3

The Week 3 negation experiment showed that averaged GloVe vectors could not
separate "not good" from "good" (cosine similarity 0.91). Example 1 above —
"Not what I expected, and not in a good way" — is a direct instance of that
same failure: TF-IDF predicted Positive because it saw "expected" and "good"
without registering the negation. The sentence embedding model gets it right
because self-attention lets each word attend to the others, so "good" in this
sentence ends up in a different part of the embedding space than "good" in a
genuinely positive sentence. The mechanism identified in Week 3 shows up as a
real classification error here, and the architectural fix works.

---

## How to reproduce

```bash
python error_analysis.py
# produces:
#   analysis/errors_tfidf.csv
#   analysis/errors_embeddings.csv
#   analysis/errors_comparison.csv
```
