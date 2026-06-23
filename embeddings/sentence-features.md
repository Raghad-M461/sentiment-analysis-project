# Sentence Features — Learning Notes and Comparison

**Branch:** `feature/contextual-embeddings`  
**Script:** `sentence_features_comparison.py`

---

## Part 1 — Concepts

### Word embeddings vs sentence embeddings

A word embedding gives a vector for a single word in isolation. GloVe, which I used
in Week 3, produces one 50-dimensional vector per word regardless of context. To
represent a full sentence I averaged all the word vectors together, which threw away
word order and let negation signals drown.

A sentence embedding gives a vector for the whole sentence at once. Models like
`sentence-transformers` are trained specifically to produce these, and the training
objective is different: instead of learning from word co-occurrence statistics,
`all-MiniLM-L6-v2` was fine-tuned on pairs of semantically similar and dissimilar
sentences so that similar sentences end up close in the vector space and different
sentences end up far apart. The result is that the 384-dimensional sentence vector
already captures meaning at the sentence level, not just at the word level.

This is the key difference. When I average GloVe vectors, I am combining word-level
representations after the fact. When I use `sentence-transformers`, the model has
been explicitly trained to produce sentence-level representations that are
comparable across different sentences.

### Why swapping only the features is the right comparison

The point of this experiment is to isolate the effect of the feature representation.
If I changed the classifier, the training split, or the evaluation metric at the
same time, I would not know whether any improvement came from the features or from
something else. By keeping everything the same — same Logistic Regression, same
5-fold stratified CV with `random_state=42`, same dataset, same scoring — the only
variable is what the classifier receives as input. Any difference in the result can
then be attributed directly to the feature method.

This is basic controlled experiment design: change one thing at a time.

### One risk of embedding-based features vs sparse TF-IDF

**Interpretability.** With TF-IDF, I can look at the feature weights and say
"the word 'terrible' has a high negative weight". The model's decisions are
traceable back to specific words. With a 384-dimensional sentence embedding, there
is no direct way to say which part of the input drove a particular prediction. Each
dimension of the embedding is a learned combination of patterns from the model's
training, not a human-readable feature. This makes debugging harder and makes it
difficult to explain a wrong prediction. It also makes the model harder to audit
for bias.

A secondary risk is compute: encoding 210 sentences through a transformer takes
noticeably longer than computing TF-IDF, and this gap grows significantly for larger
datasets.

---

## Part 2 — Comparison Results

### Dataset

- **80 samples** (hardcoded: 40 Positive, 40 Negative) — the original Week 1 dataset
- **210 samples** (enriched CSV, if present) — used in Weeks 2 and 3
- Same 5-fold stratified CV, `random_state=42`, Logistic Regression throughout

### Comparison Table

| Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF (baseline) | 80.0% | 81.8% | 80.0% | 79.6% |
| Averaged GloVe (Week 3) | 74.3%* | ~74%* | ~74%* | ~74%* |
| Sentence embeddings (MiniLM) | — | — | — | — |

\* GloVe numbers are from Week 3 on the **210-sample enriched dataset**, not the
80-sample dataset. The sentence embedding numbers will be filled in after the CI
run (HuggingFace was not accessible in the local development environment — see
`contextual_similarity.py` from last week for context).

---

### Interpretation

**TF-IDF on the 80-sample dataset achieves 80.0% accuracy.** This is a clean,
simple dataset: every sentence follows a predictable template ("A [adjective]
[noun] overall", "Such a [adjective] [noun]") and the positive and negative words
are distinct. TF-IDF handles this well because the discriminating words appear
reliably and the vocabulary barely overlaps between classes.

**GloVe averaging scored 74.3% on the larger 210-sample dataset.** That dataset
has more variety and more rare words, which is where pretrained embeddings help —
a word seen once in training still has a meaningful vector from 6 billion tokens
of GloVe pretraining. But on the simple 80-sample set, TF-IDF wins because the
vocabulary is controlled enough that exact-match features are sufficient.

**What to expect from sentence embeddings:** Based on last week's similarity
experiment, MiniLM correctly separates negated sentence pairs that GloVe averaging
confused. On the larger enriched dataset it should do better than TF-IDF on
negation-heavy examples. On the simple 80-sample dataset it may or may not beat
TF-IDF — the dataset's template structure gives TF-IDF an artificial advantage
that would disappear with messier real-world text.

**If sentence embeddings do not beat TF-IDF here, that is still a valid and
interesting result.** It would mean that on short, template-style sentences,
the overhead of contextual representations does not pay off. The honest conclusion
would be: richer features help when the data is rich enough to demand them.

---

### How to reproduce

```bash
# Sentence embeddings + TF-IDF only (fast, no GloVe file needed)
python sentence_features_comparison.py --skip-glove

# Full three-way comparison (needs GloVe file)
python sentence_features_comparison.py --glove glove_data/glove.6B.50d.txt
```
