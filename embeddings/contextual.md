# Contextual Embeddings — Learning Notes

**Branch:** `feature/contextual-embeddings`  
**Task:** Week 4 — From Static to Contextual Representations

---

## The Core Idea

### What "contextual embedding" means

In Week 3 I used GloVe, where every word gets one fixed vector that never changes.
The word "good" always has the same 50-dimensional vector, no matter what sentence
it is in. That is what "static" means.

A contextual embedding does the opposite: the vector for a word is computed fresh
for every sentence, taking into account the words around it. The word "good" in
"the food was good" gets a different vector than "good" in "not good at all". The
model looks at the whole sentence before deciding what each word means.

This matters enormously because meaning is not fixed — it depends on context. My
Week 3 results showed this directly: `cos(not, good) = 0.823`, meaning GloVe
thinks "not" and "good" are almost neighbours. It has no choice, because "not"
always has the same vector regardless of what comes after it.

### Negation and word-sense ambiguity

Two specific problems that static embeddings cannot solve:

**Negation:** "The service was great" and "The service was not great" produced
static similarity scores above 0.94 in my GloVe experiments. The averaged vectors
look almost identical because "not" is just one weak vector being averaged with
"service", "was", and "great". The positive pull of "great" dominates completely.
A contextual model reads the whole sentence before assigning vectors, so "great"
in the second sentence gets a representation that has been influenced by the
presence of "not" earlier in the sequence.

**Word-sense ambiguity:** The word "bank" has one GloVe vector that is some
average of "river bank" and "financial bank" — a blurry compromise that captures
neither meaning well. A contextual model gives "bank" a different vector in each
sentence, so "I deposited money at the bank" and "I sat by the river bank" produce
genuinely different representations for the same word.

---

## Attention and Transformers

### The self-attention intuition

The transformer architecture works through self-attention. The idea is simple to
state: when building the representation for a word, the model looks at every other
word in the sentence and decides how much to "pay attention" to each one.

For the sentence "the service was not great", when the model processes the word
"great", it does not look at "great" in isolation. It computes attention weights
over all five words and asks: which of these other words should influence what
"great" means here? The word "not" will receive a high attention weight from
"great" because the model has learned, from billions of training examples, that
negation words directly modify the sentiment of the following word. The resulting
vector for "great" is then a weighted mixture that leans toward the negative region
of the space.

This is the mechanism that static embeddings cannot replicate. GloVe never looks
at surrounding words at inference time — it just returns a fixed lookup table
entry.

### Why this helps "good" in "not good"

In "not good", the contextual model processes both tokens together. When it builds
the representation for "good", it attends to "not" with high weight. The output
vector for "good" in this context is pulled away from the pure positive-sentiment
region toward a negative-leaning area. The final sentence representation therefore
reflects that the positive word has been negated — something averaging can never
do because it treats every word as equally independent.

### What BERT is

BERT (Bidirectional Encoder Representations from Transformers) is a large pretrained
transformer model published by Google in 2018. It has 12 layers of self-attention
and 110 million parameters. It was trained on the full English Wikipedia and a
large book corpus — around 3.3 billion words — using two tasks: predicting masked
words, and predicting whether one sentence follows another.

The key word is "pretrained": BERT learned rich contextual representations from
all that data before you ever touch it. You load the model and immediately get
meaningful sentence vectors for free, just as I loaded GloVe last week. The
difference is that BERT's representations are contextual: the same word gets a
different vector in every sentence.

`sentence-transformers` is a library that wraps BERT-style models and fine-tunes
them specifically to produce good sentence-level vectors. The model used in this
task, `all-MiniLM-L6-v2`, is a smaller 6-layer version (22M parameters) optimised
for fast inference while maintaining strong semantic similarity performance.

---

## Reflection (~150 words)

In Week 3 I predicted that averaging GloVe vectors would hurt negation cases, and
it did — the embedding model got all four negation test sentences wrong. The
specific reason was that "not" has one fixed vector that sits near positive words
(`cos(not, good) = 0.823`), and averaging it with "good" still produces a vector
in the positive region of the space. The negation information simply drowns.

A contextual model fixes this at the architecture level, not through clever
preprocessing. When the model processes "not good", the self-attention mechanism
lets "good" look at "not" and adjust its own representation accordingly. The
output vector for "good" in that sentence has been shifted by attending to the
negation word — it is no longer the same as the "good" vector from "very good" or
"quite good". The sentence vector then reflects the actual meaning rather than a
blurry average of the individual word meanings.

The key difference is not that contextual models know what "not" means in isolation.
It is that they know what "not" does to the word immediately after it, because they
have seen billions of examples of that pattern during pretraining.

---

## Prediction Before Running Part 2

**Static GloVe (from last week, real measured values):**
- `"good"` vs `"not good"` (word vector vs averaged pair): ~0.91
- `"the service was great"` vs `"the service was not great"`: ~0.94
- `"I am happy with this product"` vs `"I am not happy..."`: ~0.98

**Prediction for contextual model:**  
I expect all three pairs to show substantially lower cosine similarity — somewhere
in the 0.40–0.65 range. The reason: a trained contextual model has learned from
billions of sentences that negation flips sentiment, so the sentence vectors for
the positive and negated versions should land in very different regions of the
embedding space, not near each other as they did with GloVe averaging.

See `contextual_similarity.py` for the results.
