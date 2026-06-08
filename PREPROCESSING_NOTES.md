# Part 1 — Preprocessing Concepts

Reference notes for the `feature/preprocessing-pipeline` work. Each concept lists
what it does, why it helps in general, and its specific effect on **sentiment**
analysis (which is what our downstream task cares about).

## Lowercasing
Folds all characters to lower case so `Good`, `good`, and `GOOD` map to one
token. Shrinks the vocabulary and reduces data sparsity, which helps models
generalise from limited examples. Caveat for sentiment: ALL-CAPS often signals
emphasis or shouting ("TERRIBLE"), so lowercasing can erase an intensity cue.

## Tokenization
Splits raw text into units (tokens) — usually words and punctuation. It is the
foundation every later step depends on; nothing else can run until text is
tokenized. A good tokenizer handles contractions and punctuation correctly
(e.g. `don't` → `do` + `n't`), which directly affects negation handling.

## Stop-word removal
Removes high-frequency, low-information words (`the`, `is`, `at`, ...). Reduces
noise and dimensionality for topic-style tasks. **Risk for sentiment:** standard
stop lists contain `not`, `no`, `never`, `but`, `very` — words that carry or
modify polarity. Removing them blindly destroys signal, so a sentiment pipeline
must use a *sentiment-safe* list that keeps negators and intensifiers.

## Stemming
Chops words to a crude root by rule (`amazing` → `amaz`, `studies` → `studi`).
Fast and aggressive; collapses related forms but produces non-words and can
over-merge distinct meanings, blurring fine-grained polarity differences.

## Lemmatization
Reduces words to their dictionary base form using vocabulary and morphology
(`better` → `good`, `studies` → `study`). Slower than stemming but returns real
words and preserves meaning more faithfully — generally the safer normaliser for
sentiment.

## Handling punctuation
Punctuation is often stripped to clean the token stream. For sentiment this is a
trade-off: `!`, `?`, and emoticons encode affect and intensity (`great!!!`), so
indiscriminate removal can drop useful signal. We remove pure-punctuation tokens
but keep punctuation as a *boundary* cue for negation scope first.

## Handling negation terms
Negators (`not`, `no`, `never`, `n't`, ...) flip the polarity of nearby words:
`good` and `not good` are opposites, yet share every token after naive cleaning.
The standard fix (Pang, Lee & Vaithyanathan, 2002) is **negation marking**:
prefix every token from a negator up to the next clause boundary with `neg_`, so
`not good` → `not neg_good`. This keeps the negated and non-negated senses as
distinct features.

---

# Reflection (~150 words)

Tokenization is non-negotiable, and lowercasing plus lemmatization are usually
net positive: they cut vocabulary sparsity while keeping words intact, which
helps a sentiment model generalise from limited data. The single most valuable
sentiment-specific step is **negation handling** — without it, "not good" and
"good" collapse into the same features and the model loses the very distinction
it is meant to learn.

The techniques most likely to *reduce* quality are aggressive ones applied
without a sentiment lens. Generic **stop-word removal** deletes `not`, `never`,
`but`, and `very`, stripping out polarity and its modifiers. **Stemming** can
over-merge distinct words and emit non-words, blurring nuance. Blanket
**punctuation removal** discards intensity markers like `!!!` and emoticons. The
lesson: preprocessing is not "more cleaning is better." Each step trades
sparsity reduction against information loss, and for sentiment the signal often
lives in exactly the small words a naive pipeline throws away.
