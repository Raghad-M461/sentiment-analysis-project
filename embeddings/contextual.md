# Contextual Embeddings : Learning Notes


## The Core Idea

### What is a contextual embedding?

In Week 3, I used GloVe embeddings, where each word has one fixed vector. For example, the word "good" always has the same representation regardless of the sentence it appears in. This makes GloVe a static embedding model.

Contextual embeddings work differently. The representation of a word changes depending on the surrounding words. For example, the word "good" in "the food was good" will have a different representation than "good" in "not good". The model considers the entire sentence before generating the embedding.

### Negation and word sense ambiguity

One limitation of static embeddings is negation. Sentences such as "the service was great" and "the service was not great" can appear very similar because averaging word vectors does not fully capture the effect of the word "not". Contextual embeddings solve this by considering the relationship between words when creating representations.

Another limitation is word-sense ambiguity. For example, the word "bank" can refer to a financial institution or the side of a river. Static embeddings use the same vector for both meanings, while contextual embeddings use the surrounding words to determine the correct meaning.

## Attention and Transformers

### Self-attention

Self-attention is the key idea behind transformer models. Instead of processing words independently, the model looks at all words in a sentence and decides which ones are most important for understanding the meaning.

For example, in the sentence "the service was not great", the model pays attention to both "great" and "not". This helps it understand that the positive meaning of "great" has been changed by the word "not".

### Why this helps with "not good"

When processing "not good", the model considers both words together. The representation of "good" is influenced by "not", allowing the model to understand the negative meaning of the phrase. This is something static embeddings cannot do because each word is represented independently.

### What is BERT?

BERT is a pretrained transformer model that learns language patterns from a large amount of text. Because it is pretrained, it can generate meaningful contextual embeddings without being trained from scratch for every task.

For this assignment, I used the `all-MiniLM-L6-v2` model from the `sentence-transformers` library. It is a lightweight transformer model designed to generate high-quality sentence embeddings efficiently.

## Reflection

In Week 3, I predicted that averaging GloVe vectors would cause problems with negation, and the results confirmed this. The model struggled with phrases such as "not good" because each word had a fixed representation and averaging removed important contextual information.

A contextual model handles this differently. Through self-attention, the model allows words to influence one another when building their representations. In the phrase "not good", the word "good" is interpreted together with the word "not", producing a representation that reflects the actual meaning of the phrase. This allows the model to distinguish between positive and negated sentences more effectively.

Looking back at my Week 3 results, the errors were mainly caused by the limitations of static embeddings rather than the classifier itself. Contextual embeddings address this limitation by incorporating context directly into the representation of each word.

## Prediction Before Running Part 2

### Static GloVe Results

* "good" vs "not good": ~0.91
* "the service was great" vs "the service was not great": ~0.94
* "I am happy with this product" vs "I am not happy with this product": ~0.98

### Prediction

I expect the contextual model to produce lower similarity scores for all three sentence pairs because it can capture the effect of negation. Unlike GloVe, which uses fixed word vectors, contextual embeddings consider the surrounding words and therefore should represent positive and negated sentences as more distinct from one another.

See `contextual_similarity.py` for the implementation and results.
