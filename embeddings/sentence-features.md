# Sentence Features — Learning Notes and Comparison

## Part 1 — Concepts

### Word Embeddings vs Sentence Embeddings

A word embedding represents a single word as a vector. For example, GloVe gives one fixed vector for each word regardless of where it appears. To represent a sentence, the word vectors must be combined, usually by averaging them.

A sentence embedding represents the meaning of an entire sentence using a single vector. Models such as `sentence-transformers` are trained to produce sentence vectors that can be compared directly, so sentences with similar meanings are placed closer together in the embedding space.

### Why Only the Features Were Changed

The goal of this experiment was to measure the effect of the feature representation. To make a fair comparison, I kept the same dataset, the same Logistic Regression classifier, and the same 5-fold cross-validation setup. The only change was the input features (TF-IDF, averaged GloVe, and sentence embeddings).

This ensures that any difference in performance comes from the feature representation rather than from changes to the model or evaluation process.

### One Risk of Sentence Embeddings

One disadvantage of sentence embeddings is that they are harder to interpret than TF-IDF features. With TF-IDF, it is easy to see which words influenced a prediction. With sentence embeddings, the features are dense numerical vectors, making it more difficult to understand why a prediction was made.

Another disadvantage is that generating sentence embeddings requires more computation than TF-IDF because the text must pass through a transformer model before classification.

---

## Part 2 — Comparison Results

### Experimental Setup

* Dataset: 210 labeled reviews
* Classifier: Logistic Regression
* Evaluation: 5-fold Stratified Cross-Validation
* Random State: 42
* Feature representations compared:

  * TF-IDF
  * Averaged GloVe embeddings
  * Sentence embeddings (all-MiniLM-L6-v2)

### Comparison Table

| Features                     | Accuracy | Precision | Recall | F1    |
| ---------------------------- | -------- | --------- | ------ | ----- |
| TF-IDF (baseline)            | 69.5%    | 71.1%     | 69.5%  | 68.2% |
| Averaged GloVe (Week 3)      | 74.3%    | 74.0%     | 74.3%  | 73.8% |
| Sentence Embeddings (MiniLM) | 82.4%    | 82.9%     | 82.4%  | 82.1% |

### Interpretation

The sentence embedding approach achieved the best results across all evaluation metrics. Compared with TF-IDF and averaged GloVe vectors, MiniLM produced higher accuracy, precision, recall, and F1 scores.

This suggests that contextual sentence embeddings capture more useful information about the meaning of each review than the other feature representations. Unlike TF-IDF, which relies mainly on word frequency, sentence embeddings consider the overall context of the sentence. They also performed better than averaged GloVe vectors, which lose information about word order and context when the word vectors are averaged.

Overall, sentence embeddings were the most effective feature representation on this dataset and provided the strongest classification performance while using the same classifier and evaluation setup.

