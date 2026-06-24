# Error Analysis

## What is Error Analysis?

Error analysis is the process of examining incorrect predictions to understand why a model failed. While accuracy and F1-score provide useful summary metrics, they do not explain the specific weaknesses of a model. By studying misclassified examples, we can identify recurring patterns and determine what improvements are needed.

## Confusion Matrix and Error Types

A confusion matrix shows the relationship between true labels and predicted labels.

Common error types:

* **False Positive:** The model predicts a label incorrectly (e.g., Positive instead of Negative).
* **False Negative:** The model fails to predict the correct label (e.g., Neutral instead of Positive).

Examining these errors helps reveal model biases and recurring failure patterns.

## Error Categories

The most common error categories observed were:

1. **Negation** – Sentiment is reversed by words such as "not".
2. **Sarcasm and Irony** – Positive words are used to express negative opinions.
3. **Implicit Sentiment** – Sentiment is implied through an experience rather than explicit sentiment words.
4. **Rare Vocabulary** – Uncommon sentiment expressions appear too infrequently for TF-IDF to learn effectively.

---

## Model Comparison

### TF-IDF Wrong, Embeddings Correct

**Example 1**

Review:

> "Oh fantastic, another update that breaks everything."

True Label: Negative

TF-IDF: Positive

Embeddings: Negative

Explanation: TF-IDF focused on the word "fantastic" while embeddings captured the sarcastic meaning of the full sentence.

---

**Example 2**

Review:

> "Spent more time on hold than actually using it."

True Label: Negative

TF-IDF: Neutral

Embeddings: Negative

Explanation: The negative sentiment is implied rather than expressed through explicit negative words.

---

**Example 3**

Review:

> "Can't believe they charge money for this."

True Label: Negative

TF-IDF: Neutral

Embeddings: Negative

Explanation: The embedding model captured the frustration expressed by the sentence, while TF-IDF missed the implied sentiment.

---

**Example 4**

Review:

> "Way better than the model I had before."

True Label: Positive

TF-IDF: Neutral

Embeddings: Positive

Explanation: Embeddings understood the comparative improvement while TF-IDF treated the sentence as mostly neutral.

### Embeddings Wrong, TF-IDF Correct

**Example 5**

Review:

> "Delivery was a day early, which never happens."

True Label: Positive

TF-IDF: Positive

Embeddings: Negative

Explanation: The embedding model associated "never happens" with negative contexts and misclassified the review.

---

## Failure Type Breakdown

| Failure Type       | Observation                                      |
| ------------------ | ------------------------------------------------ |
| Implicit Sentiment | Most common source of TF-IDF mistakes.           |
| Sarcasm and Irony  | Difficult for both models.                       |
| Negation           | Frequently caused incorrect TF-IDF predictions.  |
| Rare Vocabulary    | Embeddings handled these cases more effectively. |

## Conclusion

The error analysis showed that sentence embeddings made fewer mistakes than TF-IDF (37 vs 64 errors). Most of the improvement came from handling context-dependent language such as negation, sarcasm, implicit sentiment, and uncommon vocabulary. These findings explain why sentence embeddings achieved better overall performance on the dataset.
