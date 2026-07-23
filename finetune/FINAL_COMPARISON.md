# Final Model Comparison

## Model Comparison

| Model | Accuracy | Macro-F1 | Negative Recall | Neutral Recall | Positive Recall | Model Size (MB) | Latency (ms) |
|-------|---------:|---------:|----------------:|---------------:|----------------:|----------------:|-------------:|
| TF-IDF + Logistic Regression | 59.38% | 55.33% | 30.00% | 100.00% | 45.00% | 0.05 | 0.99 |
| MiniLM Embeddings + Logistic Regression | 78.13% | 77.22% | 70.00% | 100.00% | 64.00% | 87.35 | 10.20 |
| Fine-tuned DistilBERT | 84.38% | 84.24% | 80.00% | 91.00% | 82.00% | 256.11 | 23.44 |

---

## Negation Handling

The three models were evaluated using the same negation examples.

- **TF-IDF** struggled with negated expressions because it relies on word frequencies and has limited contextual understanding.
- **MiniLM** handled most negation examples correctly by using contextual sentence embeddings.
- **Fine-tuned DistilBERT** produced the most consistent predictions and showed the strongest understanding of negation and contextual meaning.

---

## Deployment Recommendation

Fine-tuned DistilBERT is the recommended model for deployment.

It achieved the highest Accuracy (84.38%) and Macro-F1 (84.24%) while also providing the most balanced recall across the three sentiment classes. Compared with MiniLM, DistilBERT improved overall prediction quality and handled contextual expressions such as negation more consistently.

Although DistilBERT has the largest model size (256.11 MB) and the highest inference latency (23.44 ms), the latency remains suitable for an online sentiment analysis service and is justified by the improvement in classification performance.

MiniLM offers a good compromise between accuracy, latency, and model size. It is a reasonable option when computational resources are limited or faster inference is required.

TF-IDF remains the smallest and fastest model, but its lower accuracy and weak recall for negative sentiment make it less suitable for production deployment.

Considering prediction quality, latency, model size, and maintainability, Fine-tuned DistilBERT provides the best overall trade-off and is the recommended production model for this project.

---

## What Surprised Me

The largest improvement came from evaluating all three models on the same frozen test set. MiniLM remained a strong baseline, but Fine-tuned DistilBERT consistently achieved better overall performance and handled contextual language more effectively. TF-IDF remained extremely fast, but its performance dropped significantly on more difficult sentiment examples, highlighting the limitations of traditional text representations.
