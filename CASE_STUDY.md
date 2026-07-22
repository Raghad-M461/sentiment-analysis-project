# Sentiment Analysis Project – Case Study

## Problem

Describe the problem the project solves.

Example:
> Understanding customer sentiment from text reviews is valuable for businesses, but manually reading thousands of reviews is slow and inconsistent. The objective of this project was to build an automated sentiment analysis system capable of classifying text into **Negative**, **Neutral**, and **Positive** sentiments while exposing the model through a production-ready API.

---

## Approach

Explain the entire internship journey.

Talk about:

- Traditional TF-IDF baseline
- Frozen DistilBERT embeddings
- Fine-tuning DistilBERT
- FastAPI deployment
- Hugging Face deployment
- Evaluation on a frozen test set

---

## Key Technical Decisions

Explain why you chose:

- DistilBERT
- Hugging Face Transformers
- FastAPI
- Frozen test set
- Macro-F1
- Latency measurement

---

## Results

Include your real numbers.

| Metric | Result |
|---------|---------|
| Accuracy | 78.12% |
| Macro-F1 | 77.88% |
| Average Latency | 24.74 ms |
| Model Size | 256.11 MB |

Then mention:

- classification report
- confusion matrix
- negation evaluation

---

## Limitations

Be honest.

Mention things like:

- Small dataset
- Some negation mistakes
- Fine-tuning requires more resources than TF-IDF
- More data would improve performance

---

## Next Steps

Examples:

- Larger dataset
- Better hyperparameter tuning
- Quantization
- Docker deployment
- CI/CD
- Monitoring
