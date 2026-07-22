# Fine-Tuning Evaluation Results

## Evaluation Metrics

| Metric | Value |
|--------|------:|
| Accuracy | 0.7812 |
| Macro-F1 | 0.7788 |
| Average Inference Latency | 24.74 ms |
| Model Size | 256.11 MB |

## Classification Report

```text
              precision    recall  f1-score   support

    Negative     0.7000    0.7000    0.7000        10
     Neutral     0.9091    0.9091    0.9091        11
    Positive     0.7273    0.7273    0.7273        11

    accuracy                         0.7812        32
   macro avg     0.7788    0.7788    0.7788        32
weighted avg     0.7812    0.7812    0.7812        32
```

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Negation Test Results

| Sentence | Prediction |
|----------|------------|
| not good | Negative |
| not bad | Negative |
| not bad at all | Positive |
| not terrible | Negative |
| not amazing | Negative |
| The service was not good. | Positive |
| The product was not bad. | Negative |
| I do not like this. | Negative |
| I am not disappointed. | Negative |
| This is not the best experience. | Positive |

## Negation Discussion

The fine-tuned model correctly handled some negation examples but remained inconsistent on others. Overall, the evaluation suggests partial improvement in negation handling, while additional training data and comparison with the baseline model would be needed to confirm consistent improvement.
