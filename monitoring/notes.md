# Monitoring Notes

## 1. Why print() is not logging

Using `print()` only writes text to the console. It does not provide timestamps, log levels, or a consistent format, which makes it difficult to monitor an application in production.

Python's `logging` module provides structured logging with useful information such as timestamps, log levels, and formatted messages. This makes it much easier to search logs, troubleshoot problems, and monitor the behavior of a deployed service.

---

## 2. Information Logged for Each Prediction

The `/predict` endpoint records the following information for every request:

- Timestamp
- Request ID
- Input text length
- Predicted sentiment label
- Prediction confidence
- Low-confidence flag
- Response latency (milliseconds)

If an error occurs during prediction or validation, the service also records the error without crashing the application.

---

## 3. Why Monitoring is Important

Offline evaluation measures model accuracy before deployment, but it cannot tell us how the service behaves after it is running.

Monitoring allows us to answer questions such as:

- Is the API currently running?
- How many requests has it processed?
- How many errors have occurred?
- What is the average response time?
- Are confidence scores becoming lower over time?

This information helps identify performance problems and unexpected model behavior in production.

---

## 4. Low-Confidence Predictions

The API marks predictions as low confidence when the confidence score is below the defined threshold.

Monitoring the low-confidence rate is useful because an increasing rate may indicate:

- Inputs outside the training distribution
- Unsupported languages
- Ambiguous text
- Possible model drift

During earlier testing, Arabic text was identified as an out-of-domain case, making this metric especially useful for detecting unsupported inputs.

---

# Live Deployment Metrics

After deploying the API on Hugging Face Spaces and generating live traffic, the `/metrics` endpoint returned:

```json
{
  "total_requests": 16,
  "error_count": 0,
  "low_confidence_count": 5,
  "low_confidence_rate": 0.3125,
  "average_latency_ms": 35.26,
  "label_counts": {
    "Positive": 7,
    "Negative": 8,
    "Neutral": 1
  }
}
```

These results show that the monitoring endpoint correctly tracked request statistics, prediction distribution, latency, and low-confidence predictions.

---

# Sample Structured Log Lines

The following log entries were collected from the live Hugging Face deployment after sending prediction requests:

```text
2026-07-08T06:59:52 INFO event=prediction_request request_id=b734b861 text_length=24 label=Negative confidence=0.7929 low_confidence=False latency_ms=185.6

2026-07-08T07:00:17 INFO event=prediction_request request_id=d0e17fc5 text_length=53 label=Positive confidence=0.5092 low_confidence=True latency_ms=23.6

2026-07-08T07:00:41 INFO event=prediction_request request_id=9f630d42 text_length=18 label=Neutral confidence=0.5771 low_confidence=False latency_ms=35.9

2026-07-08T07:01:04 INFO event=prediction_request request_id=2953a567 text_length=45 label=Negative confidence=0.6939 low_confidence=False latency_ms=32.4

2026-07-08T07:01:22 INFO event=prediction_request request_id=8e67ddbe text_length=19 label=Positive confidence=0.7086 low_confidence=False latency_ms=21.8
```

---

# Health Snapshot

The deployed API is currently operating normally. During testing, the service successfully processed 16 prediction requests without any errors. The average response time was approximately 35.26 milliseconds, indicating fast responses after startup. The model produced a balanced distribution of positive, negative, and neutral predictions across the test requests. Five predictions were marked as low confidence, resulting in a low-confidence rate of 31.25%. This metric is valuable because it highlights uncertain or out-of-domain inputs that may require additional review. Overall, the monitoring results show that the service is stable, responsive, and capable of tracking important operational metrics while providing useful diagnostic information for future maintenance.
