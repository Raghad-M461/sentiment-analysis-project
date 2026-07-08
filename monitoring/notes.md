# Monitoring Notes

## 1. Why `print()` is not logging

Using `print()` only writes plain text to the console. It does not automatically include timestamps, log levels, or a consistent format, making it difficult to monitor an application in production.

Python's `logging` module provides structured logging with timestamps, log levels, and formatted messages. This makes logs easier to search, troubleshoot, and analyze while the application is running.

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

If a validation or prediction error occurs, the application also records the error without interrupting the service. Logging is wrapped in `try/except` blocks so that logging failures never prevent the API from returning a prediction.

---

## 3. Why Monitoring is Important

Offline evaluation measures model performance before deployment, but it cannot show how the application behaves after it is live.

Monitoring helps answer important operational questions, including:

- Is the API available?
- How many requests have been processed?
- How many errors have occurred?
- What is the average response latency?
- Is the confidence distribution changing over time?

These metrics help identify performance issues, unexpected usage patterns, and potential model drift in production.

---

## 4. Low-Confidence Predictions

Predictions are marked as low confidence whenever the confidence score falls below the predefined threshold.

Monitoring the low-confidence rate is useful because an increasing rate may indicate:

- Inputs outside the model's training distribution
- Unsupported languages
- Ambiguous user input
- Possible model drift over time

During previous testing, Arabic text was identified as an out-of-domain case, making this metric a useful indicator for monitoring unsupported inputs after deployment.

---

# Live Deployment Metrics

After deploying the API to Hugging Face Spaces and generating live traffic, the `/metrics` endpoint returned:

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

These metrics confirm that the monitoring endpoint successfully tracked request statistics, response latency, prediction distribution, and low-confidence predictions during live testing.

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

After generating **16 live prediction requests**, the deployed API is operating normally. The service processed every request successfully with **0 errors**, indicating that the monitoring implementation is stable and reliable. The average response latency of **35.26 ms** shows that prediction requests were handled efficiently after startup. The monitoring system recorded **5 low-confidence predictions (31.25%)**, which is expected because the test set intentionally included mixed-sentiment, negation, and Arabic examples. The prediction distribution contained positive, negative, and neutral results, confirming that the API successfully handled different types of input. Overall, the collected metrics and structured logs indicate that the deployed service is healthy, responsive, and capable of providing useful operational information for monitoring, debugging, and future maintenance.
