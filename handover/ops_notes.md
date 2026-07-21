# Handover: Hardening & Operational Logging in Production

## 1. Why Structured Logging Beats `print()`

Python's `logging` module is more suitable for production than `print()` because it supports log levels (INFO, WARNING, ERROR, CRITICAL), automatically adds timestamps, produces searchable logs, and can write output to log files. These features make it easier to monitor, debug, and maintain a deployed service.

---

## 2. What Should Be Logged Per Request

Each prediction request should record:

- Input length
- Predicted label
- Confidence score
- Request latency
- Validation or runtime errors

Logging this information helps developers monitor API behavior, troubleshoot issues, and evaluate model performance during real-world operation.

---

## 3. Operational Monitoring

Offline evaluation metrics such as accuracy, precision, recall, and F1-score measure model performance before deployment. Operational monitoring focuses on the health of the running service by tracking:

- Uptime
- Request count
- Error rate
- Average latency
- Low-confidence predictions (confidence drift)

These metrics help identify operational issues that offline evaluation cannot detect.

---

## 4. Sample Runtime Logs

```text
INFO: Request received (input_length=42)
INFO: Prediction=Positive confidence=0.97 latency=51ms
WARNING: Validation error: Missing text field
```

---

## 5. Health Snapshot

During live testing, the deployed sentiment analysis API operated reliably and maintained stable performance. The monitoring system successfully tracked request activity, response latency, validation events, and prediction confidence while the application continued serving requests without interruption. Structured logging provided clear visibility into API execution, making it easier to trace requests and identify operational events. The runtime metrics offered a concise overview of the service's health and complemented the detailed logs collected during testing. Overall, the implemented logging and monitoring features improve the observability, maintainability, and reliability of the deployed application, making it better prepared for production use and future monitoring.
