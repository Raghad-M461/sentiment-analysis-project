# Testing Notes

## Model Evaluation vs. System Testing

Model evaluation measures how accurate the model's predictions are by comparing them with the correct labels. It uses metrics such as accuracy and F1-score to evaluate how well the model performs on unseen data.

System testing focuses on the behaviour of the complete service. It checks whether the API is fast, stable, consistent, and able to handle repeated or concurrent requests without errors.

Both are important. A highly accurate model is not useful if the service is unreliable, while a stable service has little value if the predictions are incorrect.

## Key Metrics

- **Latency:** The time it takes the service to respond to a request. Average latency shows the typical response time, while **p95 latency** shows how long the slowest 5% of requests take.
- **Throughput:** The number of requests the service can process per second.
- **Error Rate:** The percentage of valid requests that fail. A reliable service should keep this close to 0%.

## Why Regression Tests Matter

Regression tests ensure that known inputs continue to produce the expected outputs after code or model changes. They are written using **pytest** and run automatically through GitHub Actions. If a future update changes the expected behaviour, the tests fail and help detect the problem before the changes are merged.
