# Sentiment Analysis API

A FastAPI application that predicts the sentiment of input text using a trained machine learning model.

## Live API

https://raghad232-sentiment-analysis-api.hf.space

## Example

Health check:

```bash
curl https://raghad232-sentiment-analysis-api.hf.space/health
```

Prediction:

```bash
curl -X POST https://raghad232-sentiment-analysis-api.hf.space/predict \
-H "Content-Type: application/json" \
-d "{\"text\":\"I absolutely love this product.\"}"
```

## Monitoring

The API includes basic production monitoring through:

- `GET /health` – verifies that the service is running.
- `GET /metrics` – returns:
  - Total requests
  - Error count
  - Average latency
  - Low-confidence prediction count
  - Low-confidence rate
  - Prediction label distribution

Each prediction request is also written to structured logs including:

- Timestamp
- Request ID
- Input length
- Predicted label
- Confidence
- Low-confidence flag
- Response latency
