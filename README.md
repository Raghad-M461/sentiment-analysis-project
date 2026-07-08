## Monitoring

The API now includes basic production monitoring features.

### Endpoints

- `GET /health` – Service health check
- `GET /metrics` – Request statistics and monitoring metrics
- `POST /predict` – Sentiment prediction

The `/metrics` endpoint reports:

- Total requests
- Error count
- Average latency
- Low-confidence predictions
- Prediction label distribution
