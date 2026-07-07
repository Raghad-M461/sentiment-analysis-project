# Deployment Notes

## Local vs Cloud Deployment

Running the API locally means it is only accessible from my own computer using `localhost`.

Deploying to a hosting platform makes the API available through a public URL so anyone with the link can access it over the internet.

Cloud deployment also requires:
- A public URL
- Environment variables when needed (for configuration such as ports, API keys, or secrets)
- A start command to launch the application
- All required dependencies listed in `requirements.txt`

---

## Free Hosting Platform Requirements

For deployment on Hugging Face Spaces (Docker), the application requires:

- A Dockerfile
- A start command to launch the FastAPI application
- Listening on the correct application port
- Pinned dependencies in `requirements.txt`

---

## Model File

The trained model is included with the deployment (`app/model`).

Including the model with the application allows predictions without downloading the model every time the service starts. The trade-off is a larger deployment size, while downloading the model at startup would reduce the repository size but increase cold-start time.

---

## Deployment

Platform: Hugging Face Spaces (Docker)

Live API:

https://raghad232-sentiment-analysis-api.hf.space

Health endpoint:

GET https://raghad232-sentiment-analysis-api.hf.space/health

Response:

```json
{
  "status": "ok"
}
```

---

## Live Prediction Tests

### Test 1 – Positive

Request:

```json
{
  "text": "I absolutely love this product."
}
```

Response:

```json
{
  "label": "Positive",
  "confidence": 0.6333,
  "low_confidence": false
}
```

---

### Test 2 – Negation

Request:

```json
{
  "text": "This is not good at all."
}
```

Response:

```json
{
  "label": "Negative",
  "confidence": 0.7929,
  "low_confidence": false
}
```

---

### Test 3 – Mixed Sentiment

Request:

```json
{
  "text": "The service was great but the delivery was very slow."
}
```

Response:

```json
{
  "label": "Positive",
  "confidence": 0.5092,
  "low_confidence": true
}
```

---

### Test 4 – Arabic / Out-of-Domain

Request:

```json
{
  "text": "الخدمة كانت ممتازة"
}
```

Response:

```json
{
  "label": "Neutral",
  "confidence": 0.5771,
  "low_confidence": false
}
```

---

### Test 5

Request:

```json
{
  "text": "I don't hate it, but I wouldn't recommend it."
}
```

Response:

```json
{
  "label": "Negative",
  "confidence": 0.4795,
  "low_confidence": true
}
```

---

## Cold-Start Behavior

The model is loaded during the FastAPI startup event before the API becomes available. During testing, the application had already completed startup, so all `/predict` requests responded immediately. No noticeable delay was observed during the warm requests.

---

## Deployment Challenges

The initial deployment on Render failed because the application exceeded the free-tier memory limit while loading the SentenceTransformer model.

To resolve this issue, the application was deployed on Hugging Face Spaces using Docker, where the service started successfully and the API became publicly accessible.
