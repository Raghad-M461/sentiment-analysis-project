# Sentiment Analysis API

A FastAPI-based sentiment analysis API that predicts whether a piece of text is **Positive**, **Negative**, or **Neutral** using a trained Logistic Regression model with sentence embeddings.

## Live Deployment

The API is publicly available at:

https://raghad232-sentiment-analysis-api.hf.space

## Health Check

Endpoint:

GET /health

Example response:

```json
{
  "status": "ok"
}
```

## Prediction Endpoint

Endpoint:

POST /predict

Example request:

```json
{
  "text": "I absolutely love this product."
}
```

Example response:

```json
{
  "label": "Positive",
  "confidence": 0.6333,
  "low_confidence": false
}
```

## API Documentation

Interactive Swagger documentation:

https://raghad232-sentiment-analysis-api.hf.space/docs

## Technologies Used

- Python 3.11
- FastAPI
- Uvicorn
- scikit-learn
- sentence-transformers
- NLTK
- NumPy

## Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api_app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## Deployment

The API is deployed on **Hugging Face Spaces (Docker)**.

Deployment includes:

- Public API URL
- Health endpoint (`/health`)
- Prediction endpoint (`/predict`)
- Interactive Swagger documentation (`/docs`)

The deployment was tested successfully using multiple prediction requests and the health endpoint.
