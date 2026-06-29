# API Notes

**Branch:** `feature/sentiment-api`

## What is a REST API?

A REST API allows different applications to communicate with each other over the internet using HTTP requests. Instead of running the `predict_sentiment()` function directly, another application can send text to the API and receive the prediction as a response.

An endpoint is a specific URL that performs a task. In this project:

* `GET /health` checks if the API is running.
* `POST /predict` accepts text and returns the predicted sentiment.

---

## JSON Request and Response

JSON is the standard format used to exchange data between applications because it is simple, lightweight, and supported by almost every programming language.

Example request:

```json
{
  "text": "Not bad, actually quite impressed with how well it performs."
}
```

Example response:

```json
{
  "label": "Positive",
  "confidence": 0.6647
}
```

The client always sends data in the same format, and the server always returns a structured JSON response.

---

## Why FastAPI?

FastAPI makes it easy to build web APIs in Python. It handles requests, validates input automatically, and returns JSON responses.

The model is loaded once when the server starts instead of loading it every time a request is made. This makes the API much faster because the model is already in memory.

---

## HTTP Status Codes

| Status Code                   | Meaning                                               |
| ----------------------------- | ----------------------------------------------------- |
| **200 OK**                    | The request was successful.                           |
| **400 Bad Request**           | The client sent invalid input.                        |
| **422 Unprocessable Entity**  | The request body is missing or contains invalid data. |
| **500 Internal Server Error** | An unexpected error occurred on the server.           |

---

# API Test Results

### Health Check

**Request**

```bash
curl http://localhost:8000/health
```

**Response**

```json
{
  "status": "ok"
}
```

---

### Test 1 – Positive

**Request**

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"The product quality is outstanding and the delivery was fast"}'
```

**Response**

```json
{
  "label": "Positive",
  "confidence": 0.6647
}
```

---

### Test 2 – Negation

**Request**

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"This is not good at all, very disappointed with the service"}'
```

**Response**

```json
{
  "label": "Negative",
  "confidence": 0.5712
}
```

---

### Test 3 – Mixed Sentiment

**Request**

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"The app works fine but the customer support could be better"}'
```

**Response**

```json
{
  "label": "Negative",
  "confidence": 0.54
}
```

---

### Test 4 – Negative

**Request**

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"Absolutely terrible experience, would not recommend to anyone"}'
```

**Response**

```json
{
  "label": "Negative",
  "confidence": 0.53
}
```

---

### Test 5 – Negation

**Request**

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"Not bad, actually quite impressed with how well it performs"}'
```

**Response**

```json
{
  "label": "Positive",
  "confidence": 0.6647
}
```
