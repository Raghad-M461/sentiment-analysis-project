# Sentiment Analysis Project


This project uses a Sentence Transformer (`all-MiniLM-L6-v2`) with a Logistic Regression classifier to predict the sentiment of text. The model is also available through a simple FastAPI web service.

---

## Running the API

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Train and save the model (only needed the first time):

```bash
python train_and_save.py
```

Start the API server:

```bash
uvicorn api_app:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```
http://localhost:8000
```

Interactive API documentation:

```
http://localhost:8000/docs
```

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"Not bad, actually quite impressed with how well it performs"}'
```

Example response:

```json
{
  "label": "Positive",
  "confidence": 0.6647
}
```

---

## API Endpoints

| Method | Endpoint   | Description                               |
| ------ | ---------- | ----------------------------------------- |
| GET    | `/health`  | Returns the API status.                   |
| POST   | `/predict` | Predicts the sentiment of the input text. |

Example health check:

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "ok"
}
```
