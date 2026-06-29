

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict import predict_sentiment, _load_model

# ── app ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Returns a sentiment label and confidence score for any text input.",
    version="1.0.0",
)

@app.on_event("startup")
def load_model_on_startup():
    print("loading model at startup...")
    _load_model()
    print("model ready.")

class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse, status_code=200)
def health():
    """
    Health check endpoint.
    Returns {"status": "ok"} if the service is running and the model is loaded.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, status_code=200)
def predict(request: PredictRequest):
  
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail="text field must not be empty"
        )

    try:
        result = predict_sentiment(request.text)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )

    return result
