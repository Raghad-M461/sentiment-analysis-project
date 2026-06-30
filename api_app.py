
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from predict import predict_sentiment, _load_model

MAX_TEXT_LENGTH          = 1000   # characters
LOW_CONFIDENCE_THRESHOLD = 0.55   # below this, flag the prediction


# ── app ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Returns a sentiment label and confidence score for any text input.",
    version="1.1.0",
)

@app.on_event("startup")
def load_model_on_startup():
    print("loading model at startup...")
    _load_model()
    print("model ready.")

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
       
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        if len(value) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"text exceeds maximum length of {MAX_TEXT_LENGTH} characters "
                f"(got {len(value)})"
            )
        return value


class PredictResponse(BaseModel):
    label: str
    confidence: float
    low_confidence: bool = False


class HealthResponse(BaseModel):
    status: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    first_error = exc.errors()[0]
    field = ".".join(str(x) for x in first_error["loc"] if x != "body")
    message = first_error["msg"]
    return JSONResponse(
        status_code=422,
        content={"detail": f"{field}: {message}" if field else message},
    )

@app.get("/health", response_model=HealthResponse, status_code=200)
def health():
    """Returns {"status": "ok"} if the service is running and the model is loaded."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, status_code=200)
def predict(request: PredictRequest):
   
    try:
        result = predict_sentiment(request.text)
    except FileNotFoundError as e:
      
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
       
        raise HTTPException(status_code=500, detail=f"prediction failed: {str(e)}")

    result["low_confidence"] = result["confidence"] < LOW_CONFIDENCE_THRESHOLD
    return result
