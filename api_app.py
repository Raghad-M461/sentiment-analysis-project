

import logging
import time
import uuid
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from predict import predict_sentiment, _load_model

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":%(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%S.000Z",
)
logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH          = 1000
LOW_CONFIDENCE_THRESHOLD = 0.55

_metrics = {
    "total_requests":       0,
    "error_count":          0,
    "low_confidence_count": 0,
    "total_latency_ms":     0.0,
    "label_counts":         defaultdict(int),
}

app = FastAPI(
    title="Sentiment Analysis API",
    description="Returns a sentiment label and confidence score for any text input.",
    version="1.2.0",
)


@app.on_event("startup")
def load_model_on_startup():
    logger.info('"loading model at startup"')
    _load_model()
    logger.info('"model ready"')


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


class MetricsResponse(BaseModel):
    total_requests: int
    error_count: int
    low_confidence_count: int
    low_confidence_rate: float
    average_latency_ms: float
    label_counts: dict

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    _metrics["error_count"] += 1
    first_error = exc.errors()[0]
    field = ".".join(str(x) for x in first_error["loc"] if x != "body")
    message = first_error["msg"]
    detail = f"{field}: {message}" if field else message
    logger.warning(f'"validation_error","detail":"{detail}"')
    return JSONResponse(status_code=422, content={"detail": detail})

@app.get("/health", response_model=HealthResponse, status_code=200)
def health():
    """Returns {"status":"ok"} if the service is running."""
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse, status_code=200)
def metrics():
    
    total = _metrics["total_requests"]
    avg_latency = (
        _metrics["total_latency_ms"] / total if total > 0 else 0.0
    )
    low_conf_rate = (
        _metrics["low_confidence_count"] / total if total > 0 else 0.0
    )
    return {
        "total_requests":       total,
        "error_count":          _metrics["error_count"],
        "low_confidence_count": _metrics["low_confidence_count"],
        "low_confidence_rate":  round(low_conf_rate, 4),
        "average_latency_ms":   round(avg_latency, 2),
        "label_counts":         dict(_metrics["label_counts"]),
    }


@app.post("/predict", response_model=PredictResponse, status_code=200)
def predict(request: PredictRequest):
   
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()

    try:
        result = predict_sentiment(request.text)
    except FileNotFoundError as e:
        _metrics["error_count"] += 1
        logger.error(f'"server_error","request_id":"{request_id}","detail":"{e}"')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        _metrics["error_count"] += 1
        logger.error(f'"server_error","request_id":"{request_id}","detail":"{e}"')
        raise HTTPException(status_code=500, detail=f"prediction failed: {str(e)}")

    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)
    low_conf   = result["confidence"] < LOW_CONFIDENCE_THRESHOLD
    result["low_confidence"] = low_conf

    _metrics["total_requests"]   += 1
    _metrics["total_latency_ms"] += latency_ms
    _metrics["label_counts"][result["label"]] += 1
    if low_conf:
        _metrics["low_confidence_count"] += 1


    try:
        logger.info(
            f'"request_id":"{request_id}",'
            f'"text_length":{len(request.text)},'
            f'"label":"{result["label"]}",'
            f'"confidence":{result["confidence"]},'
            f'"low_confidence":{str(low_conf).lower()},'
            f'"latency_ms":{latency_ms}'
        )
    except Exception:
        # logging must never crash the prediction response
        pass

    return result
