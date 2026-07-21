import logging
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from predict import predict_sentiment, _load_model


MAX_TEXT_LENGTH = 1000
LOW_CONFIDENCE_THRESHOLD = 0.55


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("sentiment_api")

def safe_log(level: int, message: str, *args, **kwargs) -> None:
   
    try:
        logger.log(level, message, *args, **kwargs)
    except Exception:
        pass

metrics_lock = threading.Lock()

total_requests = 0
error_count = 0
total_latency_ms = 0.0
low_confidence_count = 0


def record_metrics(
    latency_ms: float,
    *,
    error: bool = False,
    low_confidence: bool = False,
) -> None:
    global total_requests
    global error_count
    global total_latency_ms
    global low_confidence_count

    with metrics_lock:
        total_requests += 1
        total_latency_ms += latency_ms

        if error:
            error_count += 1

        if low_confidence:
            low_confidence_count += 1

app = FastAPI(
    title="Sentiment Analysis API",
    description="Returns a sentiment label and confidence score for any text input.",
    version="1.2.0",
)


@app.on_event("startup")
def load_model_on_startup():
    safe_log(logging.INFO, "event=model_loading status=started")

    try:
        _load_model()
        safe_log(logging.INFO, "event=model_loading status=ready")
    except Exception:
        safe_log(
            logging.ERROR,
            "event=model_loading status=failed",
            exc_info=True,
        )
        raise

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
    error_rate: float
    average_latency_ms: float
    low_confidence_count: int


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    first_error = exc.errors()[0]
    field = ".".join(
        str(item) for item in first_error["loc"] if item != "body"
    )
    message = first_error["msg"]

    detail = f"{field}: {message}" if field else message

    if request.url.path == "/predict":
        record_metrics(
            latency_ms=0.0,
            error=True,
        )

        safe_log(
            logging.WARNING,
            "event=prediction_validation_error path=%s error=%s",
            request.url.path,
            detail,
        )

    return JSONResponse(
        status_code=422,
        content={"detail": detail},
    )


@app.get("/health", response_model=HealthResponse, status_code=200)
def health():
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse, status_code=200)
def metrics():
    with metrics_lock:
        requests = total_requests
        errors = error_count
        latency = total_latency_ms
        low_confidence = low_confidence_count

    average_latency = latency / requests if requests else 0.0
    error_rate = errors / requests if requests else 0.0

    return {
        "total_requests": requests,
        "error_count": errors,
        "error_rate": round(error_rate, 4),
        "average_latency_ms": round(average_latency, 2),
        "low_confidence_count": low_confidence,
    }


@app.post("/predict", response_model=PredictResponse, status_code=200)
def predict(request: PredictRequest):
    start_time = time.perf_counter()

    try:
        result = predict_sentiment(request.text)

        latency_ms = (time.perf_counter() - start_time) * 1000

        is_low_confidence = (
            result["confidence"] < LOW_CONFIDENCE_THRESHOLD
        )
        result["low_confidence"] = is_low_confidence

        record_metrics(
            latency_ms=latency_ms,
            low_confidence=is_low_confidence,
        )

        safe_log(
            logging.INFO,
            (
                "event=prediction_success "
                "input_length=%d "
                "label=%s "
                "confidence=%.4f "
                "low_confidence=%s "
                "latency_ms=%.2f"
            ),
            len(request.text),
            result["label"],
            result["confidence"],
            is_low_confidence,
            latency_ms,
        )

        return result

    except FileNotFoundError as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        record_metrics(
            latency_ms=latency_ms,
            error=True,
        )

        safe_log(
            logging.ERROR,
            (
                "event=prediction_error "
                "input_length=%d "
                "error_type=FileNotFoundError "
                "latency_ms=%.2f"
            ),
            len(request.text),
            latency_ms,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000

        record_metrics(
            latency_ms=latency_ms,
            error=True,
        )

        safe_log(
            logging.ERROR,
            (
                "event=prediction_error "
                "input_length=%d "
                "error_type=%s "
                "latency_ms=%.2f"
            ),
            len(request.text),
            type(exc).__name__,
            latency_ms,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(exc)}",
        ) from exc
