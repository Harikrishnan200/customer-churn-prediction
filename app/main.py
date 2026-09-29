"""
ChurnOps FastAPI application.

WHAT: Serves churn predictions over HTTP, backed by the "production"
      model version from the MLflow Model Registry.
WHY:  A trained model sitting in MLflow isn't useful to anyone until
      something exposes it as a service other systems can call.
      FastAPI is a lightweight, well-documented way to do that, with
      built-in request validation (via Pydantic) and automatic
      interactive docs at /docs.
WHERE: This is the entry point for the "serving" stage of the
       architecture (Dataset -> Training -> MLflow -> FastAPI ->
       Docker).
HOW:  Run locally with `uvicorn app.main:app --reload`, or inside
      Docker (see Dockerfile). Configuration (which MLflow server,
      which model, which alias) comes from environment variables, with
      configs/config.yaml as the default when a variable isn't set -
      see .env.example.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.predictor import ChurnPredictor
from app.schemas import ChurnRequest, ChurnResponse, HealthResponse, ModelInfoResponse
from src.data import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the production model once, when the app starts.

    Loading it here (instead of inside /predict) means: (1) every
    request after startup is fast, since the model is already in
    memory, and (2) if the model can't be loaded, we know immediately
    from the startup logs instead of discovering it on a customer's
    request.
    """
    global predictor

    config = load_config()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    model_name = os.getenv("MLFLOW_MODEL_NAME", config["mlflow"]["model_name"])
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "production")

    predictor = ChurnPredictor(tracking_uri, model_name, model_alias)
    try:
        predictor.load()
    except Exception:
        # We don't crash the app: /health and /docs should still work so
        # the failure is visible and debuggable, but /predict will
        # correctly report 503 until a model becomes available.
        logger.exception("Failed to load model at startup")

    yield


app = FastAPI(
    title="ChurnOps API",
    description="Customer churn prediction service",
    lifespan=lifespan,
)

# --- Prometheus metrics -----------------------------------------------
# WHAT: Counters and a histogram tracking API usage and behaviour.
# WHY:  Once a model is in production, you need to know it's actually
#       being called, how fast, and how often it fails or predicts
#       churn - without that, problems go unnoticed until a human
#       complains.
# WHERE: Updated in the /predict endpoint below.
# HOW:   Exposed as plain text at GET /metrics, in the format Prometheus
#       scrapes (see monitoring/prometheus.yml).
prediction_requests_total = Counter(
    "prediction_requests_total", "Total number of prediction requests received"
)
prediction_errors_total = Counter(
    "prediction_errors_total", "Total number of prediction requests that failed"
)
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds", "Time taken to serve a prediction request"
)
churn_predictions_total = Counter(
    "churn_predictions_total", "Total number of predictions, by outcome", ["outcome"]
)

predictor: ChurnPredictor | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=str(predictor.model_version),
        model_alias=predictor.model_alias,
    )


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest) -> ChurnResponse:
    prediction_requests_total.inc()

    if predictor is None or not predictor.is_ready():
        prediction_errors_total.inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start_time = time.perf_counter()
    try:
        churn, probability = predictor.predict(request.model_dump())
    except Exception:
        prediction_errors_total.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
    finally:
        prediction_latency_seconds.observe(time.perf_counter() - start_time)

    churn_predictions_total.labels(outcome=str(churn)).inc()
    logger.info("Prediction served: churn=%s probability=%.3f", churn, probability)

    return ChurnResponse(
        churn=churn,
        probability=probability,
        model_version=str(predictor.model_version),
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
