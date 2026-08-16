from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.database import (
    DatabaseConfigurationError,
    DatabaseOperationError,
    get_engine,
)
from src.inference import InferenceError
from src.model_hub import download_model_from_hf
from src.model_registry import DEFAULT_MODEL_ARTIFACT
from src.prediction_service import (
    get_air_quality,
    get_artifact_bundle,
    get_available_stations,
    get_model_performance,
    get_station_dashboard,
)
from src.scheduler import start_ingest_scheduler, stop_ingest_scheduler


LOGGER = logging.getLogger(__name__)
STATION_PATTERN = r"^[가-힣A-Za-z0-9\s-]+$"
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60.0
_request_times: defaultdict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = threading.Lock()


class SecureQueryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    station_name: str = Field(
        min_length=1,
        max_length=50,
        pattern=STATION_PATTERN,
    )
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_date_range(self) -> "SecureQueryModel":
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if (self.end_date - self.start_date).days > 90:
            raise ValueError("Date range cannot exceed 90 days")
        return self


class AirQualityQuery(SecureQueryModel):
    pollutant: Literal["PM2.5", "PM10"] = "PM2.5"


class PredictionQuery(SecureQueryModel):
    alert_threshold: float = Field(default=35.0, ge=0.0, le=500.0)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load models, optionally fetch Hub artifacts, and start ingest jobs."""
    LOGGER.info("Hugging Face에서 AI 모델 다운로드/로드 시도 중...")
    try:
        download_model_from_hf(DEFAULT_MODEL_ARTIFACT)
        get_artifact_bundle()
        LOGGER.info("✅ Hugging Face AI 모델 로드 완료!")
    except Exception as exc:
        LOGGER.error("❌ 모델 로드 실패: %s", exc)
    start_ingest_scheduler()
    yield
    stop_ingest_scheduler()


app = FastAPI(
    title="Air Quality Prediction API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

allowed_origins = {
    "http://localhost:8501",
    "http://127.0.0.1:8501",
}
configured_origins = os.getenv("DASHBOARD_ORIGIN", "").split(",")
allowed_origins.update(
    origin.strip()
    for origin in configured_origins
    if origin.strip().startswith(("http://", "https://"))
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(allowed_origins),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept", "Content-Type"],
)


@app.middleware("http")
async def security_middleware(
    request: Request,
    call_next: Any,
) -> Any:
    """Apply per-process IP throttling and browser security headers."""
    if request.url.path.startswith("/api/v1/"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with _rate_limit_lock:
            timestamps = _request_times[client_ip]
            while (
                timestamps
                and now - timestamps[0] >= RATE_LIMIT_WINDOW_SECONDS
            ):
                timestamps.popleft()
            if len(timestamps) >= RATE_LIMIT_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": "60"},
                )
            timestamps.append(now)
            if len(_request_times) > 10_000:
                stale_clients = [
                    key
                    for key, values in _request_times.items()
                    if not values
                    or now - values[-1] >= RATE_LIMIT_WINDOW_SECONDS
                ]
                for key in stale_clients:
                    _request_times.pop(key, None)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/live")
def live() -> dict[str, str]:
    """Process liveness for Render. Does not touch the database or model files."""
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    try:
        with get_engine().connect() as connection:
            connection.execute(select(1))
        get_artifact_bundle()
    except (
        DatabaseConfigurationError,
        DatabaseOperationError,
        InferenceError,
        SQLAlchemyError,
    ):
        raise HTTPException(status_code=503, detail="Backend dependency unavailable")
    return {"status": "ok"}


@app.get("/api/v1/models/pm25-t24/performance")
def model_performance() -> dict[str, Any]:
    try:
        return get_model_performance()
    except InferenceError:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are unavailable",
        ) from None


@app.get("/api/v1/stations")
def available_stations() -> dict[str, Any]:
    try:
        return get_available_stations()
    except (
        DatabaseConfigurationError,
        DatabaseOperationError,
    ):
        raise HTTPException(
            status_code=503,
            detail="TimescaleDB is unavailable",
        ) from None


@app.get("/api/v1/air-quality")
def air_quality(
    query: Annotated[AirQualityQuery, Query()],
) -> dict[str, Any]:
    """Return HTTP 200 and an empty data list when no rows match."""
    try:
        return get_air_quality(
            query.station_name,
            start_date=query.start_date,
            end_date=query.end_date,
            pollutant=query.pollutant,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (
        DatabaseConfigurationError,
        DatabaseOperationError,
    ):
        raise HTTPException(
            status_code=503,
            detail="TimescaleDB is unavailable",
        ) from None


@app.get("/api/v1/predictions")
def predictions(
    query: Annotated[PredictionQuery, Query()],
) -> dict[str, Any]:
    try:
        dashboard = get_station_dashboard(
            query.station_name,
            start_date=query.start_date,
            end_date=query.end_date,
            alert_threshold=query.alert_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except InferenceError:
        raise HTTPException(
            status_code=404,
            detail="No data found for the selected station.",
        ) from None
    except (
        DatabaseConfigurationError,
        DatabaseOperationError,
    ):
        raise HTTPException(
            status_code=503,
            detail="TimescaleDB is unavailable",
        ) from None
    prediction = dashboard.get("prediction")
    return {
        "metadata": {
            "station_name": query.station_name,
            "start_date": query.start_date.isoformat(),
            "end_date": query.end_date.isoformat(),
            "count": 1 if prediction else 0,
        },
        "data": [prediction] if prediction else [],
        "instance_shap_positive": dashboard.get("instance_shap_positive") or [],
        "global_shap": dashboard.get("global_shap") or [],
        "recommendation": dashboard.get("recommendation")
        or "데이터 부족으로 예측 보류",
        "model_scope_warning": dashboard.get("model_scope_warning"),
        "prediction_deferred": bool(dashboard.get("prediction_deferred")),
    }


@app.get("/api/v1/stations/{station_name}/dashboard")
def station_dashboard(
    station_name: Annotated[
        str,
        Path(min_length=1, max_length=50, pattern=STATION_PATTERN),
    ],
    start_date: Annotated[date, Query()],
    end_date: Annotated[date, Query()],
    alert_threshold: Annotated[float, Query(ge=0.0, le=500.0)] = 35.0,
) -> dict[str, Any]:
    try:
        return get_station_dashboard(
            station_name,
            start_date=start_date,
            end_date=end_date,
            alert_threshold=alert_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except InferenceError:
        raise HTTPException(
            status_code=404,
            detail="No data found for the selected station.",
        ) from None
    except (
        DatabaseConfigurationError,
        DatabaseOperationError,
    ):
        raise HTTPException(
            status_code=503,
            detail="TimescaleDB is unavailable",
        ) from None
