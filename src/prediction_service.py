from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.database import (
    latest_measurement_time,
    list_available_stations,
    load_measurements,
)
from src.inference import (
    ArtifactBundle,
    InferenceError,
    engineer_db_history,
    load_artifacts,
    predict_latest,
)
from src.model_registry import DEFAULT_MODEL_ARTIFACT


LOGGER = logging.getLogger(__name__)
PREDICTION_DEFERRED_MESSAGE = "데이터 부족으로 예측 보류"


@lru_cache(maxsize=1)
def get_artifact_bundle() -> ArtifactBundle:
    return load_artifacts(DEFAULT_MODEL_ARTIFACT)


def _risk_grade(pm25: float | None) -> str:
    if pm25 is None or not np.isfinite(pm25) or pm25 < 0:
        return "데이터 없음"
    if pm25 <= 15:
        return "좋음"
    if pm25 <= 35:
        return "보통"
    if pm25 <= 75:
        return "나쁨"
    return "매우 나쁨"


def _json_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _history_records(frame: pd.DataFrame, display_start: pd.Timestamp) -> list[dict[str, Any]]:
    selected = frame.loc[
        pd.to_datetime(frame["measured_at"], utc=True) >= display_start
    ]
    return [
        {
            "measured_at": pd.Timestamp(row.measured_at).isoformat(),
            "pm25": _json_number(row.pm25),
            "pm10": _json_number(row.pm10),
            "aqi": _json_number(row.aqi),
        }
        for row in selected.itertuples(index=False)
    ]


def _date_bounds(
    start_date: date,
    end_date: date,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if (end_date - start_date).days > 90:
        raise ValueError("Date range cannot exceed 90 days")
    start = pd.Timestamp(start_date).tz_localize("Asia/Seoul").tz_convert("UTC")
    end_exclusive = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).tz_localize("Asia/Seoul").tz_convert("UTC")
    return start, end_exclusive


def get_available_stations() -> dict[str, Any]:
    stations = list_available_stations()
    return {
        "count": len(stations),
        "stations": [
            {
                "station_name": row["station_name"],
                "start_at": pd.Timestamp(row["start_at"]).isoformat(),
                "end_at": pd.Timestamp(row["end_at"]).isoformat(),
                "row_count": row["row_count"],
            }
            for row in stations
        ],
    }


def get_air_quality(
    station_name: str,
    *,
    start_date: date,
    end_date: date,
    pollutant: str,
) -> dict[str, Any]:
    if pollutant not in {"PM2.5", "PM10"}:
        raise ValueError("pollutant must be PM2.5 or PM10")
    start, end_exclusive = _date_bounds(start_date, end_date)
    frame = load_measurements(
        start.to_pydatetime(),
        end_exclusive.to_pydatetime(),
        station_name=station_name,
        limit=24 * 91,
    )
    value_column = "pm25" if pollutant == "PM2.5" else "pm10"
    records = [
        {
            "measured_at": pd.Timestamp(row.measured_at).isoformat(),
            "value": _json_number(getattr(row, value_column)),
            "pollutant": pollutant,
        }
        for row in frame.itertuples(index=False)
    ]
    return {
        "metadata": {
            "station_name": station_name,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "pollutant": pollutant,
            "count": len(records),
        },
        "data": records,
    }


def _model_scope_warning(station_name: str, bundle: ArtifactBundle) -> str | None:
    trained = [
        str(name)
        for name in bundle.metadata.get("station_categories", [])
        if str(name).strip()
    ]
    if not trained:
        return None
    if station_name in trained:
        return None
    return (
        "선택한 측정소는 전역 모델 학습 집합에 없어 "
        "지역 특성이 반영되지 않은 참고 예측입니다."
    )


def get_model_performance() -> dict[str, Any]:
    bundle = get_artifact_bundle()
    metrics = bundle.metadata.get("metrics", {})
    ensemble = metrics.get("ensemble", {})
    r2 = float(ensemble.get("r2", float("nan")))
    if not np.isfinite(r2):
        assessment = "평가 지표를 확인할 수 없습니다."
        production_ready = False
    elif r2 < 0:
        assessment = (
            "R²가 0보다 작아 평균값 기준선보다 성능이 낮습니다. "
            "현재 모델은 의사결정 참고용이며 자동 제어에 사용하면 안 됩니다."
        )
        production_ready = False
    else:
        assessment = "기준선보다 설명력이 높지만 운영 드리프트 검증이 필요합니다."
        production_ready = True
    return {
        "artifact_name": bundle.artifact_name,
        "target": bundle.metadata.get("target"),
        "metrics": metrics,
        "time_series_cv": bundle.metadata.get("time_series_cv", []),
        "ensemble_lstm_weight": bundle.ensemble_lstm_weight,
        "production_ready": production_ready,
        "assessment": assessment,
        "interval_type": "holdout absolute residual 90th percentile",
        "interval_radius": bundle.empirical_interval_radius,
        "station_categories": bundle.metadata.get("station_categories", []),
        "model_type": bundle.metadata.get("model_type", "global_multi_station"),
    }


def get_station_dashboard(
    station_name: str,
    *,
    days: int = 7,
    start_date: date | None = None,
    end_date: date | None = None,
    alert_threshold: float = 35.0,
) -> dict[str, Any]:
    """Return live history, T+24 forecast, SHAP, and model performance."""
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 30:
        raise ValueError("days must be an integer between 1 and 30")
    if not 0.0 <= alert_threshold <= 500.0:
        raise ValueError("alert_threshold must be between 0 and 500")

    if (start_date is None) != (end_date is None):
        raise ValueError("start_date and end_date must be provided together")
    if start_date is not None and end_date is not None:
        display_start, end_exclusive = _date_bounds(start_date, end_date)
        display_latest = latest_measurement_time(
            station_name,
            end_at=end_exclusive.to_pydatetime(),
        )
    else:
        display_start = None
        end_exclusive = None
        display_latest = None

    prediction_as_of = latest_measurement_time(station_name)
    if prediction_as_of is None:
        raise InferenceError("No TimescaleDB data exists for this station")
    latest_timestamp = pd.Timestamp(prediction_as_of)
    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize("UTC")
    else:
        latest_timestamp = latest_timestamp.tz_convert("UTC")

    if display_start is not None and display_latest is None:
        raise InferenceError("No data found for the selected station and date range")
    history_hours = max(days * 24, 72)
    if display_start is not None and end_exclusive is not None:
        history_start = display_start
        history_end = end_exclusive
    else:
        history_start = latest_timestamp - pd.Timedelta(hours=history_hours)
        history_end = latest_timestamp + pd.Timedelta(hours=1)
    history = load_measurements(
        history_start.to_pydatetime(),
        history_end.to_pydatetime(),
        station_name=station_name,
        limit=max(history_hours + 48, 24 * 94),
    )
    if history.empty:
        raise InferenceError("No data found for the selected station and date range")
    if display_start is None:
        display_start = latest_timestamp - pd.Timedelta(days=days)
    history_records = _history_records(history, display_start)
    if not history_records:
        raise InferenceError("No data found for the selected station and date range")

    latest_pm25 = _json_number(history["pm25"].iloc[-1])
    payload: dict[str, Any] = {
        "station_name": station_name,
        "latest_measurement_time": latest_timestamp.isoformat(),
        "current": {
            "pm25": latest_pm25,
            "risk_grade": _risk_grade(latest_pm25),
        },
        "history": history_records,
        "prediction": None,
        "instance_shap_positive": [],
        "global_shap": [],
        "recommendation": PREDICTION_DEFERRED_MESSAGE,
        "performance": None,
        "model_scope_warning": None,
        "prediction_deferred": True,
    }
    try:
        payload["performance"] = get_model_performance()
    except InferenceError:
        LOGGER.warning("Model performance artifacts are unavailable")

    try:
        feature_start = latest_timestamp - pd.Timedelta(hours=72)
        feature_end = latest_timestamp + pd.Timedelta(hours=1)
        feature_history = load_measurements(
            feature_start.to_pydatetime(),
            feature_end.to_pydatetime(),
            station_name=station_name,
            limit=24 * 8,
        )
        if feature_history.empty:
            raise InferenceError("Latest feature window is empty")
        engineered = engineer_db_history(feature_history)
        bundle = get_artifact_bundle()
        prediction = predict_latest(bundle, engineered)
        prediction_payload = asdict(prediction)
        prediction_payload["feature_time"] = prediction.feature_time.isoformat()
        prediction_payload["target_time"] = prediction.target_time.isoformat()
        positive_causes = sorted(
            (
                {"feature": feature, "shap_value": value}
                for feature, value in prediction.shap_values.items()
                if value > 0
            ),
            key=lambda item: item["shap_value"],
            reverse=True,
        )[:10]
        global_shap = [
            {"feature": feature, "importance": importance}
            for feature, importance in sorted(
                bundle.global_shap.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        ]
        payload["current"] = {
            "pm25": prediction.current_pm25,
            "risk_grade": _risk_grade(prediction.current_pm25),
        }
        payload["prediction"] = prediction_payload
        payload["instance_shap_positive"] = positive_causes
        payload["global_shap"] = global_shap
        payload["recommendation"] = (
            f"예측 농도가 기준 {alert_threshold:.1f} µg/m³ 이상입니다. "
            "목표 시각 전에 방지시설 사전 점검 및 가동을 권장합니다."
            if prediction.predicted_pm25 >= alert_threshold
            else "현재 예측은 설정한 기준 미만입니다. 추세를 계속 모니터링하세요."
        )
        payload["model_scope_warning"] = _model_scope_warning(station_name, bundle)
        payload["prediction_deferred"] = False
    except InferenceError:
        LOGGER.warning(
            "T+24 prediction deferred for station %s; latest window is incomplete",
            station_name,
        )
    return payload
