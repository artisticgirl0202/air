from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from xgboost import DMatrix, XGBRegressor

from src.model_registry import DEFAULT_MODEL_ARTIFACT
from src.preprocessing import add_derived_forecast_features, ensure_model_features
from src.pm25_lstm import PM25LSTM
from src.utils.path import resolve_model_path


ARTIFACT_PATTERN: Final = re.compile(r"^[0-9A-Za-z_-]{1,50}$")
KST: Final = "Asia/Seoul"
DB_TO_MODEL_COLUMNS: Final = {
    "measured_at": "dataTime",
    "pm10": "pm10Value",
    "pm25": "pm25Value",
    "so2": "so2Value",
    "co": "coValue",
    "o3": "o3Value",
    "no2": "no2Value",
    "temperature": "ta",
    "rainfall": "rn",
    "humidity": "hm",
    "pressure": "pa",
}


class InferenceError(RuntimeError):
    """Safe inference error suitable for an API response."""


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_name: str
    feature_names: list[str]
    sequence_length: int
    ensemble_lstm_weight: float
    metadata: dict[str, Any]
    global_shap: dict[str, float]
    empirical_interval_radius: float
    lstm_model: PM25LSTM
    xgb_model: XGBRegressor
    scaler: RobustScaler


@dataclass(frozen=True)
class PredictionResult:
    station_name: str
    feature_time: datetime
    target_time: datetime
    current_pm25: float
    predicted_pm25: float
    lower_bound: float
    upper_bound: float
    lstm_prediction: float
    xgb_prediction: float
    shap_values: dict[str, float]


def _artifact_directory(artifact_name: str) -> Path:
    if not ARTIFACT_PATTERN.fullmatch(artifact_name):
        raise ValueError("Invalid model artifact name")
    directory = resolve_model_path(artifact_name)
    if not directory.is_dir():
        raise InferenceError("Model artifacts were not found")
    return directory


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise InferenceError("A model metadata file is invalid") from None
    if not isinstance(payload, dict):
        raise InferenceError("A model metadata file has an invalid schema")
    return payload


def load_artifacts(
    artifact_name: str = DEFAULT_MODEL_ARTIFACT,
) -> ArtifactBundle:
    """Load one allowlisted local artifact bundle for read-only inference."""
    directory = _artifact_directory(artifact_name)
    required_files = (
        "metadata.json",
        "shap_importance.json",
        "lstm.pt",
        "xgboost.json",
        "scaler.joblib",
        "test_predictions.csv",
    )
    if any(not (directory / filename).is_file() for filename in required_files):
        raise InferenceError("The model artifact bundle is incomplete")

    metadata = _read_json(directory / "metadata.json")
    global_shap_raw = _read_json(directory / "shap_importance.json")
    try:
        feature_names = [str(value) for value in metadata["features"]]
        config = metadata["config"]
        sequence_length = int(config["sequence_length"])
        weight = float(metadata["ensemble_lstm_weight"])
        checkpoint = torch.load(
            directory / "lstm.pt",
            map_location="cpu",
            weights_only=True,
        )
        input_size = int(checkpoint["input_size"])
        lstm_model = PM25LSTM(
            input_size=input_size,
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            dropout=float(config["dropout"]),
        )
        lstm_model.load_state_dict(checkpoint["state_dict"])
        lstm_model.eval()
        xgb_model = XGBRegressor()
        xgb_model.load_model(str(directory / "xgboost.json"))
        scaler = joblib.load(directory / "scaler.joblib")
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        raise InferenceError("Unable to load the model artifact bundle") from None
    if input_size != len(feature_names) or not isinstance(scaler, RobustScaler):
        raise InferenceError("Model artifact dimensions are inconsistent")
    if not 0.0 <= weight <= 1.0:
        raise InferenceError("The ensemble weight is invalid")

    predictions = pd.read_csv(directory / "test_predictions.csv")
    if {"actual_pm25", "predicted_pm25"}.issubset(predictions.columns):
        residuals = (
            pd.to_numeric(predictions["actual_pm25"], errors="coerce")
            - pd.to_numeric(predictions["predicted_pm25"], errors="coerce")
        ).abs()
        interval_radius = float(residuals.quantile(0.90))
    else:
        interval_radius = float("nan")
    if not math.isfinite(interval_radius):
        interval_radius = 0.0

    global_shap = {
        str(feature): float(value)
        for feature, value in global_shap_raw.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    return ArtifactBundle(
        artifact_name=artifact_name,
        feature_names=feature_names,
        sequence_length=sequence_length,
        ensemble_lstm_weight=weight,
        metadata=metadata,
        global_shap=global_shap,
        empirical_interval_radius=interval_radius,
        lstm_model=lstm_model,
        xgb_model=xgb_model,
        scaler=scaler,
    )


def engineer_db_history(measurements: pd.DataFrame) -> pd.DataFrame:
    """Recreate training features from hourly TimescaleDB measurements."""
    required = {"station_name", *DB_TO_MODEL_COLUMNS.keys(), "wind_u", "wind_v"}
    missing = required - set(measurements.columns)
    if missing:
        raise InferenceError(
            "Database history is missing required measurement columns"
        )
    frame = measurements.rename(columns=DB_TO_MODEL_COLUMNS).copy()
    frame["dataTime"] = pd.to_datetime(
        frame["dataTime"], utc=True, errors="coerce"
    )
    if frame["dataTime"].isna().any():
        raise InferenceError("Database contains invalid measurement times")
    frame = frame.sort_values(["station_name", "dataTime"]).reset_index(drop=True)
    if frame.duplicated(["station_name", "dataTime"]).any():
        raise InferenceError("Database contains duplicate station hours")

    local_time = frame["dataTime"].dt.tz_convert(KST)
    frame["hour_sin"] = np.sin(2 * np.pi * local_time.dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * local_time.dt.hour / 24)
    frame["month_sin"] = np.sin(
        2 * np.pi * (local_time.dt.month - 1) / 12
    )
    frame["month_cos"] = np.cos(
        2 * np.pi * (local_time.dt.month - 1) / 12
    )
    frame["day_of_week"] = local_time.dt.dayofweek

    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("station_name", sort=False):
        ordered = group.sort_values("dataTime").copy()
        indexed = ordered.set_index("dataTime")
        for source in ("pm10Value", "pm25Value"):
            source_series = indexed[source]
            for hours in (1, 24):
                lookup_times = ordered["dataTime"] - pd.Timedelta(hours=hours)
                ordered[f"{source.removesuffix('Value')}_lag{hours}"] = (
                    source_series.reindex(pd.DatetimeIndex(lookup_times)).to_numpy()
                )
            for hours in (6, 24):
                minimum = max(2, hours // 2)
                ordered[f"{source.removesuffix('Value')}_ma{hours}"] = (
                    source_series
                    .rolling(f"{hours}h", min_periods=minimum, closed="right")
                    .mean()
                    .to_numpy()
                )
        parts.append(ordered)
    return add_derived_forecast_features(pd.concat(parts, ignore_index=True))


def predict_latest(
    bundle: ArtifactBundle,
    engineered_history: pd.DataFrame,
) -> PredictionResult:
    """Predict from the latest contiguous, complete 24-hour feature window."""
    if "station_name" not in engineered_history.columns or "dataTime" not in engineered_history.columns:
        raise InferenceError("Engineered history lacks model features")
    frame = ensure_model_features(
        engineered_history.copy(),
        bundle.feature_names,
        station_categories=bundle.metadata.get("station_categories"),
    )
    missing = set(bundle.feature_names) - set(frame.columns)
    if missing:
        raise InferenceError("Engineered history lacks model features")
    for column in bundle.feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[~np.isfinite(frame[column]), column] = np.nan
    frame = frame.dropna(subset=bundle.feature_names)
    if frame.empty:
        raise InferenceError("No complete feature rows are available")

    station_names = frame["station_name"].dropna().unique()
    if len(station_names) != 1:
        raise InferenceError("Inference requires exactly one station")
    frame = frame.sort_values("dataTime")
    window = frame.tail(bundle.sequence_length)
    if len(window) != bundle.sequence_length:
        raise InferenceError("At least 24 complete hourly feature rows are required")
    times = pd.DatetimeIndex(window["dataTime"])
    if not np.all(np.diff(times.asi8) == pd.Timedelta(hours=1).value):
        raise InferenceError("The latest model window is not hourly-contiguous")

    raw_sequence = window[bundle.feature_names].to_numpy(dtype=np.float32)
    scaled_sequence = bundle.scaler.transform(raw_sequence).astype(np.float32)
    scaled_current = scaled_sequence[-1].reshape(1, -1)
    with torch.no_grad():
        lstm_prediction = float(
            bundle.lstm_model(
                torch.from_numpy(scaled_sequence).unsqueeze(0)
            ).item()
        )
    xgb_prediction = float(bundle.xgb_model.predict(scaled_current)[0])
    prediction = (
        bundle.ensemble_lstm_weight * lstm_prediction
        + (1.0 - bundle.ensemble_lstm_weight) * xgb_prediction
    )

    contributions = bundle.xgb_model.get_booster().predict(
        DMatrix(scaled_current, feature_names=bundle.feature_names),
        pred_contribs=True,
    )[0, :-1]
    shap_values = {
        feature: float(value)
        for feature, value in zip(
            bundle.feature_names, contributions, strict=True
        )
    }
    feature_time = pd.Timestamp(times[-1]).to_pydatetime()
    radius = bundle.empirical_interval_radius
    return PredictionResult(
        station_name=str(station_names[0]),
        feature_time=feature_time,
        target_time=(pd.Timestamp(times[-1]) + pd.Timedelta(hours=24)).to_pydatetime(),
        current_pm25=float(window.iloc[-1]["pm25Value"]),
        predicted_pm25=float(prediction),
        lower_bound=max(0.0, float(prediction - radius)),
        upper_bound=float(prediction + radius),
        lstm_prediction=lstm_prediction,
        xgb_prediction=xgb_prediction,
        shap_values=shap_values,
    )
