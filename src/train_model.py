

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import DMatrix, XGBRegressor

from src.preprocessing import (
    INTERACTION_FEATURES,
    STATION_FEATURE_PREFIX,
    add_weather_interactions,
    encode_station_features,
)
from src.utils.path import resolve_under_root


LOGGER = logging.getLogger(__name__)
ARTIFACT_NAME_PATTERN: Final = re.compile(r"^[0-9A-Za-z_-]{1,50}$")
DEFAULT_FEATURES: Final = (
    "pm10Value",
    "pm25Value",
    "so2Value",
    "coValue",
    "o3Value",
    "no2Value",
    "ta",
    "rn",
    "wind_u",
    "wind_v",
    "hm",
    "pa",
    "pm10_lag1",
    "pm10_lag24",
    "pm25_lag1",
    "pm25_lag24",
    "pm10_ma6",
    "pm10_ma24",
    "pm25_ma6",
    "pm25_ma24",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",
)


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int = 24
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    xgb_estimators: int = 800
    seed: int = 42

    def __post_init__(self) -> None:
        integer_ranges = {
            "sequence_length": (6, 168),
            "hidden_size": (8, 512),
            "num_layers": (1, 8),
            "batch_size": (1, 4096),
            "epochs": (1, 1000),
            "xgb_estimators": (10, 5000),
        }
        for name, (lower, upper) in integer_ranges.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")


@dataclass
class SequenceSamples:
    sequences: np.ndarray
    current_features: np.ndarray
    targets: np.ndarray
    feature_times: pd.DatetimeIndex
    target_times: pd.DatetimeIndex
    station_names: np.ndarray

    def subset(self, mask: np.ndarray) -> "SequenceSamples":
        return SequenceSamples(
            sequences=self.sequences[mask],
            current_features=self.current_features[mask],
            targets=self.targets[mask],
            feature_times=self.feature_times[mask],
            target_times=self.target_times[mask],
            station_names=self.station_names[mask],
        )


@dataclass
class TrainingResult:
    metrics: dict[str, dict[str, float]]
    cv_metrics: list[dict[str, float]]
    ensemble_lstm_weight: float
    feature_names: list[str]
    test_predictions: np.ndarray
    test_targets: np.ndarray
    test_target_times: pd.DatetimeIndex
    test_stations: list[str]
    station_categories: list[str]
    lstm_model: "PM25LSTM"
    xgb_model: XGBRegressor
    scaler: RobustScaler


class PM25LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(inputs)
        return self.output(encoded[:, -1, :]).squeeze(-1)


def resolve_storage_path(area: str, relative_path: str | Path) -> Path:
    """Resolve a path under data/ or models/ and reject traversal/symlinks."""
    return resolve_under_root(area, relative_path)


def _validate_training_frame(
    frame: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    required = {
        "station_name",
        "dataTime",
        "pm25_target_t24",
        *feature_names,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Training data is missing columns: {', '.join(sorted(missing))}"
        )
    clean = frame.copy()
    clean["dataTime"] = pd.to_datetime(clean["dataTime"], utc=True, errors="raise")
    for column in [*feature_names, "pm25_target_t24"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean.loc[~np.isfinite(clean[column]), column] = np.nan
    clean = clean.dropna(subset=[*feature_names, "pm25_target_t24"])
    clean = clean.sort_values(["station_name", "dataTime"]).reset_index(drop=True)
    if clean.duplicated(["station_name", "dataTime"]).any():
        raise ValueError("Training data contains duplicate station timestamps")
    if clean.empty:
        raise ValueError("No complete rows remain for model training")
    return clean


def build_sequence_samples(
    frame: pd.DataFrame,
    feature_names: list[str],
    sequence_length: int,
) -> SequenceSamples:
    """Build contiguous station-specific windows ending at feature time t."""
    clean = _validate_training_frame(frame, feature_names)
    sequences: list[np.ndarray] = []
    current: list[np.ndarray] = []
    targets: list[float] = []
    feature_times: list[pd.Timestamp] = []
    target_times: list[pd.Timestamp] = []
    station_names: list[str] = []

    for station_name, group in clean.groupby("station_name", sort=False):
        group = group.sort_values("dataTime").reset_index(drop=True)
        times = pd.DatetimeIndex(group["dataTime"])
        features = group[feature_names].to_numpy(dtype=np.float32)
        labels = group["pm25_target_t24"].to_numpy(dtype=np.float32)
        for end in range(sequence_length - 1, len(group)):
            start = end - sequence_length + 1
            window_times = times[start : end + 1]
            if len(window_times) != sequence_length:
                continue
            deltas = np.diff(window_times.asi8)
            if not np.all(deltas == pd.Timedelta(hours=1).value):
                continue
            sequences.append(features[start : end + 1])
            current.append(features[end])
            targets.append(float(labels[end]))
            feature_times.append(times[end])
            target_times.append(times[end] + pd.Timedelta(hours=24))
            station_names.append(str(station_name))

    if not sequences:
        raise ValueError("No contiguous hourly sequences are available")
    return SequenceSamples(
        sequences=np.asarray(sequences, dtype=np.float32),
        current_features=np.asarray(current, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        feature_times=pd.DatetimeIndex(feature_times),
        target_times=pd.DatetimeIndex(target_times),
        station_names=np.asarray(station_names, dtype=object),
    )


def resolve_feature_names(frame: pd.DataFrame) -> list[str]:
    """Use shared numeric features plus station one-hots and interactions."""
    names = [column for column in DEFAULT_FEATURES if column in frame.columns]
    names.extend(
        sorted(
            column
            for column in frame.columns
            if column.startswith(STATION_FEATURE_PREFIX)
        )
    )
    names.extend(
        column for column in INTERACTION_FEATURES if column in frame.columns
    )
    return list(dict.fromkeys(names))


def chronological_split(
    samples: SequenceSamples,
) -> tuple[SequenceSamples, SequenceSamples, SequenceSamples]:
    """Split 70/15/15 by global target time across all stations.

    A single time cut is applied to every station so a later hour at station B
    cannot leak into training while an earlier hour at station A is held out.
    """
    order = np.argsort(samples.target_times.asi8, kind="stable")
    ordered = samples.subset(order)
    unique_times = ordered.target_times.unique().sort_values()
    if len(unique_times) < 20:
        raise ValueError("At least 20 unique target hours are required")

    validation_index = max(1, math.floor(len(unique_times) * 0.70))
    test_index = max(validation_index + 1, math.floor(len(unique_times) * 0.85))
    validation_start = unique_times[validation_index]
    test_start = unique_times[test_index]

    train_mask = np.asarray(ordered.target_times < validation_start)
    validation_mask = np.asarray(
        (ordered.target_times >= validation_start)
        & (ordered.target_times < test_start)
    )
    test_mask = np.asarray(ordered.target_times >= test_start)
    split = (
        ordered.subset(train_mask),
        ordered.subset(validation_mask),
        ordered.subset(test_mask),
    )
    if any(part.targets.size == 0 for part in split):
        raise ValueError("Chronological split produced an empty partition")
    for label, part in zip(("train", "validation", "test"), split, strict=True):
        stations = sorted({str(name) for name in part.station_names.tolist()})
        LOGGER.info(
            "%s split: %d samples across %d station(s) [%s]",
            label,
            part.targets.size,
            len(stations),
            ", ".join(stations),
        )
    return split


def _scale_samples(
    train: SequenceSamples,
    validation: SequenceSamples,
    test: SequenceSamples,
) -> tuple[SequenceSamples, SequenceSamples, SequenceSamples, RobustScaler]:
    feature_count = train.sequences.shape[-1]
    scaler = RobustScaler()
    scaler.fit(train.sequences.reshape(-1, feature_count))

    def transform(samples: SequenceSamples) -> SequenceSamples:
        sequence_shape = samples.sequences.shape
        scaled_sequences = scaler.transform(
            samples.sequences.reshape(-1, feature_count)
        ).reshape(sequence_shape)
        return SequenceSamples(
            sequences=scaled_sequences.astype(np.float32),
            current_features=scaler.transform(
                samples.current_features
            ).astype(np.float32),
            targets=samples.targets,
            feature_times=samples.feature_times,
            target_times=samples.target_times,
            station_names=samples.station_names,
        )

    return transform(train), transform(validation), transform(test), scaler


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_lstm(
    train: SequenceSamples,
    validation: SequenceSamples,
    config: TrainingConfig,
    device: torch.device,
) -> PM25LSTM:
    model = PM25LSTM(
        input_size=train.sequences.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate
    )
    loss_function = nn.HuberLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.sequences),
            torch.from_numpy(train.targets),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    validation_x = torch.from_numpy(validation.sequences).to(device)
    validation_y = torch.from_numpy(validation.targets).to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _ in range(config.epochs):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features.to(device))
            loss = loss_function(predictions, targets.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(validation_x), validation_y
            ).item()
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 10:
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a valid model")
    model.load_state_dict(best_state)
    return model


def _predict_lstm(
    model: PM25LSTM,
    samples: SequenceSamples,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.from_numpy(samples.sequences).to(device)
        )
    return predictions.cpu().numpy()


def _new_xgb(config: TrainingConfig, seed: int | None = None) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=config.xgb_estimators,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=config.seed if seed is None else seed,
    )


def regression_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
    }


def time_series_cross_validate_xgb(
    features: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
    splits: int = 3,
) -> list[dict[str, float]]:
    """Run expanding-window validation with a scaler fitted per fold."""
    if not 2 <= splits <= 10:
        raise ValueError("splits must be between 2 and 10")
    if len(targets) <= splits:
        raise ValueError("Not enough samples for TimeSeriesSplit")
    results: list[dict[str, float]] = []
    for fold, (train_index, validation_index) in enumerate(
        TimeSeriesSplit(n_splits=splits).split(features),
        start=1,
    ):
        scaler = RobustScaler().fit(features[train_index])
        model = _new_xgb(config, seed=config.seed + fold)
        train_x = scaler.transform(features[train_index])
        validation_x = scaler.transform(features[validation_index])
        try:
            model.fit(
                train_x,
                targets[train_index],
                eval_set=[(validation_x, targets[validation_index])],
                verbose=False,
            )
        except TypeError:
            model.fit(train_x, targets[train_index])
        predictions = model.predict(validation_x)
        results.append(regression_metrics(targets[validation_index], predictions))
    return results


def train_ensemble(
    frame: pd.DataFrame,
    *,
    feature_names: list[str] | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train models and select the blend weight using validation data only."""
    prepared = add_weather_interactions(frame)
    prepared, station_categories = encode_station_features(prepared)
    selected_features = feature_names or resolve_feature_names(prepared)
    if not selected_features:
        raise ValueError("No supported model features were found")
    training_config = config or TrainingConfig()
    _set_seed(training_config.seed)
    samples = build_sequence_samples(
        prepared, selected_features, training_config.sequence_length
    )
    raw_train, raw_validation, raw_test = chronological_split(samples)
    train, validation, test, scaler = _scale_samples(
        raw_train, raw_validation, raw_test
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm_model = _train_lstm(train, validation, training_config, device)
    xgb_model = _new_xgb(training_config)
    try:
        xgb_model.fit(
            train.current_features,
            train.targets,
            eval_set=[(validation.current_features, validation.targets)],
            verbose=False,
        )
    except TypeError:
        xgb_model.fit(train.current_features, train.targets)

    validation_lstm = _predict_lstm(lstm_model, validation, device)
    validation_xgb = xgb_model.predict(validation.current_features)
    difference = validation_lstm - validation_xgb
    denominator = float(np.dot(difference, difference))
    blend_weight = (
        0.5
        if denominator <= 1e-12
        else float(
            np.clip(
                np.dot(
                    validation.targets - validation_xgb,
                    difference,
                )
                / denominator,
                0.0,
                1.0,
            )
        )
    )
    validation_blend = (
        blend_weight * validation_lstm + (1.0 - blend_weight) * validation_xgb
    )
    validation_scores = {
        "lstm": r2_score(validation.targets, validation_lstm),
        "xgboost": r2_score(validation.targets, validation_xgb),
        "ensemble": r2_score(validation.targets, validation_blend),
    }
    lstm_val = validation_scores["lstm"]
    xgb_val = validation_scores["xgboost"]
    if max(lstm_val, xgb_val) < 0:
        # A negative validation window is treated as uninformative; keep the
        # sequence model as the primary T+24 forecaster and XGBoost for SHAP.
        weight = max(blend_weight, 0.8)
        selected = "lstm_prior"
    elif xgb_val > lstm_val + 0.05 and xgb_val > validation_scores["ensemble"]:
        weight = 0.0
        selected = "xgboost"
    elif lstm_val > xgb_val + 0.05 and lstm_val > validation_scores["ensemble"]:
        weight = 1.0
        selected = "lstm"
    else:
        weight = blend_weight
        selected = "blend"
    LOGGER.info(
        "Validation R² lstm=%.3f xgboost=%.3f blend=%.3f; using %s (lstm_weight=%.3f)",
        lstm_val,
        xgb_val,
        validation_scores["ensemble"],
        selected,
        weight,
    )

    test_lstm = _predict_lstm(lstm_model, test, device)
    test_xgb = xgb_model.predict(test.current_features)
    ensemble = weight * test_lstm + (1.0 - weight) * test_xgb
    metrics = {
        "lstm": regression_metrics(test.targets, test_lstm),
        "xgboost": regression_metrics(test.targets, test_xgb),
        "ensemble": regression_metrics(test.targets, ensemble),
    }
    cv_metrics = time_series_cross_validate_xgb(
        raw_train.current_features,
        raw_train.targets,
        training_config,
    )
    return TrainingResult(
        metrics=metrics,
        cv_metrics=cv_metrics,
        ensemble_lstm_weight=weight,
        feature_names=selected_features,
        test_predictions=ensemble,
        test_targets=test.targets,
        test_target_times=test.target_times,
        test_stations=[str(name) for name in test.station_names.tolist()],
        station_categories=station_categories,
        lstm_model=lstm_model.cpu(),
        xgb_model=xgb_model,
        scaler=scaler,
    )


def save_artifacts(
    result: TrainingResult,
    config: TrainingConfig,
    *,
    artifact_name: str = "pm25_t24_ensemble",
) -> Path:
    """Persist models and metadata only under the project models directory."""
    if not ARTIFACT_NAME_PATTERN.fullmatch(artifact_name):
        raise ValueError("artifact_name contains invalid characters")
    output_directory = resolve_storage_path("models", artifact_name)
    output_directory.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": result.lstm_model.state_dict(),
            "input_size": len(result.feature_names),
            "config": asdict(config),
        },
        output_directory / "lstm.pt",
    )
    result.xgb_model.save_model(str(output_directory / "xgboost.json"))
    joblib.dump(result.scaler, output_directory / "scaler.joblib")
    metadata = {
        "target": "pm25_target_t24",
        "model_type": "global_multi_station",
        "features": result.feature_names,
        "station_categories": result.station_categories,
        "metrics": result.metrics,
        "time_series_cv": result.cv_metrics,
        "ensemble_lstm_weight": result.ensemble_lstm_weight,
        "config": asdict(config),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            "station_name": result.test_stations,
            "target_time": result.test_target_times,
            "actual_pm25": result.test_targets,
            "predicted_pm25": result.test_predictions,
        }
    )
    predictions.to_csv(output_directory / "test_predictions.csv", index=False)
    return output_directory


def save_shap_importance(
    result: TrainingResult,
    frame: pd.DataFrame,
    *,
    artifact_name: str = "pm25_t24_ensemble",
    max_samples: int = 2_000,
) -> Path:
    """Save aggregate TreeSHAP importance without serializing source data."""
    if not 1 <= max_samples <= 10_000:
        raise ValueError("max_samples must be between 1 and 10000")
    try:
        import shap
    except ImportError:
        raise RuntimeError("Install shap to generate XAI artifacts") from None

    validated = _validate_training_frame(frame, result.feature_names)
    sample = validated[result.feature_names].tail(max_samples).to_numpy(
        dtype=float
    )
    scaled = result.scaler.transform(sample)
    try:
        explanation = shap.TreeExplainer(result.xgb_model).shap_values(scaled)
    except (TypeError, ValueError):
        # Some SHAP releases lag XGBoost's vector-valued base_score schema.
        # XGBoost pred_contribs uses its native exact TreeSHAP implementation.
        LOGGER.warning(
            "SHAP package is incompatible with this XGBoost version; "
            "using native TreeSHAP contributions"
        )
        contribution_matrix = DMatrix(
            scaled,
            feature_names=result.feature_names,
        )
        explanation = result.xgb_model.get_booster().predict(
            contribution_matrix,
            pred_contribs=True,
        )[:, :-1]
    mean_absolute = np.abs(np.asarray(explanation)).mean(axis=0)
    importance = dict(
        sorted(
            zip(result.feature_names, mean_absolute.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    output_directory = resolve_storage_path("models", artifact_name)
    if not output_directory.is_dir():
        raise ValueError("Model artifact directory does not exist")
    output_path = output_directory / "shap_importance.json"
    output_path.write_text(
        json.dumps(importance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="engineered_air_weather.csv",
        help="CSV path relative to data/",
    )
    parser.add_argument(
        "--artifact-name",
        default="pm25_t24_ensemble",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--xgb-estimators", type=int, default=800)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--with-shap", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    arguments = _parse_arguments()
    input_path = resolve_storage_path("data", arguments.input)
    if not input_path.is_file():
        raise SystemExit("Input data file was not found under data/")

    config = TrainingConfig(
        epochs=arguments.epochs,
        sequence_length=arguments.sequence_length,
        xgb_estimators=arguments.xgb_estimators,
    )
    data = pd.read_csv(input_path)
    result = train_ensemble(data, config=config)
    output = save_artifacts(
        result,
        config,
        artifact_name=arguments.artifact_name,
    )
    if arguments.with_shap:
        save_shap_importance(
            result,
            data,
            artifact_name=arguments.artifact_name,
        )
    LOGGER.info("Training completed; artifacts saved under models/%s", output.name)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
