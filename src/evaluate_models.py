from __future__ import annotations

import argparse
import json
import logging
import math
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.inference import ArtifactBundle, load_artifacts
from src.model_registry import CANDIDATE_MODEL_ARTIFACTS
from src.preprocessing import ensure_model_features
from src.train_model import SequenceSamples, build_sequence_samples
from src.utils.path import resolve_data_path


LOGGER = logging.getLogger(__name__)
CANDIDATE_ARTIFACTS: Final[tuple[str, ...]] = CANDIDATE_MODEL_ARTIFACTS
LSTM_BATCH_SIZE: Final = 1024


def _shared_time_cuts(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    times = pd.to_datetime(frame["dataTime"], utc=True, errors="coerce")
    unique_times = pd.DatetimeIndex(times.dropna().unique()).sort_values()
    if len(unique_times) < 20:
        raise ValueError("At least 20 unique hours are required for evaluation")
    validation_index = max(1, math.floor(len(unique_times) * 0.70))
    test_index = max(validation_index + 1, math.floor(len(unique_times) * 0.85))
    return unique_times[validation_index], unique_times[test_index]


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if actual.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0.0}
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "n": float(actual.size),
    }


def _predict_ensemble(
    bundle: ArtifactBundle,
    samples: SequenceSamples,
) -> np.ndarray:
    count, sequence_length, feature_count = samples.sequences.shape
    scaled_sequences = bundle.scaler.transform(
        samples.sequences.reshape(-1, feature_count)
    ).reshape(count, sequence_length, feature_count).astype(np.float32)
    scaled_current = bundle.scaler.transform(samples.current_features).astype(
        np.float32
    )
    lstm_predictions = np.empty(count, dtype=np.float32)
    bundle.lstm_model.eval()
    with torch.no_grad():
        for start in range(0, count, LSTM_BATCH_SIZE):
            end = min(start + LSTM_BATCH_SIZE, count)
            batch = torch.from_numpy(scaled_sequences[start:end])
            lstm_predictions[start:end] = (
                bundle.lstm_model(batch).detach().cpu().numpy()
            )
    xgb_predictions = bundle.xgb_model.predict(scaled_current)
    weight = bundle.ensemble_lstm_weight
    return weight * lstm_predictions + (1.0 - weight) * xgb_predictions


def _split_mask(
    target_times: pd.DatetimeIndex,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    split: str,
) -> np.ndarray:
    if split == "validation":
        return np.asarray(
            (target_times >= validation_start) & (target_times < test_start)
        )
    if split == "test":
        return np.asarray(target_times >= test_start)
    raise ValueError("split must be validation or test")


def _evaluate_bundle(
    artifact_name: str,
    frame: pd.DataFrame,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> dict[str, Any]:
    bundle = load_artifacts(artifact_name)
    prepared = ensure_model_features(
        frame,
        bundle.feature_names,
        station_categories=bundle.metadata.get("station_categories"),
    )
    samples = build_sequence_samples(
        prepared,
        bundle.feature_names,
        bundle.sequence_length,
    )
    predictions = _predict_ensemble(bundle, samples)
    actual = samples.targets.astype(np.float64)
    predicted = predictions.astype(np.float64)
    rows = pd.DataFrame(
        {
            "station_name": samples.station_names.astype(str),
            "target_time": samples.target_times,
            "actual": actual,
            "predicted": predicted,
        }
    )
    split_metrics: dict[str, dict[str, float]] = {}
    for split in ("validation", "test"):
        mask = _split_mask(
            samples.target_times,
            validation_start,
            test_start,
            split,
        )
        split_metrics[split] = _regression_metrics(
            actual[mask],
            predicted[mask],
        )
    LOGGER.info(
        "%s: sequence_length=%d features=%d test_n=%.0f rmse=%.4f mae=%.4f r2=%.4f",
        artifact_name,
        bundle.sequence_length,
        len(bundle.feature_names),
        split_metrics["test"]["n"],
        split_metrics["test"]["rmse"],
        split_metrics["test"]["mae"],
        split_metrics["test"]["r2"],
    )
    return {
        "artifact_name": artifact_name,
        "sequence_length": bundle.sequence_length,
        "feature_count": len(bundle.feature_names),
        "ensemble_lstm_weight": bundle.ensemble_lstm_weight,
        "metrics": split_metrics,
        "own_holdout_metrics": bundle.metadata.get("metrics", {}).get("ensemble", {}),
        "predictions": rows,
    }


def _matched_metrics(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], int]:
    merged = left.merge(
        right,
        on=["station_name", "target_time"],
        suffixes=("_left", "_right"),
        how="inner",
    )
    actual = merged["actual_left"].to_numpy(dtype=np.float64)
    return (
        _regression_metrics(actual, merged["predicted_left"].to_numpy(dtype=np.float64)),
        _regression_metrics(actual, merged["predicted_right"].to_numpy(dtype=np.float64)),
        int(len(merged)),
    )


def _select_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda item: (
            item["rmse"],
            item["mae"],
            -item["r2"],
        ),
    )
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    reasons: list[str] = []
    if runner_up is None:
        reasons.append("비교 대상이 하나뿐입니다.")
    else:
        if winner["rmse"] < runner_up["rmse"]:
            reasons.append(
                f"RMSE가 {winner['rmse']:.4f}로 {runner_up['artifact_name']} "
                f"({runner_up['rmse']:.4f})보다 낮습니다."
            )
        if winner["mae"] < runner_up["mae"]:
            reasons.append(
                f"MAE가 {winner['mae']:.4f}로 {runner_up['artifact_name']} "
                f"({runner_up['mae']:.4f})보다 낮습니다."
            )
        if winner["r2"] > runner_up["r2"]:
            reasons.append(
                f"R²가 {winner['r2']:.4f}로 {runner_up['artifact_name']} "
                f"({runner_up['r2']:.4f})보다 높습니다."
            )
        if not reasons:
            reasons.append("동점 시 RMSE, MAE, 그다음 -R² 순으로 선정했습니다.")
    return {
        "artifact_name": winner["artifact_name"],
        "reason": " ".join(reasons),
    }


def _format_table(rows: list[dict[str, Any]]) -> str:
    headers = ("model", "split", "n", "RMSE", "MAE", "R²")
    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["artifact_name"]),
                    str(row["split"]),
                    f"{int(row['n'])}",
                    f"{row['rmse']:.4f}",
                    f"{row['mae']:.4f}",
                    f"{row['r2']:.4f}",
                ]
            )
        )
    return "\n".join(lines)


def evaluate_candidates(
    *,
    input_relative_path: str = "engineered_air_weather.csv",
    artifact_names: tuple[str, ...] = CANDIDATE_ARTIFACTS,
) -> dict[str, Any]:
    input_path = resolve_data_path(input_relative_path)
    if not input_path.is_file():
        raise FileNotFoundError("Engineered evaluation dataset was not found")
    frame = pd.read_csv(input_path)
    if "pm25_target_t24" not in frame.columns:
        raise ValueError("Evaluation data must include pm25_target_t24")
    validation_start, test_start = _shared_time_cuts(frame)

    evaluations: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for artifact_name in artifact_names:
        result = _evaluate_bundle(
            artifact_name,
            frame,
            validation_start,
            test_start,
        )
        evaluations.append(result)
        for split, metrics in result["metrics"].items():
            table_rows.append(
                {
                    "artifact_name": artifact_name,
                    "split": split,
                    **metrics,
                }
            )

    matched_rows: list[dict[str, Any]] = []
    if len(evaluations) == 2:
        test_frames = []
        for result in evaluations:
            mask = result["predictions"]["target_time"] >= test_start
            test_frames.append(result["predictions"].loc[mask].copy())
        left_metrics, right_metrics, matched_n = _matched_metrics(
            test_frames[0],
            test_frames[1],
        )
        for result, metrics in zip(
            evaluations,
            (left_metrics, right_metrics),
            strict=True,
        ):
            matched_row = {
                "artifact_name": result["artifact_name"],
                "split": "test_matched",
                **metrics,
            }
            matched_row["n"] = float(matched_n)
            matched_rows.append(matched_row)
            table_rows.append(matched_row)

    decision_rows = matched_rows or [
        row for row in table_rows if row["split"] == "test"
    ]
    winner = _select_winner(decision_rows)
    summary = {
        "target": "pm25_target_t24",
        "dataset": input_relative_path,
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "metrics": [
            {
                key: value
                for key, value in row.items()
            }
            for row in table_rows
        ],
        "winner": winner,
    }
    return summary


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="engineered_air_weather.csv",
        help="CSV path relative to data/",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    arguments = _parse_arguments()
    summary = evaluate_candidates(input_relative_path=arguments.input)
    print(_format_table(summary["metrics"]))
    print()
    print(f"Winner: {summary['winner']['artifact_name']}")
    print(summary["winner"]["reason"])
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
