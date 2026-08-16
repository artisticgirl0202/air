

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import numpy as np
import pandas as pd


KST: Final = "Asia/Seoul"
AIR_VALUE_COLUMNS: Final = (
    "pm10Value",
    "pm25Value",
    "so2Value",
    "coValue",
    "o3Value",
    "no2Value",
    "khaiValue",
)
WEATHER_VALUE_COLUMNS: Final = ("ta", "rn", "ws", "wd", "hm", "pa")
QC_COLUMNS: Final = {
    "ta": ("taQcflg", "taQcflag"),
    "rn": ("rnQcflg", "rnQcflag"),
    "ws": ("wsQcflg", "wsQcflag"),
    "wd": ("wdQcflg", "wdQcflag"),
    "hm": ("hmQcflg", "hmQcflag"),
    "pa": ("paQcflg", "paQcflag"),
}
INTERPOLATED_WEATHER_COLUMNS: Final = ("ta", "ws", "wd", "hm", "pa")
STATION_FEATURE_PREFIX: Final = "station__"
INTERACTION_FEATURES: Final = (
    "pm25_x_ta",
    "pm25_x_hm",
    "pm25_x_rn",
    "pm25_x_wind",
)
DB_COLUMN_MAP: Final = {
    "dataTime": "measured_at",
    "pm10Value": "pm10",
    "pm25Value": "pm25",
    "so2Value": "so2",
    "coValue": "co",
    "o3Value": "o3",
    "no2Value": "no2",
    "khaiValue": "aqi",
    "ta": "temperature",
    "rn": "rainfall",
    "hm": "humidity",
    "pa": "pressure",
    "ws": "wind_speed",
    "wd": "wind_direction",
}


class PreprocessingError(ValueError):
    """Raised for invalid input data without exposing source file paths."""


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], frame_name: str
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame")
    missing = set(required) - set(frame.columns)
    if missing:
        raise PreprocessingError(
            f"{frame_name} is missing columns: {', '.join(sorted(missing))}"
        )


def _to_kst(values: pd.Series, column_name: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values, errors="raise")
        if parsed.dt.tz is None:
            return parsed.dt.tz_localize(
                KST, ambiguous="raise", nonexistent="raise"
            )
        return parsed.dt.tz_convert(KST)
    except (TypeError, ValueError):
        raise PreprocessingError(
            f"{column_name} contains invalid timestamps"
        ) from None


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _interpolate_per_station(
    frame: pd.DataFrame,
    *,
    station_column: str,
    time_column: str,
    columns: Iterable[str],
    limit: int,
) -> pd.DataFrame:
    ordered = frame.sort_values([station_column, time_column]).copy()
    for column in columns:
        if column not in ordered.columns:
            continue
        interpolated: list[pd.Series] = []
        for _, group in ordered.groupby(station_column, sort=False):
            series = pd.Series(
                group[column].to_numpy(dtype=float),
                index=pd.DatetimeIndex(group[time_column]),
            )
            values = series.interpolate(
                method="time",
                limit=limit,
                limit_direction="both",
                limit_area="inside",
            )
            interpolated.append(
                pd.Series(values.to_numpy(), index=group.index, dtype=float)
            )
        if interpolated:
            ordered[column] = pd.concat(interpolated).sort_index()
    return ordered


def _apply_kma_quality_flags(weather: pd.DataFrame) -> None:
    """Set observations to NaN when their separate QC flag is 9.

    A measurement value of 9 is not itself missing. KMA documents 9 as the
    missing state in the corresponding *Qcflag field.
    """
    for value_column, flag_candidates in QC_COLUMNS.items():
        flag_column = next(
            (
                candidate
                for candidate in flag_candidates
                if candidate in weather.columns
            ),
            None,
        )
        if value_column not in weather.columns or flag_column is None:
            continue
        flags = pd.to_numeric(weather[flag_column], errors="coerce")
        weather.loc[flags.eq(9), value_column] = np.nan


def _attach_station_mapping(
    air: pd.DataFrame, station_mapping: pd.DataFrame | None
) -> pd.DataFrame:
    if "mapped_stnId" in air.columns:
        return air
    if station_mapping is None:
        raise PreprocessingError(
            "air_df needs mapped_stnId or a station_mapping DataFrame"
        )

    mapping = station_mapping.copy()
    if "stationName" in mapping.columns and "station_name" not in mapping.columns:
        mapping = mapping.rename(columns={"stationName": "station_name"})
    _require_columns(mapping, ("station_name", "stnId"), "station_mapping")

    station_column = (
        "stationName" if "stationName" in air.columns else "station_name"
    )
    _require_columns(air, (station_column,), "air_df")
    if mapping["station_name"].duplicated().any():
        raise PreprocessingError("station_mapping contains duplicate station names")

    mapped = air.merge(
        mapping[["station_name", "stnId"]],
        left_on=station_column,
        right_on="station_name",
        how="left",
        validate="many_to_one",
    )
    if mapped["stnId"].isna().any():
        raise PreprocessingError("Some AirKorea stations have no KMA mapping")
    return mapped.rename(columns={"stnId": "mapped_stnId"})


def _add_exact_lag(
    frame: pd.DataFrame,
    *,
    source_column: str,
    hours: int,
) -> pd.DataFrame:
    feature_name = f"{source_column.removesuffix('Value')}_lag{hours}"
    lookup = frame[["station_name", "dataTime", source_column]].copy()
    lookup["dataTime"] = lookup["dataTime"] + pd.Timedelta(hours=hours)
    lookup = lookup.rename(columns={source_column: feature_name})
    return frame.merge(
        lookup,
        on=["station_name", "dataTime"],
        how="left",
        validate="one_to_one",
    )


def _add_time_rolling_means(frame: pd.DataFrame) -> pd.DataFrame:
    result_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("station_name", sort=False):
        ordered = group.sort_values("dataTime").copy()
        indexed = ordered.set_index("dataTime")
        for source in ("pm10Value", "pm25Value"):
            for hours in (6, 24):
                minimum = max(2, hours // 2)
                ordered[f"{source.removesuffix('Value')}_ma{hours}"] = (
                    indexed[source]
                    .rolling(f"{hours}h", min_periods=minimum, closed="right")
                    .mean()
                    .to_numpy()
                )
        result_parts.append(ordered)
    return pd.concat(result_parts, ignore_index=True)


def encode_station_features(
    frame: pd.DataFrame,
    *,
    station_categories: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode station names for a single global model."""
    _require_columns(frame, ("station_name",), "feature frame")
    names = frame["station_name"].astype("string").str.strip()
    if station_categories is None:
        categories = sorted(
            str(name) for name in names.dropna().unique().tolist() if str(name)
        )
    else:
        categories = [str(name).strip() for name in station_categories if str(name).strip()]
    if not categories:
        raise PreprocessingError("At least one station category is required")
    encoded = frame.copy()
    for category in categories:
        encoded[f"{STATION_FEATURE_PREFIX}{category}"] = names.eq(category).astype(
            np.float32
        )
    return encoded, categories


def add_weather_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add current-time pollutant/weather products used by the global model."""
    result = frame.copy()
    pm25 = pd.to_numeric(result.get("pm25Value"), errors="coerce")
    temperature = pd.to_numeric(result.get("ta"), errors="coerce")
    humidity = pd.to_numeric(result.get("hm"), errors="coerce")
    rainfall = pd.to_numeric(result.get("rn"), errors="coerce")
    if {"wind_u", "wind_v"}.issubset(result.columns):
        wind_speed = np.hypot(
            pd.to_numeric(result["wind_u"], errors="coerce"),
            pd.to_numeric(result["wind_v"], errors="coerce"),
        )
    else:
        wind_speed = pd.to_numeric(result.get("ws"), errors="coerce")
    result["pm25_x_ta"] = pm25 * temperature
    result["pm25_x_hm"] = pm25 * humidity
    result["pm25_x_rn"] = pm25 * rainfall
    result["pm25_x_wind"] = pm25 * wind_speed
    return result


def add_t24_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add PM2.5 at the exact same station and timestamp 24 hours later."""
    _require_columns(
        frame,
        ("station_name", "dataTime", "pm25Value"),
        "feature frame",
    )
    target = frame[["station_name", "dataTime", "pm25Value"]].copy()
    target["dataTime"] = target["dataTime"] - pd.Timedelta(hours=24)
    target = target.rename(columns={"pm25Value": "pm25_target_t24"})
    return frame.merge(
        target,
        on=["station_name", "dataTime"],
        how="left",
        validate="one_to_one",
    )


def preprocess_and_engineer_features(
    air_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    *,
    station_mapping: pd.DataFrame | None = None,
    interpolation_limit: int = 3,
    include_target: bool = True,
) -> pd.DataFrame:
    """Merge hourly observations and build T+24 forecasting features.

    Naive source timestamps are interpreted as KST. Interpolation is performed
    independently per station and cannot bridge more than ``interpolation_limit``
    consecutive rows.
    """
    if (
        isinstance(interpolation_limit, bool)
        or not isinstance(interpolation_limit, int)
        or not 1 <= interpolation_limit <= 24
    ):
        raise ValueError("interpolation_limit must be an integer from 1 to 24")

    _require_columns(air_df, ("dataTime", "pm25Value"), "air_df")
    _require_columns(weather_df, ("tm", "stnId"), "weather_df")
    air = _attach_station_mapping(air_df.copy(), station_mapping)
    weather = weather_df.copy()

    station_column = (
        "stationName" if "stationName" in air.columns else "station_name"
    )
    _require_columns(air, (station_column, "mapped_stnId"), "air_df")
    air["station_name"] = air[station_column].astype("string").str.strip()
    if air["station_name"].isna().any() or air["station_name"].eq("").any():
        raise PreprocessingError("AirKorea station names cannot be empty")

    air["dataTime"] = _to_kst(air["dataTime"], "dataTime")
    weather["tm"] = _to_kst(weather["tm"], "tm")
    air["mapped_stnId"] = pd.to_numeric(
        air["mapped_stnId"], errors="coerce"
    ).astype("Int64")
    weather["stnId"] = pd.to_numeric(
        weather["stnId"], errors="coerce"
    ).astype("Int64")
    if air["mapped_stnId"].isna().any() or weather["stnId"].isna().any():
        raise PreprocessingError("Station IDs must be integers")

    _numeric(air, AIR_VALUE_COLUMNS)
    _apply_kma_quality_flags(weather)
    _numeric(weather, WEATHER_VALUE_COLUMNS)

    air = _interpolate_per_station(
        air,
        station_column="station_name",
        time_column="dataTime",
        columns=AIR_VALUE_COLUMNS,
        limit=interpolation_limit,
    )
    weather = _interpolate_per_station(
        weather,
        station_column="stnId",
        time_column="tm",
        columns=INTERPOLATED_WEATHER_COLUMNS,
        limit=interpolation_limit,
    )

    # Rain is an accumulated/event variable; linear interpolation would create
    # artificial rainfall. Missing values with a non-missing QC state become 0.
    if "rn" in weather.columns:
        rain_flag_column = next(
            (
                candidate
                for candidate in QC_COLUMNS["rn"]
                if candidate in weather.columns
            ),
            None,
        )
        missing_qc = (
            pd.to_numeric(weather[rain_flag_column], errors="coerce").eq(9)
            if rain_flag_column is not None
            else pd.Series(False, index=weather.index)
        )
        weather.loc[weather["rn"].isna() & ~missing_qc, "rn"] = 0.0

    weather_columns = [
        "tm",
        "stnId",
        *WEATHER_VALUE_COLUMNS,
        *(
            flag
            for flag_candidates in QC_COLUMNS.values()
            for flag in flag_candidates
        ),
    ]
    available_weather = [
        column for column in weather_columns if column in weather.columns
    ]
    air_keep = [
        column
        for column in (
            "station_name",
            "dataTime",
            "mapped_stnId",
            *AIR_VALUE_COLUMNS,
        )
        if column in air.columns
    ]
    air = air.loc[:, air_keep]
    merged = air.merge(
        weather[available_weather],
        left_on=["dataTime", "mapped_stnId"],
        right_on=["tm", "stnId"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise PreprocessingError(
            "No matching hourly station observations were found"
        )
    if merged.duplicated(["station_name", "dataTime"]).any():
        raise PreprocessingError("Merged data contains duplicate station hours")

    local_time = merged["dataTime"].dt.tz_convert(KST)
    merged["hour"] = local_time.dt.hour
    merged["month"] = local_time.dt.month
    merged["day_of_week"] = local_time.dt.dayofweek
    merged["hour_sin"] = np.sin(2 * np.pi * merged["hour"] / 24)
    merged["hour_cos"] = np.cos(2 * np.pi * merged["hour"] / 24)
    merged["month_sin"] = np.sin(2 * np.pi * (merged["month"] - 1) / 12)
    merged["month_cos"] = np.cos(2 * np.pi * (merged["month"] - 1) / 12)

    if {"wd", "ws"}.issubset(merged.columns):
        radians = np.deg2rad(merged["wd"])
        # Meteorological WD is where wind comes from.
        merged["wind_u"] = -merged["ws"] * np.sin(radians)
        merged["wind_v"] = -merged["ws"] * np.cos(radians)

    ordered = merged.sort_values(["station_name", "dataTime"]).reset_index(drop=True)
    for source in ("pm10Value", "pm25Value"):
        if source not in ordered.columns:
            continue
        for hours in (1, 24):
            ordered = _add_exact_lag(
                ordered, source_column=source, hours=hours
            )
    ordered = _add_time_rolling_means(ordered)
    ordered = add_weather_interactions(ordered)
    ordered, _ = encode_station_features(ordered)
    if include_target:
        ordered = add_t24_target(ordered)
    return ordered.sort_values(["dataTime", "station_name"]).reset_index(drop=True)


def to_database_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt engineered data to the allowlisted ORM measurement schema."""
    _require_columns(
        feature_frame,
        ("station_name", "dataTime", "pm25Value"),
        "feature_frame",
    )
    selected = ["station_name", *DB_COLUMN_MAP.keys(), "wind_u", "wind_v"]
    available = [column for column in selected if column in feature_frame.columns]
    result = feature_frame[available].rename(columns=DB_COLUMN_MAP).copy()
    result["measured_at"] = pd.to_datetime(
        result["measured_at"], utc=True, errors="raise"
    )
    # The ORM requires PM2.5; unresolved sensor outages cannot be persisted as
    # measurements or used as a reliable inference anchor.
    return result.dropna(
        subset=["station_name", "measured_at", "pm25"]
    ).reset_index(drop=True)
