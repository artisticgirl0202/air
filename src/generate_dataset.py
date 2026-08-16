from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final

import pandas as pd

from src.data_loader.api_client import AirKoreaClient, AirKoreaError
from src.database import ensure_measurement_schema, save_measurements
from src.preprocessing import (
    PreprocessingError,
    preprocess_and_engineer_features,
    to_database_frame,
)
from src.station_mapping import (
    default_weather_stations,
    get_station_mapping,
)
from src.utils.path import resolve_data_path as secure_data_path
from src.weather_client import KMAAPIError, KMAWeatherClient


LOGGER = logging.getLogger(__name__)
KST: Final = "Asia/Seoul"
DEFAULT_OUTPUT: Final = "engineered_air_weather.csv"
DEFAULT_SEOUL_STATIONS: Final = (
    "종로구",
    "관악구",
    "서초구",
    "강남구",
    "마포구",
)


def resolve_data_path(relative_path: str | Path) -> Path:
    """Resolve an output below data/ and reject absolute/traversal paths."""
    resolved = secure_data_path(relative_path)
    if resolved.suffix.lower() != ".csv":
        raise ValueError("Output file must use the .csv extension")
    return resolved


def _parse_iso_date(value: str, name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        raise ValueError(f"{name} must use YYYY-MM-DD format") from None


def parse_station_names(raw_stations: list[str]) -> list[str]:
    """Accept space-separated names or a single comma-separated string."""
    names: list[str] = []
    for item in raw_stations:
        names.extend(part.strip() for part in str(item).split(","))
    cleaned = [name for name in names if name]
    if not cleaned or len(cleaned) > 20:
        raise ValueError("Provide between 1 and 20 AirKorea stations")
    return list(dict.fromkeys(cleaned))


def resolve_collection_window(
    *,
    days: int,
    start_date: str | None,
    end_date: str | None,
    today: date | None = None,
) -> tuple[date, date]:
    """Return an inclusive date window of 2 to 366 days."""
    if isinstance(days, bool) or not isinstance(days, int) or not 2 <= days <= 366:
        raise ValueError("--days must be an integer between 2 and 366")
    if (start_date is None) != (end_date is None):
        raise ValueError("--start-date and --end-date must be provided together")

    reference_date = today or date.today()
    latest_complete_date = reference_date - timedelta(days=1)
    if start_date is not None and end_date is not None:
        start = _parse_iso_date(start_date, "--start-date")
        end = _parse_iso_date(end_date, "--end-date")
    else:
        end = latest_complete_date
        start = end - timedelta(days=days - 1)

    if start > end:
        raise ValueError("Start date must not be after end date")
    if end > latest_complete_date:
        raise ValueError("End date must be yesterday or earlier for ASOS data")
    window_days = (end - start).days + 1
    if window_days < 2 or window_days > 366:
        raise ValueError("Collection window must be between 2 and 366 days")
    return start, end


def _airkorea_data_term(window_days: int) -> str:
    if window_days <= 2:
        return "DAILY"
    if window_days <= 31:
        return "MONTH"
    return "3MONTH"


def parse_airkorea_times(values: pd.Series) -> pd.Series:
    """Parse AirKorea timestamps, including its non-standard ``24:00``."""
    text = values.astype("string").str.strip()
    midnight_mask = text.str.fullmatch(r"\d{4}-\d{2}-\d{2} 24:00")
    regular_values = text.mask(midnight_mask)
    parsed = pd.to_datetime(
        regular_values,
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    if midnight_mask.any():
        midnight_dates = pd.to_datetime(
            text.loc[midnight_mask].str.slice(0, 10),
            format="%Y-%m-%d",
            errors="coerce",
        ) + pd.Timedelta(days=1)
        parsed.loc[midnight_mask] = midnight_dates
    if parsed.isna().any():
        raise ValueError("AirKorea returned invalid observation times")
    return parsed


async def fetch_airkorea_data(
    station_names: list[str],
    *,
    start: date,
    end: date,
    include_daily: bool = True,
) -> pd.DataFrame:
    """Fetch rolling AirKorea observations and filter the requested dates."""
    station_names = parse_station_names(station_names)
    window_days = (end - start).days + 1
    LOGGER.info(
        "Fetching AirKorea observations for %d station(s) over %d days",
        len(station_names),
        window_days,
    )
    if window_days > 92:
        LOGGER.warning(
            "AirKorea hourly OpenAPI typically exposes the latest ~3 months. "
            "Older hours are filled from daily station statistics when available."
        )
    async with AirKoreaClient(
        concurrency=2,
        request_pause_seconds=0.5,
    ) as client:
        station_items = await client.fetch_many(
            station_names,
            rows=1_000,
            data_term=_airkorea_data_term(min(window_days, 92)),
        )
        daily_records: list[dict[str, object]] = []
        if include_daily:
            for station_name in station_names:
                for chunk_start, chunk_end in _iter_date_chunks(start, end):
                    try:
                        daily_items = await client.fetch_daily_station(
                            station_name,
                            start_date=chunk_start.strftime("%Y%m%d"),
                            end_date=chunk_end.strftime("%Y%m%d"),
                        )
                    except AirKoreaError as exc:
                        LOGGER.warning(
                            "Daily AirKorea history unavailable for %s (%s–%s): %s",
                            station_name,
                            chunk_start.isoformat(),
                            chunk_end.isoformat(),
                            exc,
                        )
                        daily_items = []
                    for item in daily_items:
                        record = dict(item)
                        record.setdefault("stationName", station_name)
                        daily_records.append(record)
                    await asyncio.sleep(0.5)

    records: list[dict[str, object]] = []
    for station_name, items in station_items.items():
        for item in items:
            record = dict(item)
            record.setdefault("stationName", station_name)
            records.append(record)
    if not records and not daily_records:
        raise ValueError("AirKorea returned no observations")

    if records:
        frame = pd.DataFrame(records)
        if "dataTime" not in frame.columns:
            raise ValueError("AirKorea response does not contain dataTime")
        parsed_time = parse_airkorea_times(frame["dataTime"])
    else:
        frame = pd.DataFrame()
        parsed_time = pd.Series(dtype="datetime64[ns, Asia/Seoul]")
    if parsed_time.dt.tz is None:
        parsed_time = parsed_time.dt.tz_localize(KST)
    else:
        parsed_time = parsed_time.dt.tz_convert(KST)

    start_time = pd.Timestamp(start).tz_localize(KST)
    end_exclusive = pd.Timestamp(end + timedelta(days=1)).tz_localize(KST)
    if frame.empty:
        selected = pd.DataFrame()
    else:
        selected = frame.loc[
            parsed_time.ge(start_time) & parsed_time.lt(end_exclusive)
        ].copy()
        if not selected.empty:
            selected["dataTime"] = (
                parsed_time.loc[selected.index].dt.strftime("%Y-%m-%d %H:00")
            )
    if daily_records:
        daily_hourly = _expand_daily_to_hourly(
            pd.DataFrame(daily_records),
            start=start,
            end=end,
        )
        if not daily_hourly.empty:
            selected = pd.concat(
                [daily_hourly, selected],
                ignore_index=True,
                copy=False,
            )
            selected = selected.drop_duplicates(
                ["stationName", "dataTime"],
                keep="last",
            )
    if selected.empty or "dataTime" not in selected.columns:
        raise ValueError(
            "AirKorea has no data in the requested rolling window"
        )
    selected = selected.sort_values(
        ["stationName", "dataTime"]
    ).reset_index(drop=True)
    _log_airkorea_coverage(selected, start=start, end=end)
    return selected


def _iter_date_chunks(
    start: date,
    end: date,
    *,
    chunk_days: int = 30,
) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        windows.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return windows


def _normalize_airkorea_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Map daily-stat aliases onto the hourly AirKorea column names."""
    aliases = {
        "stationName": ("stationName", "msrstnName", "station_name"),
        "dataTime": ("msrDt", "dataTime", "msurDt", "msurDate"),
        "pm10Value": ("pm10Value", "pm10Value24", "pm10Avg"),
        "pm25Value": ("pm25Value", "pm25Value24", "pm25Avg"),
        "so2Value": ("so2Value", "so2Value24", "so2Avg"),
        "coValue": ("coValue", "coValue24", "coAvg"),
        "o3Value": ("o3Value", "o3Value24", "o3Avg"),
        "no2Value": ("no2Value", "no2Value24", "no2Avg"),
        "khaiValue": ("khaiValue", "khaiValue24", "khaiAvg"),
    }
    renamed = daily.copy()
    for canonical, candidates in aliases.items():
        source = next((name for name in candidates if name in renamed.columns), None)
        if source is not None and source != canonical:
            renamed[canonical] = renamed[source]
    return renamed


def _expand_daily_to_hourly(
    daily: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Repeat daily averages across 24 hours when hourly rows are missing."""
    if daily.empty:
        return daily
    normalized = _normalize_airkorea_daily(daily)
    if "dataTime" not in normalized.columns:
        return pd.DataFrame()
    day = pd.to_datetime(normalized["dataTime"], errors="coerce")
    normalized = normalized.assign(_day=day.dt.normalize()).dropna(subset=["_day"])
    in_window = (normalized["_day"].dt.date >= start) & (
        normalized["_day"].dt.date <= end
    )
    normalized = normalized.loc[in_window]
    if normalized.empty:
        return pd.DataFrame()
    hours = pd.DataFrame({"_hour": range(24)})
    expanded = normalized.assign(_join=1).merge(
        hours.assign(_join=1),
        on="_join",
        how="inner",
        copy=False,
    )
    expanded["dataTime"] = (
        expanded["_day"] + pd.to_timedelta(expanded["_hour"], unit="h")
    ).dt.strftime("%Y-%m-%d %H:00")
    return expanded.drop(columns=["_day", "_hour", "_join"])


def _log_airkorea_coverage(
    frame: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> None:
    expected = ((end - start).days + 1) * 24
    for station_name, group in frame.groupby("stationName", sort=False):
        coverage = 100.0 * len(group) / expected if expected else 0.0
        LOGGER.info(
            "AirKorea coverage %s: %d/%d hours (%.1f%%)",
            station_name,
            len(group),
            expected,
            coverage,
        )


def fetch_kma_data(
    station_ids: list[int],
    *,
    start: date,
    end: date,
    client: KMAWeatherClient,
) -> pd.DataFrame:
    """Fetch and concatenate ASOS hourly observations for mapped stations."""
    frames: list[pd.DataFrame] = []
    for station_id in sorted(set(station_ids)):
        frame = client.fetch_hourly_range(
            start=start,
            end=end,
            stnIds=station_id,
        )
        if frame.empty:
            LOGGER.warning(
                "KMA ASOS station %s returned no observations",
                station_id,
            )
            continue
        frames.append(frame)
    if not frames:
        raise ValueError("No KMA station IDs were mapped")
    combined = pd.concat(frames, ignore_index=True)
    combined["tm"] = combined["tm"].dt.strftime("%Y-%m-%d %H:00")
    return combined.sort_values(["stnId", "tm"]).reset_index(drop=True)


def build_station_mapping(air_data: pd.DataFrame) -> pd.DataFrame:
    station_columns = [
        column
        for column in (
            "stationName",
            "stationCode",
            "dmX",
            "dmY",
            "addr",
            "sidoName",
        )
        if column in air_data.columns
    ]
    station_catalog = air_data[station_columns].drop_duplicates(
        subset=["stationName"]
    )
    mapping = get_station_mapping(
        station_catalog,
        default_weather_stations(),
    )
    LOGGER.info("Mapped %d AirKorea station(s) to ASOS", len(mapping))
    return mapping


def export_engineered_dataset(
    frame: pd.DataFrame,
    output_relative_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Write the engineered CSV atomically inside the data directory."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Engineered dataset cannot be empty")
    output_path = resolve_data_path(output_relative_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    try:
        frame.to_csv(
            temporary_path,
            encoding="utf-8-sig",
            index=False,
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def generate_dataset(
    station_names: list[str],
    *,
    start: date,
    end: date,
    output_relative_path: str | Path = DEFAULT_OUTPUT,
    persist_db: bool = False,
    write_csv: bool = True,
    include_daily: bool = True,
) -> Path | None:
    """Run collection, spatial matching, feature engineering, and export."""
    # Instantiate both clients before network calls so missing keys fail early.
    with KMAWeatherClient() as weather_client:
        air_data = asyncio.run(
            fetch_airkorea_data(
                station_names,
                start=start,
                end=end,
                include_daily=include_daily,
            )
        )
        mapping = build_station_mapping(air_data)
        weather_data = fetch_kma_data(
            mapping["stnId"].astype(int).tolist(),
            start=start,
            end=end,
            client=weather_client,
        )

    LOGGER.info("Preprocessing and engineering time-series features")
    engineered = preprocess_and_engineer_features(
        air_data,
        weather_data,
        station_mapping=mapping,
        include_target=True,
    )
    output: Path | None = None
    if write_csv:
        output = export_engineered_dataset(engineered, output_relative_path)
        LOGGER.info("Generated %d rows at data/%s", len(engineered), output.name)
    if persist_db:
        LOGGER.info("Upserting measurements into TimescaleDB")
        ensure_measurement_schema()
        inserted = save_measurements(to_database_frame(engineered))
        LOGGER.info("Upserted %d measurements", inserted)
    return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stations",
        nargs="+",
        default=list(DEFAULT_SEOUL_STATIONS),
        help=(
            "AirKorea station names (max 20). "
            'Accepts spaces or commas, e.g. "종로구,관악구,서초구,강남구,마포구"'
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Inclusive lookback days (2-366). Example: --days 365",
    )
    parser.add_argument("--start-date", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", help="Inclusive YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CSV path relative to data/",
    )
    parser.add_argument(
        "--persist-db",
        action="store_true",
        help="Upsert generated observations into TimescaleDB",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = _parse_arguments()
    try:
        start, end = resolve_collection_window(
            days=arguments.days,
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )
        generate_dataset(
            parse_station_names(arguments.stations),
            start=start,
            end=end,
            output_relative_path=arguments.output,
            persist_db=arguments.persist_db,
        )
    except ValueError as exc:
        LOGGER.error("ValueError: %s", exc)
        raise SystemExit(1) from None
    except (AirKoreaError, KMAAPIError, PreprocessingError) as exc:
        LOGGER.error("Dataset generation failed: %s", exc)
        raise SystemExit(1) from None
    except Exception:
        LOGGER.error("Dataset generation failed unexpectedly")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
