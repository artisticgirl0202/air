from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd

from src.generate_dataset import (
    DEFAULT_OUTPUT,
    DEFAULT_SEOUL_STATIONS,
    build_station_mapping,
    export_engineered_dataset,
    fetch_kma_data,
    parse_station_names,
)
from src.preprocessing import PreprocessingError, preprocess_and_engineer_features
from src.utils.path import resolve_under_root
from src.weather_client import KMAAPIError, KMAWeatherClient


LOGGER = logging.getLogger(__name__)
KST: Final = "Asia/Seoul"
RAW_DIRECTORY: Final = "raw"
ALLOWED_SUFFIXES: Final = {".xlsx", ".csv"}
RAW_COLUMN_MAP: Final = {
    "측정소명": "stationName",
    "측정소코드": "stationCode",
    "측정일시": "dataTime",
    "PM25": "pm25Value",
    "PM10": "pm10Value",
    "SO2": "so2Value",
    "CO": "coValue",
    "O3": "o3Value",
    "NO2": "no2Value",
    "주소": "addr",
    "지역": "sidoName",
}
REQUIRED_RAW_COLUMNS: Final = ("측정소명", "측정일시", "PM25")


def resolve_raw_directory(relative_path: str | Path = RAW_DIRECTORY) -> Path:
    """Resolve data/raw (or another data/ subdirectory) without traversal."""
    directory = resolve_under_root("data", relative_path)
    if not directory.is_dir():
        raise ValueError("The raw data directory was not found under data/")
    return directory


def list_raw_files(directory: Path) -> list[Path]:
    """Return allowlisted Excel/CSV files directly inside the raw directory."""
    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in ALLOWED_SUFFIXES
        and not path.name.startswith("~$")
    ]
    files.sort(key=lambda path: path.name)
    if not files:
        raise ValueError("No .xlsx or .csv files were found in data/raw")
    return files


def parse_raw_measure_times(values: pd.Series) -> pd.Series:
    """Parse AirKorea ``YYYYMMDDHH`` stamps, including hour 24 as next-day 00:00."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.9:
        digits = numeric.round().astype("Int64").astype("string").str.zfill(10)
    else:
        digits = (
            values.astype("string")
            .str.replace(r"\D", "", regex=True)
            .str.zfill(10)
        )
    if digits.str.fullmatch(r"\d{10}").fillna(False).eq(False).any():
        raise ValueError("Raw measurement times must use YYYYMMDDHH")
    hour = pd.to_numeric(digits.str.slice(8, 10), errors="coerce")
    if hour.isna().any() or hour.lt(0).any() or hour.gt(24).any():
        raise ValueError("Raw measurement hours must be between 00 and 24")
    base = pd.to_datetime(digits.str.slice(0, 8), format="%Y%m%d", errors="coerce")
    if base.isna().any():
        raise ValueError("Raw measurement dates are invalid")
    midnight = hour.eq(24)
    hour_offset = hour.mask(midnight, 0)
    parsed = base + pd.to_timedelta(hour_offset, unit="h")
    parsed.loc[midnight] = base.loc[midnight] + pd.Timedelta(days=1)
    return parsed.dt.tz_localize(KST)


def _normalize_raw_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns=lambda name: str(name).strip())
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(
            "A raw file is missing required columns: " + ", ".join(missing)
        )
    return renamed


def _wanted_raw_columns(name: object) -> bool:
    return str(name).strip() in RAW_COLUMN_MAP


def _read_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(
            path,
            sheet_name=0,
            engine="openpyxl",
            usecols=_wanted_raw_columns,
        )
    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=_wanted_raw_columns,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            encoding="cp949",
            usecols=_wanted_raw_columns,
        )


def load_raw_airkorea(
    directory: Path,
    *,
    station_names: list[str] | None,
    seoul_only: bool,
) -> pd.DataFrame:
    """Read every raw file and keep only the requested station rows."""
    frames: list[pd.DataFrame] = []
    for path in list_raw_files(directory):
        LOGGER.info("Reading %s", path.name)
        raw = _normalize_raw_columns(_read_raw_table(path))
        selected = raw.copy()
        selected["측정소명"] = selected["측정소명"].astype("string").str.strip()
        if seoul_only and "지역" in selected.columns:
            region = selected["지역"].astype("string")
            selected = selected.loc[region.str.contains("서울", na=False)]
        if station_names:
            selected = selected.loc[selected["측정소명"].isin(station_names)]
        if selected.empty:
            LOGGER.warning("No matching stations in %s", path.name)
            continue
        keep = [column for column in RAW_COLUMN_MAP if column in selected.columns]
        frames.append(selected.loc[:, keep].copy())
    if not frames:
        raise ValueError("No matching AirKorea rows were found in data/raw")

    combined = pd.concat(frames, ignore_index=True, copy=False)
    combined = combined.rename(columns=RAW_COLUMN_MAP)
    combined["dataTime"] = parse_raw_measure_times(combined["dataTime"])
    combined["dataTime"] = combined["dataTime"].dt.strftime("%Y-%m-%d %H:00")
    combined = (
        combined.dropna(subset=["stationName", "dataTime"])
        .drop_duplicates(["stationName", "dataTime"], keep="last")
        .sort_values(["stationName", "dataTime"])
        .reset_index(drop=True)
    )
    LOGGER.info(
        "Loaded %d hourly rows for %d station(s)",
        len(combined),
        combined["stationName"].nunique(),
    )
    return combined


def _date_span(air_data: pd.DataFrame) -> tuple[date, date]:
    times = pd.to_datetime(air_data["dataTime"], errors="coerce")
    if times.isna().any():
        raise ValueError("Normalized measurement times are invalid")
    start = times.min().date()
    end = times.max().date()
    if start > end:
        raise ValueError("Raw files produced an inverted date range")
    return start, end


def build_dataset_from_raw(
    *,
    raw_relative_path: str | Path = RAW_DIRECTORY,
    station_names: list[str] | None = None,
    seoul_only: bool = False,
    output_relative_path: str | Path = DEFAULT_OUTPUT,
    persist_db: bool = False,
) -> Path:
    """Parse local AirKorea files, fetch matching KMA weather, and engineer features."""
    stations = parse_station_names(station_names) if station_names else None
    air_data = load_raw_airkorea(
        resolve_raw_directory(raw_relative_path),
        station_names=stations,
        seoul_only=seoul_only,
    )
    start, end = _date_span(air_data)
    mapping = build_station_mapping(air_data)
    LOGGER.info("Fetching KMA ASOS weather from %s to %s", start, end)
    with KMAWeatherClient() as weather_client:
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
    output = export_engineered_dataset(engineered, output_relative_path)
    if persist_db:
        from src.database import initialize_timescaledb, save_measurements
        from src.preprocessing import to_database_frame

        LOGGER.info("Upserting measurements into TimescaleDB")
        initialize_timescaledb()
        save_measurements(to_database_frame(engineered))
    LOGGER.info("Wrote %d engineered rows to data/%s", len(engineered), output.name)
    return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=RAW_DIRECTORY,
        help="Directory relative to data/ that contains monthly AirKorea files",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=list(DEFAULT_SEOUL_STATIONS),
        help='Station names, e.g. "종로구,관악구,서초구,강남구,마포구"',
    )
    parser.add_argument(
        "--all-seoul",
        action="store_true",
        help="Keep every station whose 지역 value contains 서울",
    )
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
        stations = None if arguments.all_seoul else parse_station_names(arguments.stations)
        build_dataset_from_raw(
            raw_relative_path=arguments.raw_dir,
            station_names=stations,
            seoul_only=arguments.all_seoul,
            output_relative_path=arguments.output,
            persist_db=arguments.persist_db,
        )
    except ValueError as exc:
        LOGGER.error("ValueError: %s", exc)
        raise SystemExit(1) from None
    except (KMAAPIError, PreprocessingError) as exc:
        LOGGER.error("Raw dataset build failed: %s", exc)
        raise SystemExit(1) from None
    except Exception:
        LOGGER.error("Raw dataset build failed unexpectedly")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
