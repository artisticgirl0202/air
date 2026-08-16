

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd


EARTH_RADIUS_KM: Final = 6_371.0088
DEFAULT_ASOS_STATIONS: Final = (
    {"stnId": 108, "station_name": "서울", "latitude": 37.5714, "longitude": 126.9658},
    {"stnId": 112, "station_name": "인천", "latitude": 37.4777, "longitude": 126.6249},
    {"stnId": 119, "station_name": "수원", "latitude": 37.2575, "longitude": 126.9830},
    {"stnId": 133, "station_name": "대전", "latitude": 36.3719, "longitude": 127.3721},
    {"stnId": 143, "station_name": "대구", "latitude": 35.8779, "longitude": 128.6529},
    {"stnId": 152, "station_name": "울산", "latitude": 35.5824, "longitude": 129.3347},
    {"stnId": 156, "station_name": "광주", "latitude": 35.1729, "longitude": 126.8916},
    {"stnId": 159, "station_name": "부산", "latitude": 35.1047, "longitude": 129.0320},
    {"stnId": 184, "station_name": "제주", "latitude": 33.5141, "longitude": 126.5297},
)
STATIC_NAME_TO_STATION: Final = (
    ("종로구", 108),
    ("관악구", 108),
    ("서초구", 108),
    ("강남구", 108),
    ("마포구", 108),
    ("서울", 108),
    ("인천", 112),
    ("수원", 119),
    ("대전", 133),
    ("대구", 143),
    ("울산", 152),
    ("광주", 156),
    ("부산", 159),
    ("제주", 184),
)


def default_weather_stations() -> pd.DataFrame:
    """Return the built-in major-city ASOS coordinate catalog."""
    return pd.DataFrame(DEFAULT_ASOS_STATIONS).copy()


def haversine_distance_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: np.ndarray,
    longitude_b: np.ndarray,
) -> np.ndarray:
    """Calculate vectorized great-circle distances in kilometers."""
    values = (latitude_a, longitude_a)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Origin coordinates must be finite")
    if not -90.0 <= latitude_a <= 90.0 or not -180.0 <= longitude_a <= 180.0:
        raise ValueError("Origin coordinates are outside valid ranges")

    lat_a = np.radians(latitude_a)
    lon_a = np.radians(longitude_a)
    lat_b = np.radians(latitude_b.astype(float))
    lon_b = np.radians(longitude_b.astype(float))
    delta_latitude = lat_b - lat_a
    delta_longitude = lon_b - lon_a
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat_a)
        * np.cos(lat_b)
        * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(
        np.sqrt(np.clip(haversine, 0.0, 1.0))
    )


def _first_numeric_column(
    frame: pd.DataFrame, candidates: tuple[str, ...]
) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _coordinate_columns(
    frame: pd.DataFrame, *, is_airkorea: bool
) -> tuple[pd.Series, pd.Series]:
    if is_airkorea:
        latitude_candidates = ("latitude", "lat", "dmX")
        longitude_candidates = ("longitude", "lon", "lng", "dmY")
    else:
        latitude_candidates = ("latitude", "lat")
        longitude_candidates = ("longitude", "lon", "lng")
    return (
        _first_numeric_column(frame, latitude_candidates),
        _first_numeric_column(frame, longitude_candidates),
    )


def _valid_coordinates(
    latitudes: pd.Series, longitudes: pd.Series
) -> pd.Series:
    return (
        latitudes.between(-90.0, 90.0, inclusive="both")
        & longitudes.between(-180.0, 180.0, inclusive="both")
        & np.isfinite(latitudes)
        & np.isfinite(longitudes)
    )


def _static_station_id(row: pd.Series) -> int | None:
    searchable_columns = (
        "stationName",
        "addr",
        "address",
        "sidoName",
        "city",
    )
    searchable = " ".join(
        str(row[column])
        for column in searchable_columns
        if column in row.index and pd.notna(row[column])
    )
    for token, station_id in STATIC_NAME_TO_STATION:
        if token in searchable:
            return station_id
    return None


def get_station_mapping(
    air_stations_df: pd.DataFrame,
    weather_stations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map each AirKorea station to its nearest ASOS station.

    Dynamic Haversine matching is preferred when both coordinate catalogs are
    available. Rows without coordinates use a conservative major-city mapping.
    """
    if not isinstance(air_stations_df, pd.DataFrame):
        raise TypeError("air_stations_df must be a pandas DataFrame")
    if not isinstance(weather_stations_df, pd.DataFrame):
        raise TypeError("weather_stations_df must be a pandas DataFrame")
    if "stationName" not in air_stations_df.columns:
        raise ValueError("air_stations_df requires stationName")
    if "stnId" not in weather_stations_df.columns:
        raise ValueError("weather_stations_df requires stnId")
    if air_stations_df.empty:
        raise ValueError("air_stations_df cannot be empty")

    air = air_stations_df.copy()
    weather = weather_stations_df.copy()
    if "stationCode" not in air.columns:
        air["stationCode"] = pd.NA
    air["stationName"] = air["stationName"].astype("string").str.strip()
    if air["stationName"].isna().any() or air["stationName"].eq("").any():
        raise ValueError("AirKorea station names cannot be empty")
    weather["stnId"] = pd.to_numeric(
        weather["stnId"], errors="coerce"
    ).astype("Int64")

    air_latitude, air_longitude = _coordinate_columns(
        air, is_airkorea=True
    )
    weather_latitude, weather_longitude = _coordinate_columns(
        weather, is_airkorea=False
    )
    valid_air = _valid_coordinates(air_latitude, air_longitude)
    valid_weather = _valid_coordinates(weather_latitude, weather_longitude)
    valid_weather &= weather["stnId"].notna()

    mapping_rows: list[dict[str, object]] = []
    unresolved: list[str] = []
    for index, station in air.iterrows():
        station_id: int | None = None
        distance_km = float("nan")
        if bool(valid_air.loc[index]) and valid_weather.any():
            candidates = weather.loc[valid_weather]
            distances = haversine_distance_km(
                float(air_latitude.loc[index]),
                float(air_longitude.loc[index]),
                weather_latitude.loc[valid_weather].to_numpy(dtype=float),
                weather_longitude.loc[valid_weather].to_numpy(dtype=float),
            )
            nearest_position = int(np.argmin(distances))
            station_id = int(candidates.iloc[nearest_position]["stnId"])
            distance_km = float(distances[nearest_position])
        else:
            station_id = _static_station_id(station)
            if station_id is not None and bool(valid_air.loc[index]):
                default_catalog = default_weather_stations().set_index("stnId")
                default_station = default_catalog.loc[station_id]
                distance_km = float(
                    haversine_distance_km(
                        float(air_latitude.loc[index]),
                        float(air_longitude.loc[index]),
                        np.array([default_station["latitude"]]),
                        np.array([default_station["longitude"]]),
                    )[0]
                )

        if station_id is None:
            unresolved.append(str(station["stationName"]))
            continue
        mapping_rows.append(
            {
                "stationName": str(station["stationName"]),
                "stationCode": station["stationCode"],
                "stnId": station_id,
                "distance_km": distance_km,
            }
        )

    if unresolved:
        raise ValueError(
            "ASOS station mapping failed for: " + ", ".join(unresolved)
        )
    result = pd.DataFrame(
        mapping_rows,
        columns=["stationName", "stationCode", "stnId", "distance_km"],
    )
    if result.duplicated(["stationName"]).any():
        raise ValueError("AirKorea station names must be unique")
    return result
