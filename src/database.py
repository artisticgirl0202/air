from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, Index, String, create_engine, func, select, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

LOGGER = logging.getLogger(__name__)
_STATION_PATTERN = re.compile(r"^[가-힣A-Za-z0-9\s-]{1,50}$")
_DATABASE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_$-]{0,62}$")
_ALLOWED_SSLMODES = frozenset(
    {
        "disable",
        "allow",
        "prefer",
        "require",
        "verify-ca",
        "verify-full",
    }
)
_REQUIRED_COLUMNS = frozenset({"station_name", "measured_at", "pm25"})
_ALLOWED_COLUMNS = frozenset(
    {
        "station_name",
        "measured_at",
        "pm10",
        "pm25",
        "so2",
        "co",
        "o3",
        "no2",
        "aqi",
        "temperature",
        "rainfall",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_direction",
        "wind_u",
        "wind_v",
    }
)
_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "pm10": (0.0, 5_000.0),
    "pm25": (0.0, 5_000.0),
    "so2": (0.0, 10.0),
    "co": (0.0, 100.0),
    "o3": (0.0, 10.0),
    "no2": (0.0, 10.0),
    "aqi": (0.0, 5_000.0),
    "temperature": (-100.0, 100.0),
    "rainfall": (0.0, 2_000.0),
    "humidity": (0.0, 100.0),
    "pressure": (800.0, 1_200.0),
    "wind_speed": (0.0, 150.0),
    "wind_direction": (0.0, 360.0),
    "wind_u": (-150.0, 150.0),
    "wind_v": (-150.0, 150.0),
}


class DatabaseConfigurationError(RuntimeError):
    """Raised without revealing connection credentials."""


class DatabaseOperationError(RuntimeError):
    """Safe database error suitable for presentation to a UI."""


@dataclass(frozen=True)
class DatabaseSettings:
    """Validated database settings loaded exclusively from environment values."""

    host: str
    port: int
    user: str
    password: str = field(repr=False)
    database: str

    @classmethod
    def from_environment(cls) -> "DatabaseSettings":
        variable_names = (
            "DB_HOST",
            "DB_PORT",
            "DB_USER",
            "DB_PASSWORD",
            "DB_NAME",
        )
        values = {
            name: os.getenv(name, "").strip()
            for name in variable_names
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise DatabaseConfigurationError(
                "Missing database environment variables: "
                + ", ".join(missing)
            )
        try:
            port = int(values["DB_PORT"])
        except ValueError as exc:
            raise DatabaseConfigurationError(
                "DB_PORT must be an integer"
            ) from exc
        if not 1 <= port <= 65_535:
            raise DatabaseConfigurationError(
                "DB_PORT is outside the valid range"
            )
        for variable_name in ("DB_USER", "DB_NAME"):
            if not _DATABASE_IDENTIFIER_PATTERN.fullmatch(
                values[variable_name]
            ):
                raise DatabaseConfigurationError(
                    f"{variable_name} contains invalid characters"
                )
        if any(character in values["DB_HOST"] for character in "\r\n"):
            raise DatabaseConfigurationError(
                "DB_HOST contains invalid characters"
            )
        return cls(
            host=values["DB_HOST"],
            port=port,
            user=values["DB_USER"],
            password=values["DB_PASSWORD"],
            database=values["DB_NAME"],
        )

    def _sslmode(self) -> str | None:
        configured = os.getenv("DB_SSLMODE", "").strip().lower()
        if configured:
            if configured not in _ALLOWED_SSLMODES:
                raise DatabaseConfigurationError("DB_SSLMODE is invalid")
            return configured
        host = self.host.lower()
        if "neon.tech" in host or "supabase.co" in host:
            return "require"
        return None

    def sqlalchemy_url(self) -> URL:
        sslmode = self._sslmode()
        query = {"sslmode": sslmode} if sslmode else {}
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
            query=query,
        )

    @property
    def safe_target(self) -> str:
        return f"{self.host}:{self.port}/{self.database} as {self.user}"


class Base(DeclarativeBase):
    pass


class AirQualityMeasurement(Base):
    """Hourly air-quality and weather observation."""

    __tablename__ = "air_quality_measurements"
    __table_args__ = (
        Index("ix_air_quality_measured_at", "measured_at"),
    )

    station_name: Mapped[str] = mapped_column(
        String(50), primary_key=True, index=True
    )
    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    pm10: Mapped[float | None] = mapped_column(Float)
    pm25: Mapped[float] = mapped_column(Float, nullable=False)
    so2: Mapped[float | None] = mapped_column(Float)
    co: Mapped[float | None] = mapped_column(Float)
    o3: Mapped[float | None] = mapped_column(Float)
    no2: Mapped[float | None] = mapped_column(Float)
    aqi: Mapped[float | None] = mapped_column(Float)
    temperature: Mapped[float | None] = mapped_column(Float)
    rainfall: Mapped[float | None] = mapped_column(Float)
    humidity: Mapped[float | None] = mapped_column(Float)
    pressure: Mapped[float | None] = mapped_column(Float)
    wind_speed: Mapped[float | None] = mapped_column(Float)
    wind_direction: Mapped[float | None] = mapped_column(Float)
    wind_u: Mapped[float | None] = mapped_column(Float)
    wind_v: Mapped[float | None] = mapped_column(Float)


def _build_database_url() -> URL:
    return DatabaseSettings.from_environment().sqlalchemy_url()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create one pooled engine without logging its credential-bearing URL."""
    try:
        return create_engine(
            _build_database_url(),
            pool_pre_ping=True,
            pool_recycle=1_800,
            future=True,
            connect_args={
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )
    except (SQLAlchemyError, ValueError) as exc:
        LOGGER.error("Database engine configuration failed. Details: %s", exc)
        raise DatabaseConfigurationError(
            "Unable to configure the database connection"
        ) from exc


def create_schema(engine: Engine | None = None) -> None:
    """Create the ORM-managed table with a Timescale-compatible primary key."""
    database_engine = engine or get_engine()
    target = (
        f"{database_engine.url.host}:{database_engine.url.port}/"
        f"{database_engine.url.database}"
    )
    try:
        Base.metadata.create_all(bind=database_engine)
    except SQLAlchemyError as exc:
        LOGGER.error(
            "Database schema initialization failed for %s. Details: %s",
            target,
            exc,
        )
        raise DatabaseOperationError(
            "Unable to initialize database schema for "
            f"'{database_engine.url.database}' at "
            f"{database_engine.url.host}:{database_engine.url.port}. "
            "Verify that the server is reachable and DB_NAME exists."
        ) from exc


def initialize_timescaledb(engine: Engine | None = None) -> None:
    """Enable TimescaleDB and convert the fixed ORM table to a hypertable.

    The only DDL statement is a fixed application constant. Hypertable
    arguments are bound parameters; no user-controlled SQL identifiers are
    accepted.
    """
    database_engine = engine or get_engine()
    try:
        create_schema(database_engine)
        with database_engine.begin() as connection:
            # Fixed schema DDL only; no user-controlled identifiers are used.
            connection.execute(
                text(
                    "ALTER TABLE air_quality_measurements "
                    "ADD COLUMN IF NOT EXISTS rainfall DOUBLE PRECISION"
                )
            )
            connection.execute(
                text(
                    "ALTER TABLE air_quality_measurements "
                    "ADD COLUMN IF NOT EXISTS pressure DOUBLE PRECISION"
                )
            )
            connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS timescaledb")
            )
            connection.execute(
                text(
                    "SELECT create_hypertable("
                    ":table_name, :time_column, "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                ),
                {
                    "table_name": AirQualityMeasurement.__tablename__,
                    "time_column": "measured_at",
                },
            )
    except SQLAlchemyError as exc:
        LOGGER.error(
            "TimescaleDB hypertable initialization failed. Details: %s",
            exc,
        )
        raise DatabaseOperationError(
            "Unable to initialize the TimescaleDB hypertable"
        ) from exc


def ensure_measurement_schema(engine: Engine | None = None) -> None:
    """Create the measurement table; use a hypertable when Timescale exists."""
    try:
        initialize_timescaledb(engine)
    except DatabaseOperationError:
        LOGGER.warning(
            "TimescaleDB is unavailable; creating a regular Postgres table"
        )
        create_schema(engine)


def latest_global_measurement_time(
    *,
    engine: Engine | None = None,
) -> datetime | None:
    """Return the newest measurement timestamp across all stations."""
    statement = select(func.max(AirQualityMeasurement.measured_at))
    try:
        with Session(engine or get_engine()) as session:
            return session.scalar(statement)
    except SQLAlchemyError as exc:
        LOGGER.error(
            "Global latest measurement timestamp query failed. Details: %s",
            exc,
        )
        raise DatabaseOperationError(
            "Unable to load the latest measurement time"
        ) from exc


def _normalize_station(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("station_name must contain strings")
    station = " ".join(value.split())
    if not _STATION_PATTERN.fullmatch(station):
        raise ValueError("station_name contains invalid characters")
    return station


def _normalize_datetime(value: Any) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError("measured_at contains an invalid datetime") from None
    if pd.isna(timestamp):
        raise ValueError("measured_at cannot be null")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.to_pydatetime()


def _normalize_number(column: str, value: Any) -> float | None:
    if value is None or pd.isna(value):
        if column == "pm25":
            raise ValueError("pm25 cannot be null")
        return None
    if isinstance(value, bool):
        raise ValueError(f"{column} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{column} must be numeric") from None
    lower, upper = _NUMERIC_RANGES[column]
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{column} is outside the allowed range")
    return number


def _validated_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    unknown = set(frame.columns) - _ALLOWED_COLUMNS
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if unknown:
        raise ValueError(f"Unexpected columns: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        return []
    if len(frame) > 1_000_000:
        raise ValueError("DataFrame exceeds the maximum row count")

    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        record: dict[str, Any] = {
            "station_name": _normalize_station(raw["station_name"]),
            "measured_at": _normalize_datetime(raw["measured_at"]),
        }
        for column in _NUMERIC_RANGES:
            record[column] = _normalize_number(column, raw.get(column))
        records.append(record)
    return records


def _chunks(
    records: list[dict[str, Any]], size: int
) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(records), size):
        yield records[start : start + size]


def save_measurements(
    frame: pd.DataFrame,
    *,
    engine: Engine | None = None,
    chunk_size: int = 1_000,
    upsert: bool = True,
) -> int:
    """Validate and bulk insert/upsert a DataFrame in one transaction."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if not 1 <= chunk_size <= 10_000:
        raise ValueError("chunk_size must be between 1 and 10000")
    if not isinstance(upsert, bool):
        raise TypeError("upsert must be a boolean")

    records = _validated_records(frame)
    if not records:
        return 0

    try:
        with Session(engine or get_engine()) as session:
            with session.begin():
                for chunk in _chunks(records, chunk_size):
                    statement = postgresql_insert(
                        AirQualityMeasurement
                    ).values(chunk)
                    if upsert:
                        mutable_columns = {
                            column: getattr(statement.excluded, column)
                            for column in _NUMERIC_RANGES
                        }
                        statement = statement.on_conflict_do_update(
                            index_elements=[
                                AirQualityMeasurement.station_name,
                                AirQualityMeasurement.measured_at,
                            ],
                            set_=mutable_columns,
                        )
                    session.execute(statement)
    except SQLAlchemyError as exc:
        LOGGER.error(
            "Database insert failed for air-quality measurements. Details: %s",
            exc,
        )
        raise DatabaseOperationError("Unable to save measurements") from exc
    return len(records)


def load_measurements(
    start_at: datetime,
    end_at: datetime,
    *,
    station_name: str | None = None,
    limit: int = 100_000,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Read measurements with an ORM-built, parameterized SELECT statement."""
    start = _normalize_datetime(start_at)
    end = _normalize_datetime(end_at)
    if start >= end:
        raise ValueError("start_at must be earlier than end_at")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1_000_000:
        raise ValueError("limit must be an integer between 1 and 1000000")

    statement = (
        select(AirQualityMeasurement)
        .where(
            AirQualityMeasurement.measured_at >= start,
            AirQualityMeasurement.measured_at < end,
        )
        .order_by(AirQualityMeasurement.measured_at)
        .limit(limit)
    )
    if station_name is not None:
        statement = statement.where(
            AirQualityMeasurement.station_name
            == _normalize_station(station_name)
        )

    try:
        with Session(engine or get_engine()) as session:
            rows = session.scalars(statement).all()
    except SQLAlchemyError as exc:
        LOGGER.error(
            "Database query failed for air-quality measurements. Details: %s",
            exc,
        )
        raise DatabaseOperationError("Unable to load measurements") from exc

    columns = [
        "station_name",
        "measured_at",
        *_NUMERIC_RANGES.keys(),
    ]
    return pd.DataFrame(
        [
            {column: getattr(row, column) for column in columns}
            for row in rows
        ],
        columns=columns,
    )


def latest_measurement_time(
    station_name: str,
    *,
    end_at: datetime | None = None,
    engine: Engine | None = None,
) -> datetime | None:
    """Return the latest timestamp for a validated station."""
    statement = select(func.max(AirQualityMeasurement.measured_at)).where(
        AirQualityMeasurement.station_name == _normalize_station(station_name)
    )
    if end_at is not None:
        statement = statement.where(
            AirQualityMeasurement.measured_at < _normalize_datetime(end_at)
        )
    try:
        with Session(engine or get_engine()) as session:
            return session.scalar(statement)
    except SQLAlchemyError as exc:
        LOGGER.error(
            "Latest measurement timestamp query failed. Details: %s",
            exc,
        )
        raise DatabaseOperationError(
            "Unable to load the latest measurement time"
        ) from exc


def list_available_stations(
    *,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """Return station names and their parameterized observation coverage."""
    statement = (
        select(
            AirQualityMeasurement.station_name,
            func.min(AirQualityMeasurement.measured_at).label("start_at"),
            func.max(AirQualityMeasurement.measured_at).label("end_at"),
            func.count().label("row_count"),
        )
        .group_by(AirQualityMeasurement.station_name)
        .order_by(AirQualityMeasurement.station_name)
    )
    try:
        with Session(engine or get_engine()) as session:
            rows = session.execute(statement).all()
    except SQLAlchemyError as exc:
        LOGGER.error("Station list query failed. Details: %s", exc)
        raise DatabaseOperationError(
            "Unable to load available stations"
        ) from exc
    return [
        {
            "station_name": row.station_name,
            "start_at": row.start_at,
            "end_at": row.end_at,
            "row_count": int(row.row_count),
        }
        for row in rows
    ]
