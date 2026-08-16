from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Final

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
)

from src.data_loader.api_client import AirKoreaError
from src.database import (
    DatabaseConfigurationError,
    DatabaseOperationError,
    latest_global_measurement_time,
)
from src.generate_dataset import (
    DEFAULT_SEOUL_STATIONS,
    generate_dataset,
    parse_station_names,
)
from src.preprocessing import PreprocessingError
from src.weather_client import KMAAPIError, asos_latest_complete_date


LOGGER = logging.getLogger(__name__)
STALE_AFTER: Final = timedelta(hours=36)
BACKFILL_DAYS: Final = 90
HOURLY_LOOKBACK_DAYS: Final = 2
_URL_PATTERN: Final = re.compile(r"https?://\S+", re.IGNORECASE)
_ingest_lock = threading.Lock()
_scheduler: BackgroundScheduler | None = None
_INGEST_ERRORS: Final = (
    AirKoreaError,
    KMAAPIError,
    PreprocessingError,
    DatabaseConfigurationError,
    DatabaseOperationError,
    ValueError,
    OSError,
)
_KEY_ERROR_MARKERS: Final = (
    "service_key",
    "servicekey",
    "등록되지",
    "not_registered",
    "denied the",
)
_INGEST_RETRY_WAITS: Final = (wait_fixed(300), wait_fixed(600), wait_fixed(1_200))


def ingest_enabled() -> bool:
    flag = os.getenv("ENABLE_DATA_INGEST", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    return os.getenv("RENDER", "").strip().lower() == "true"


def _collection_end() -> date:
    """ASOS OpenAPI publishes through D-1, and D-1 only after 11:00 KST."""
    return asos_latest_complete_date()


def _safe_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return _URL_PATTERN.sub("[url]", text)[:200]


def _log_ingest_failure(label: str, exc: BaseException) -> None:
    LOGGER.error("%s failed (%s: %s)", label, type(exc).__name__, _safe_error(exc))


def _is_transient_ingest_error(exc: BaseException) -> bool:
    """Retry network/service blips, not invalid portal keys or bad inputs."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(exc, ValueError):
        return False
    if not isinstance(exc, (AirKoreaError, KMAAPIError)):
        return False
    message = str(exc).lower()
    if any(marker in message for marker in _KEY_ERROR_MARKERS):
        return False
    return True


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_chain(*_INGEST_RETRY_WAITS),
    retry=retry_if_exception(_is_transient_ingest_error),
    before_sleep=lambda retry_state: LOGGER.warning(
        "Transient ingest error (%s); next attempt in %s s",
        type(retry_state.outcome.exception()).__name__
        if retry_state.outcome is not None and retry_state.outcome.failed
        else "unknown",
        getattr(retry_state.next_action, "sleep", None),
    ),
)
def _upsert_window(*, start: date, end: date, label: str) -> None:
    if start > end:
        raise ValueError("Ingest start date must not be after end date")
    LOGGER.info(
        "Starting %s ingest for %s to %s",
        label,
        start.isoformat(),
        end.isoformat(),
    )
    generate_dataset(
        parse_station_names(list(DEFAULT_SEOUL_STATIONS)),
        start=start,
        end=end,
        persist_db=True,
        write_csv=False,
        include_daily=False,
    )


def _latest_timestamp_with_retry() -> datetime | None:
    last_error: DatabaseOperationError | None = None
    for attempt in range(1, 4):
        try:
            return latest_global_measurement_time()
        except DatabaseOperationError as exc:
            last_error = exc
            LOGGER.warning(
                "Database not ready for ingest probe (attempt %d/3)",
                attempt,
            )
            time.sleep(1.5 * attempt)
    if last_error is not None:
        raise last_error
    return None


def run_backfill_if_stale() -> None:
    """Fetch the latest ~3 months when the database is empty or stale."""
    if not _ingest_lock.acquire(blocking=False):
        LOGGER.info("Skipping backfill; another ingest job is running")
        return
    try:
        latest = _latest_timestamp_with_retry()
        now = datetime.now(timezone.utc)
        if latest is not None:
            latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
            age = now - latest_utc.astimezone(timezone.utc)
            if age <= STALE_AFTER:
                LOGGER.info("Measurement table is fresh; skipping 3-month backfill")
                return
            LOGGER.info("Latest measurement is stale; running 3-month backfill")
        else:
            LOGGER.info("No measurements found; running 3-month backfill")
        end = _collection_end()
        start = end - timedelta(days=BACKFILL_DAYS - 1)
        _upsert_window(start=start, end=end, label="3MONTH")
    except _INGEST_ERRORS as exc:
        _log_ingest_failure("3-month backfill", exc)
    except Exception as exc:
        _log_ingest_failure("3-month backfill", exc)
    finally:
        _ingest_lock.release()


def run_hourly_ingest() -> None:
    """Fetch the latest daily AirKorea window and upsert into Postgres."""
    if not _ingest_lock.acquire(blocking=False):
        LOGGER.info("Skipping hourly ingest; another ingest job is running")
        return
    try:
        end = _collection_end()
        start = end - timedelta(days=HOURLY_LOOKBACK_DAYS - 1)
        _upsert_window(start=start, end=end, label="DAILY")
    except _INGEST_ERRORS as exc:
        _log_ingest_failure("Hourly ingest", exc)
    except Exception as exc:
        _log_ingest_failure("Hourly ingest", exc)
    finally:
        _ingest_lock.release()


def start_ingest_scheduler() -> None:
    """Start the hourly ingest job. Safe to call once from FastAPI lifespan."""
    global _scheduler
    if not ingest_enabled():
        LOGGER.info("Data ingest scheduler is disabled")
        return
    if _scheduler is not None and _scheduler.running:
        return
    worker = threading.Thread(
        target=run_backfill_if_stale,
        name="airkorea-backfill",
        daemon=True,
    )
    worker.start()
    _scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    _scheduler.add_job(
        run_hourly_ingest,
        trigger=IntervalTrigger(hours=1),
        id="airkorea-hourly-ingest",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3_600,
        replace_existing=True,
    )
    _scheduler.start()
    LOGGER.info("Hourly AirKorea ingest scheduler started")


def stop_ingest_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.shutdown(wait=False)
    _scheduler = None
