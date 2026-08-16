from __future__ import annotations

import logging
import os
import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from src.portal_key import (
    DEFAULT_KMA_ENDPOINT,
    KMA_HOURLY_OPERATION,
    build_portal_request_url,
    join_portal_operation,
    resolve_portal_endpoint,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

LOGGER = logging.getLogger(__name__)
KST: Final = timezone(timedelta(hours=9))
ASOS_COLUMNS: Final = (
    "tm",
    "stnId",
    "ta",
    "taQcflg",
    "rn",
    "rnQcflg",
    "ws",
    "wsQcflg",
    "wd",
    "wdQcflg",
    "hm",
    "hmQcflg",
    "pa",
    "paQcflg",
)
ASOS_NUMERIC_COLUMNS: Final = tuple(
    column for column in ASOS_COLUMNS if column != "tm"
)
_KEY_PLACEHOLDER_PREFIXES: Final = (
    "your_",
    "replace_",
    "insert_",
    "example_",
    "<",
)


class KMAAPIError(RuntimeError):
    """Credential-safe error returned for KMA API failures."""


def asos_latest_complete_date(now: datetime | None = None) -> date:
    """Return the latest ASOS date that the OpenAPI actually serves.

    The portal publishes through D-1, and D-1 itself is only queryable after
    11:00 KST.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(KST)
    yesterday = current.date() - timedelta(days=1)
    if current.hour < 11:
        return yesterday - timedelta(days=1)
    return yesterday


class KMAWeatherClient:
    """Synchronous, retrying ASOS hourly observation client."""

    def __init__(
        self,
        service_key: Optional[str] = None,
        *,
        timeout_seconds: float = 15.0,
        max_attempts: int = 4,
        session: Optional[requests.Session] = None,
    ) -> None:
        raw_key = service_key if service_key is not None else os.getenv(
            "KMA_SERVICE_KEY", ""
        )
        if not isinstance(raw_key, str):
            raise TypeError("service_key must be a string")
        if not raw_key.strip() or raw_key.strip().lower().startswith(
            _KEY_PLACEHOLDER_PREFIXES
        ):
            raise ValueError("KMA_SERVICE_KEY가 설정되지 않았습니다.")
        self._service_key = raw_key
        kma_base = resolve_portal_endpoint(
            "KMA_ENDPOINT",
            DEFAULT_KMA_ENDPOINT,
        )
        self._asos_url = join_portal_operation(kma_base, KMA_HOURLY_OPERATION)
        if not 1.0 <= timeout_seconds <= 120.0:
            raise ValueError("timeout_seconds must be between 1 and 120")
        if not 1 <= max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._session = session or requests.Session()
        self._owns_session = session is None

    def __enter__(self) -> "KMAWeatherClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    @staticmethod
    def _validate_date(value: str, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError:
            raise ValueError(f"{name} must use YYYYMMDD format") from None
        return value

    @staticmethod
    def _validate_hour(value: str, name: str) -> str:
        if not isinstance(value, str) or len(value) != 2 or not value.isdigit():
            raise ValueError(f"{name} must use HH format")
        if not 0 <= int(value) <= 23:
            raise ValueError(f"{name} must be between 00 and 23")
        return value

    @staticmethod
    def _validate_positive_integer(
        value: int, name: str, maximum: int
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
        return value

    def fetch_hourly(
        self,
        *,
        startDt: str,
        startHh: str,
        endDt: str,
        endHh: str,
        stnIds: int | str,
        numOfRows: int = 999,
        pageNo: int = 1,
        dataType: str = "JSON",
        dataCd: str = "ASOS",
        dateCd: str = "HR",
    ) -> pd.DataFrame:
        """Fetch all pages for one ASOS station and return a clean DataFrame."""
        start_date = self._validate_date(startDt, "startDt")
        end_date = self._validate_date(endDt, "endDt")
        start_hour = self._validate_hour(startHh, "startHh")
        end_hour = self._validate_hour(endHh, "endHh")
        rows = self._validate_positive_integer(numOfRows, "numOfRows", 999)
        first_page = self._validate_positive_integer(pageNo, "pageNo", 100_000)
        if (start_date, start_hour) > (end_date, end_hour):
            raise ValueError("KMA start time must not be after end time")
        try:
            station_id = int(stnIds)
        except (TypeError, ValueError):
            raise ValueError("stnIds must be an integer station ID") from None
        if not 1 <= station_id <= 999:
            raise ValueError("stnIds must be between 1 and 999")
        if dataType.upper() != "JSON" or dataCd != "ASOS" or dateCd != "HR":
            raise ValueError("Only JSON ASOS hourly observations are supported")

        common_params: dict[str, Any] = {
            "numOfRows": rows,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": start_date,
            "startHh": start_hour,
            "endDt": end_date,
            "endHh": end_hour,
            "stnIds": station_id,
        }
        items: list[dict[str, Any]] = []
        current_page = first_page
        total_count: int | None = None

        while total_count is None or len(items) < total_count:
            payload = self._request_json(
                {**common_params, "pageNo": current_page}
            )
            page_items, total_count = self._extract_items(payload)
            items.extend(page_items)
            if not page_items or len(page_items) < rows:
                break
            current_page += 1
            if current_page - first_page > 10_000:
                raise KMAAPIError("KMA pagination exceeded the safety limit")

        return self._to_dataframe(items)

    def fetch_hourly_range(
        self,
        *,
        start: date,
        end: date,
        stnIds: int | str,
        chunk_days: int = 180,
    ) -> pd.DataFrame:
        """Fetch a long ASOS window in bounded chunks with request pauses."""
        if not isinstance(start, date) or not isinstance(end, date):
            raise TypeError("start and end must be date values")
        if start > end:
            raise ValueError("KMA start date must not be after end date")
        if not 30 <= chunk_days <= 366:
            raise ValueError("chunk_days must be between 30 and 366")
        complete = asos_latest_complete_date()
        if end > complete:
            LOGGER.warning(
                "Clamping KMA end date from %s to %s (ASOS serves D-1 after 11:00 KST)",
                end.isoformat(),
                complete.isoformat(),
            )
            end = complete
        if start > end:
            return self._to_dataframe([])

        frames: list[pd.DataFrame] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
            LOGGER.info(
                "Fetching KMA ASOS %s from %s to %s",
                stnIds,
                cursor.isoformat(),
                chunk_end.isoformat(),
            )
            frames.append(
                self.fetch_hourly(
                    startDt=cursor.strftime("%Y%m%d"),
                    startHh="00",
                    endDt=chunk_end.strftime("%Y%m%d"),
                    endHh="23",
                    stnIds=stnIds,
                )
            )
            cursor = chunk_end + timedelta(days=1)
            time.sleep(random.uniform(0.3, 0.8))
        if not frames:
            return self._to_dataframe([])
        combined = pd.concat(frames, ignore_index=True)
        return (
            combined.drop_duplicates(["stnId", "tm"])
            .sort_values(["stnId", "tm"])
            .reset_index(drop=True)
        )

    def _request_json(self, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._session.get(
                    build_portal_request_url(
                        self._asos_url,
                        self._service_key,
                        params,
                    ),
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt >= self._max_attempts:
                        raise KMAAPIError(
                            "KMA API is temporarily unavailable"
                        )
                    retry_after = response.headers.get("Retry-After")
                    delay = self._retry_delay(attempt, retry_after)
                    time.sleep(delay)
                    continue
                if response.status_code != 200:
                    raise KMAAPIError(
                        "KMA API rejected the request "
                        f"(HTTP {response.status_code})"
                    )
                try:
                    payload = response.json()
                except requests.exceptions.JSONDecodeError:
                    raise KMAAPIError("KMA API returned invalid JSON") from None
                if not isinstance(payload, dict):
                    raise KMAAPIError("KMA API returned an unexpected response")
                return payload
            except KMAAPIError:
                raise
            except (requests.Timeout, requests.ConnectionError):
                if attempt >= self._max_attempts:
                    LOGGER.warning(
                        "KMA request failed after %d attempts", attempt
                    )
                    raise KMAAPIError("Unable to retrieve KMA data") from None
                time.sleep(self._retry_delay(attempt, None))
            except requests.RequestException:
                raise KMAAPIError("KMA request failed") from None

        raise KMAAPIError("Unable to retrieve KMA data")

    @staticmethod
    def _retry_delay(attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 30.0)
            except ValueError:
                pass
        return random.uniform(0.0, min(2 ** (attempt - 1), 8.0))

    @staticmethod
    def _extract_items(
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        try:
            response = payload["response"]
            header = response["header"]
            result_code = str(header["resultCode"])
            if result_code not in {"00", "0"}:
                raise KMAAPIError("KMA API rejected the request parameters")
            body = response["body"]
            total_count = int(body.get("totalCount", 0))
            item_container = body.get("items") or {}
            raw_items = (
                item_container.get("item", [])
                if isinstance(item_container, dict)
                else item_container
            )
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raise TypeError
            return (
                [item for item in raw_items if isinstance(item, dict)],
                total_count,
            )
        except KMAAPIError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError):
            raise KMAAPIError("KMA API returned an unexpected schema") from None

    @staticmethod
    def _to_dataframe(items: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(items)
        for column in ASOS_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[list(ASOS_COLUMNS)].copy()
        if frame.empty:
            frame["tm"] = pd.to_datetime(frame["tm"])
            return frame

        frame["tm"] = pd.to_datetime(
            frame["tm"],
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        if frame["tm"].isna().any():
            raise KMAAPIError("KMA response contains invalid observation times")
        for column in ASOS_NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["stnId"] = frame["stnId"].astype("Int64")
        return frame.sort_values(["stnId", "tm"]).reset_index(drop=True)
