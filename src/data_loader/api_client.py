
from __future__ import annotations

import asyncio
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import aiohttp
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

LOGGER = logging.getLogger(__name__)
AIRKOREA_URL = (
    "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/"
    "getMsrstnAcctoRltmMesureDnsty"
)
AIRKOREA_DAILY_URL = (
    "https://apis.data.go.kr/B552584/ArpltnStatsSvc/getMsrstnAcctoRDyrg"
)
_STATION_PATTERN = re.compile(r"^[0-9A-Za-z가-힣\s()_-]{1,50}$")
_KEY_PLACEHOLDER_PREFIXES = ("your_", "replace_", "insert_", "example_", "<")


class AirKoreaError(RuntimeError):
    """Public exception that never exposes API keys or response bodies."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_attempts <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if self.base_delay_seconds <= 0 or self.max_delay_seconds <= 0:
            raise ValueError("Retry delays must be positive")


class AirKoreaClient:
    """Fetch real-time station measurements from the public data portal."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        retry_policy: RetryPolicy | None = None,
        concurrency: int = 2,
        request_pause_seconds: float = 0.4,
    ) -> None:
        configured_key = api_key or os.getenv("AIRKOREA_SERVICE_KEY", "")
        if not isinstance(configured_key, str):
            raise TypeError("api_key must be a string")
        raw_key = configured_key.strip()
        if not raw_key or raw_key.lower().startswith(
            _KEY_PLACEHOLDER_PREFIXES
        ):
            raise ValueError("AIRKOREA_SERVICE_KEY가 설정되지 않았습니다.")
        # Decode an encoded portal key once; aiohttp params safely re-encodes it.
        self._api_key = unquote(raw_key)
        if not 1.0 <= timeout_seconds <= 120.0:
            raise ValueError("timeout_seconds must be between 1 and 120")
        if not 1 <= concurrency <= 20:
            raise ValueError("concurrency must be between 1 and 20")

        if not 0.0 <= request_pause_seconds <= 10.0:
            raise ValueError("request_pause_seconds must be between 0 and 10")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._retry = retry_policy or RetryPolicy(
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=20.0,
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._request_pause_seconds = request_pause_seconds
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AirKoreaClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @staticmethod
    def _validate_station_name(station_name: str) -> str:
        if not isinstance(station_name, str):
            raise TypeError("station_name must be a string")
        normalized = " ".join(station_name.split())
        if not _STATION_PATTERN.fullmatch(normalized):
            raise ValueError("station_name contains invalid characters")
        return normalized

    @staticmethod
    def _validate_pagination(page_no: int, rows: int) -> None:
        if isinstance(page_no, bool) or not isinstance(page_no, int):
            raise TypeError("page_no must be an integer")
        if isinstance(rows, bool) or not isinstance(rows, int):
            raise TypeError("rows must be an integer")
        if not 1 <= page_no <= 10_000 or not 1 <= rows <= 1_000:
            raise ValueError("Pagination values are outside the allowed range")

    async def fetch_station(
        self,
        station_name: str,
        *,
        page_no: int = 1,
        rows: int = 100,
        data_term: str = "DAILY",
    ) -> list[dict[str, Any]]:
        """Return measurements for one validated station."""
        station = self._validate_station_name(station_name)
        self._validate_pagination(page_no, rows)
        normalized_term = data_term.upper()
        if normalized_term not in {"DAILY", "MONTH", "3MONTH"}:
            raise ValueError("data_term must be DAILY, MONTH, or 3MONTH")
        params = {
            "serviceKey": self._api_key,
            "returnType": "json",
            "numOfRows": rows,
            "pageNo": page_no,
            "stationName": station,
            "dataTerm": normalized_term,
            "ver": "1.3",
        }

        items: list[dict[str, Any]] = []
        current_page = page_no
        total_count: int | None = None
        while total_count is None or len(items) < total_count:
            page_params = {**params, "pageNo": current_page}
            async with self._semaphore:
                payload = await self._request_json(AIRKOREA_URL, page_params)
            page_items, total_count = self._extract_items(payload)
            items.extend(page_items)
            if not page_items or len(page_items) < rows:
                break
            current_page += 1
            if current_page - page_no > 200:
                raise AirKoreaError("AirKorea pagination exceeded the safety limit")
            await asyncio.sleep(self._request_pause_seconds)
        return items

    async def fetch_daily_station(
        self,
        station_name: str,
        *,
        start_date: str,
        end_date: str,
        rows: int = 1_000,
    ) -> list[dict[str, Any]]:
        """Fetch daily confirmed/stat averages for one station and date range."""
        station = self._validate_station_name(station_name)
        self._validate_pagination(1, rows)
        if len(start_date) != 8 or len(end_date) != 8:
            raise ValueError("Daily inquiry dates must use YYYYMMDD")
        params = {
            "serviceKey": self._api_key,
            "returnType": "json",
            "numOfRows": rows,
            "inqBginDt": start_date,
            "inqEndDt": end_date,
            "msrstnName": station,
        }
        items: list[dict[str, Any]] = []
        current_page = 1
        total_count: int | None = None
        while total_count is None or len(items) < total_count:
            async with self._semaphore:
                payload = await self._request_json(
                    AIRKOREA_DAILY_URL,
                    {**params, "pageNo": current_page},
                )
            page_items, total_count = self._extract_items(payload)
            items.extend(page_items)
            if not page_items or len(page_items) < rows:
                break
            current_page += 1
            if current_page > 200:
                raise AirKoreaError("AirKorea daily pagination exceeded the limit")
            await asyncio.sleep(self._request_pause_seconds)
        return items

    @staticmethod
    def _extract_items(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        try:
            response = payload["response"]
            header = response["header"]
            result_code = str(header["resultCode"])
            if result_code not in {"00", "0"}:
                result_message = str(header.get("resultMsg") or "unknown")
                raise AirKoreaError(
                    f"AirKorea rejected the request ({result_code}: {result_message})"
                )
            body = response.get("body") or {}
            raw_items = body.get("items", [])
            if isinstance(raw_items, dict):
                raw_items = raw_items.get("item", [])
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raise TypeError
            total_count = int(body.get("totalCount") or len(raw_items) or 0)
            return (
                [item for item in raw_items if isinstance(item, dict)],
                total_count,
            )
        except AirKoreaError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise AirKoreaError("AirKorea returned an unexpected response") from exc

    async def fetch_many(
        self,
        station_names: Iterable[str],
        *,
        rows: int = 100,
        data_term: str = "DAILY",
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch multiple stations concurrently within the configured limit."""
        stations = [self._validate_station_name(name) for name in station_names]
        if not stations or len(stations) > 100:
            raise ValueError("Provide between 1 and 100 station names")
        results: list[list[dict[str, Any]]] = []
        for station in stations:
            results.append(
                await self.fetch_station(
                    station,
                    rows=rows,
                    data_term=data_term,
                )
            )
            await asyncio.sleep(self._request_pause_seconds)
        return dict(zip(stations, results, strict=True))

    async def _request_json(
        self,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if self._session is None:
            raise AirKoreaError(
                "Use AirKoreaClient as an asynchronous context manager"
            )

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                async with self._session.get(
                    url,
                    params=params,
                    allow_redirects=False,
                ) as response:
                    if response.status in {429, 500, 502, 503, 504}:
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                        )
                    if response.status == 403:
                        raise AirKoreaError(
                            "AirKorea denied this service. "
                            "The statistics API may require a separate portal approval."
                        )
                    if response.status != 200:
                        raise AirKoreaError(
                            f"AirKorea request was rejected (HTTP {response.status})"
                        )
                    payload = await response.json(content_type=None)
                    if not isinstance(payload, dict):
                        raise AirKoreaError("AirKorea returned invalid data")
                    return payload
            except AirKoreaError:
                raise
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ValueError,
            ):
                if attempt >= self._retry.max_attempts:
                    LOGGER.warning(
                        "AirKorea request failed after %d attempts",
                        attempt,
                    )
                    raise AirKoreaError(
                        "Unable to retrieve AirKorea data"
                    ) from None

                upper_bound = min(
                    self._retry.max_delay_seconds,
                    self._retry.base_delay_seconds * (2 ** (attempt - 1)),
                )
                await asyncio.sleep(random.uniform(0, upper_bound))

        raise AirKoreaError("Unable to retrieve AirKorea data")
