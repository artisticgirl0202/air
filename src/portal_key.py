from __future__ import annotations

import os
from typing import Any, Mapping
from urllib.parse import urlencode, urlsplit


_ALLOWED_PORTAL_HOSTS = frozenset({"apis.data.go.kr"})
DEFAULT_AIRKOREA_ENDPOINT = (
    "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc"
)
DEFAULT_AIRKOREA_DAILY_ENDPOINT = (
    "http://apis.data.go.kr/B552584/ArpltnStatsSvc"
)
DEFAULT_KMA_ENDPOINT = (
    "http://apis.data.go.kr/1360000/AsosHourlyInfoService"
)
AIRKOREA_HOURLY_OPERATION = "/getMsrstnAcctoRltmMesureDnsty"
AIRKOREA_DAILY_OPERATION = "/getMsrstnAcctoRDyrg"
KMA_HOURLY_OPERATION = "/getWthrDataList"


def resolve_portal_endpoint(env_name: str, default: str) -> str:
    """Load a portal base URL from the environment, falling back to the spec default.

    Only ``http``/``https`` URLs on ``apis.data.go.kr`` are accepted. Empty env
    values use ``default``, which follows the portal spec (often ``http``).
    """
    if not env_name.isidentifier() or not env_name.isupper():
        raise ValueError("Invalid endpoint environment variable name")
    raw = os.getenv(env_name, "").strip().strip('"').strip("'")
    candidate = raw or default
    candidate = candidate.strip().rstrip("/")
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in _ALLOWED_PORTAL_HOSTS:
        raise ValueError(f"{env_name} must be an apis.data.go.kr http(s) URL")
    if any(character in candidate for character in "\r\n"):
        raise ValueError(f"{env_name} contains invalid characters")
    return candidate


def join_portal_operation(base_url: str, operation: str) -> str:
    """Append an operation path once, even if the env value already includes it."""
    if not operation.startswith("/"):
        operation = f"/{operation}"
    if base_url.endswith(operation):
        return base_url
    return f"{base_url}{operation}"


def build_portal_request_url(
    base_url: str,
    service_key: str,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Insert the env service key as-is so ``%2B`` is not turned into a space.

    Remaining parameters are encoded once with ``urlencode``. The key itself is
    never passed through ``unquote`` or ``params``.
    """
    rest = {
        key: value
        for key, value in (params or {}).items()
        if key not in {"serviceKey", "ServiceKey"}
    }
    encoded_rest = urlencode(rest, doseq=True)
    if encoded_rest:
        return f"{base_url}?serviceKey={service_key}&{encoded_rest}"
    return f"{base_url}?serviceKey={service_key}"
