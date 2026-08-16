

from __future__ import annotations

import html
import logging
import math
import os
import re
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import bcrypt
import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from dotenv import load_dotenv
from sqlalchemy import text
from streamlit_authenticator.utilities import LoginError
from streamlit_folium import st_folium
from yaml.loader import SafeLoader


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

STATION_PATTERN: Final = re.compile(r"^[가-힣A-Za-z0-9\s-]{1,50}$")
SEOUL_MAP_CENTER: Final = (37.5326, 126.9900)
SEOUL_STATION_COORDINATES: Final = {
    "종로구": (37.5720, 127.0050),
    "관악구": (37.4782, 126.9515),
    "서초구": (37.4837, 127.0324),
    "강남구": (37.5172, 127.0473),
    "마포구": (37.5615, 126.9086),
}
MOE_PM25_GRADES: Final = (
    (15.0, "좋음", "#2563EB"),
    (35.0, "보통", "#16A34A"),
    (75.0, "나쁨", "#EA580C"),
    (float("inf"), "매우나쁨", "#DC2626"),
)
LICENSE_TEXT: Final = """
**[데이터 출처 및 라이선스]**
- 기상청_지상(종관, ASOS) 시간자료 조회서비스: 공공저작물 출처표시 (제 1유형)
- 한국환경공단_에어코리아_대기오염정보: 공공저작물 출처표시, 변경금지 (제 3유형)
"""
AUTH_CONFIG_PATH: Final = PROJECT_ROOT / "dashboard" / "config.yaml"
# 아이디·측정소·검색어: 허용 문자만 통과시켜 SQLi 페이로드를 1차로 차단합니다.
SAFE_IDENTIFIER_PATTERN: Final = re.compile(r"^[가-힣A-Za-z0-9_-]{1,50}$")
SAFE_SEARCH_PATTERN: Final = re.compile(r"^[가-힣A-Za-z0-9\s._-]{0,80}$")
ALLOWED_ROLES: Final = frozenset({"admin", "user"})
ALLOWED_UPLOAD_EXTENSIONS: Final = frozenset({".csv", ".xlsx"})
MAX_UPLOAD_BYTES: Final = 10 * 1024 * 1024
UPLOAD_ROOT: Final = (PROJECT_ROOT / "data" / "uploads").resolve()


class NoDataError(RuntimeError):
    """Raised when the API has no rows for a valid selection."""


class SecurityError(ValueError):
    """Raised when a user input fails an OWASP-oriented server-side check."""


def display_text(value: object) -> str:
    """Escape API or user text before it is rendered in Streamlit."""
    return html.escape(str(value), quote=True)


def validate_station_name(value: str) -> str:
    """Return a whitelist-validated station name for API requests."""
    if not isinstance(value, str):
        raise TypeError("측정소명은 문자열이어야 합니다.")
    normalized = " ".join(value.split())
    if not STATION_PATTERN.fullmatch(normalized):
        raise ValueError("측정소명에 허용되지 않은 문자가 포함되어 있습니다.")
    return normalized


def validate_date_range(value: object) -> tuple[date, date]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("시작일과 종료일을 모두 선택해 주세요.")
    start, end = value
    if not isinstance(start, date) or not isinstance(end, date):
        raise TypeError("날짜 형식이 올바르지 않습니다.")
    if start > end:
        raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
    if (end - start).days > 366:
        raise ValueError("조회 기간은 최대 366일입니다.")
    return start, end


# ---------------------------------------------------------------------------
# OWASP 대응 보안 검증 (위젯 제약만 믿지 않고 서버에서 다시 검사합니다)
# Streamlit의 options/type/max_chars 는 브라우저 UX용이며 보안 경계가 아닙니다.
# ---------------------------------------------------------------------------


def validate_safe_identifier(value: object, *, field_name: str = "입력값") -> str:
    """SQL 인젝션 1차 방어: 허용된 문자 패턴만 통과시킵니다.

    아이디, 측정소명, 검색 키처럼 DB나 파일 경로에 들어갈 값에 사용합니다.
    따옴표, 세미콜론, 주석(--) 같은 SQL 메타문자는 여기서 거부됩니다.
    """
    if not isinstance(value, str):
        raise SecurityError(f"{field_name}은(는) 문자열이어야 합니다.")
    normalized = value.strip()
    if not SAFE_IDENTIFIER_PATTERN.fullmatch(normalized):
        raise SecurityError(f"{field_name}에 허용되지 않은 문자가 있습니다.")
    return normalized


def validate_search_query(value: object) -> str:
    """검색창 입력용 화이트리스트. 빈 문자열은 허용합니다."""
    if not isinstance(value, str):
        raise SecurityError("검색어는 문자열이어야 합니다.")
    query = value.strip()
    if not SAFE_SEARCH_PATTERN.fullmatch(query):
        raise SecurityError("검색어에 허용되지 않은 문자가 있습니다.")
    return query


def parameterized_station_lookup(station_name: str) -> tuple[Any, dict[str, str]]:
    """SQL 인젝션 2차 방어: f-string으로 SQL을 만들지 않습니다.

    금지 예: text(f\"SELECT * FROM t WHERE station_name = '{station_name}'\")
    허용 예: :station_name 자리표시자와 바인딩 딕셔너리.
    실제 조회는 src/database.py 의 SQLAlchemy ORM select() 를 사용합니다.
    """
    safe_station = validate_safe_identifier(station_name, field_name="측정소명")
    statement = text(
        "SELECT station_name, measured_at, pm25 "
        "FROM air_quality_measurements "
        "WHERE station_name = :station_name"
    )
    return statement, {"station_name": safe_station}


def escape_html(value: object) -> str:
    """XSS 방어: HTML 특수문자를 엔티티로 바꿉니다. <script> 는 텍스트가 됩니다."""
    return html.escape(str(value), quote=True)


def safe_markdown(value: object, *, unsafe_allow_html: bool = False) -> None:
    """화면에 쓰기 전에 항상 escape 합니다.

    unsafe_allow_html=True 가 필요해도 사용자 입력은 먼저 escape_html() 합니다.
    신뢰할 수 없는 값에 raw HTML 을 넣지 마세요.
    """
    escaped = escape_html(value)
    if unsafe_allow_html:
        st.markdown(f"<p>{escaped}</p>", unsafe_allow_html=True)
        return
    st.markdown(escaped)


def is_authenticated() -> bool:
    """로그인 성공 여부. streamlit-authenticator 가 session_state 에 기록합니다."""
    return st.session_state.get("authentication_status") is True


def current_username() -> str:
    raw = st.session_state.get("username")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    return raw.strip()


def current_roles(auth_config: dict[str, Any]) -> list[str]:
    """역할을 위젯 값이 아니라 config.yaml 에서 매 요청마다 다시 읽습니다.

    st.selectbox 로 고른 역할은 클라이언트가 조작할 수 있으므로 인가에 쓰지 않습니다.
    """
    username = current_username()
    users = (auth_config.get("credentials") or {}).get("usernames") or {}
    profile = users.get(username) or users.get(username.lower()) or {}
    raw_roles = profile.get("roles") or profile.get("role") or []
    if isinstance(raw_roles, str):
        raw_roles = [raw_roles]
    return [
        role
        for role in raw_roles
        if isinstance(role, str) and role in ALLOWED_ROLES
    ]


def require_authentication() -> None:
    """Broken Access Control 방어: 로그인하지 않으면 이후 화면을 그리지 않습니다."""
    if not is_authenticated():
        st.error("로그인이 필요합니다. 권한이 있는 계정으로 다시 시도해 주세요.")
        st.stop()


def require_role(auth_config: dict[str, Any], *allowed: str) -> None:
    """탭을 숨기는 것만으로는 부족합니다. 렌더 직전에 역할을 다시 확인합니다."""
    require_authentication()
    permitted = {role for role in allowed if role in ALLOWED_ROLES}
    if not permitted.intersection(current_roles(auth_config)):
        st.error("이 기능을 사용할 권한이 없습니다.")
        st.stop()


def sanitize_upload_filename(filename: object) -> str:
    """경로 이탈 방어: 디렉터리 성분을 버리고 안전한 파일명만 남깁니다.

    '../../etc/passwd' 나 'C:\\\\Windows\\\\win.ini' 는 basename 만 취합니다.
    """
    if not isinstance(filename, str) or not filename.strip():
        raise SecurityError("파일명이 올바르지 않습니다.")
    name = Path(filename.replace("\\", "/")).name
    if name in {"", ".", ".."} or ".." in name:
        raise SecurityError("경로 이탈이 의심되는 파일명입니다.")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise SecurityError("허용된 확장자는 .csv, .xlsx 입니다.")
    stem = re.sub(r"[^가-힣A-Za-z0-9_-]", "_", Path(name).stem)[:50].strip("._")
    if not stem:
        raise SecurityError("파일명에 사용 가능한 문자가 없습니다.")
    return f"{stem}{suffix}"


def validate_uploaded_file(uploaded: Any) -> tuple[str, bytes]:
    """file_uploader 의 type= 필터는 브라우저 UX입니다. 확장자·크기·시그니처를 재검증합니다."""
    if uploaded is None:
        raise SecurityError("파일이 없습니다.")
    safe_name = sanitize_upload_filename(getattr(uploaded, "name", ""))
    payload = uploaded.getvalue()
    if not payload:
        raise SecurityError("빈 파일은 업로드할 수 없습니다.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise SecurityError("파일 크기는 10MB 이하여야 합니다.")
    suffix = Path(safe_name).suffix.lower()
    if suffix == ".xlsx" and not payload.startswith(b"PK"):
        raise SecurityError("xlsx 형식이 아닙니다.")
    if suffix == ".csv" and (payload.lstrip().startswith(b"<") or b"\x00" in payload[:1024]):
        raise SecurityError("CSV 형식이 아닙니다.")
    return safe_name, payload


def save_uploaded_file(uploaded: Any, *, username: str) -> Path:
    """저장 경로가 data/uploads 밖으로 나가지 못하게 pathlib 로 잠급니다."""
    safe_name, payload = validate_uploaded_file(uploaded)
    safe_user = validate_safe_identifier(username, field_name="사용자 ID")
    destination = (UPLOAD_ROOT / safe_user / safe_name).resolve()
    try:
        destination.relative_to(UPLOAD_ROOT)
    except ValueError as exc:
        raise SecurityError("업로드 경로가 허용 영역을 벗어났습니다.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def _plain_mapping(value: Any) -> Any:
    """st.secrets AttrDict 를 일반 dict 로 풀어 Authenticate 에 넘깁니다."""
    if hasattr(value, "to_dict"):
        return {key: _plain_mapping(item) for key, item in value.to_dict().items()}
    if isinstance(value, dict):
        return {key: _plain_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_mapping(item) for item in value]
    return value


def _has_streamlit_secrets() -> bool:
    """secrets.toml 이 있을 때만 st.secrets 를 엽니다.

    파일이 없는데 st.secrets 를 읽으면 Streamlit 이 빨간 오류를 띄웁니다.
    load_if_toml_exists() 는 파일이 없으면 False 만 반환합니다.
    """
    loader = getattr(st.secrets, "load_if_toml_exists", None)
    if not callable(loader):
        return False
    try:
        return bool(loader())
    except Exception:
        return False


def _auth_config_from_secrets() -> dict[str, Any] | None:
    """Streamlit Community Cloud 의 Secrets 에 auth 가 있으면 그것을 씁니다."""
    if not _has_streamlit_secrets() or "auth" not in st.secrets:
        return None
    config = _plain_mapping(st.secrets["auth"])
    return config if isinstance(config, dict) else None


def backend_api_url() -> str:
    if _has_streamlit_secrets() and "BACKEND_API_URL" in st.secrets:
        return str(st.secrets["BACKEND_API_URL"]).rstrip("/")
    return os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000").rstrip("/")


def load_auth_config() -> dict[str, Any]:
    """로컬은 config.yaml, 클라우드는 st.secrets['auth'] 를 읽습니다."""
    config = _auth_config_from_secrets()
    if config is None:
        if not AUTH_CONFIG_PATH.is_file():
            raise FileNotFoundError(
                "dashboard/config.yaml 이 없거나 Streamlit Secrets 의 auth 가 없습니다. "
                "config.yaml.example 을 복사하거나 Cloud Secrets 에 auth 를 넣으세요."
            )
        with AUTH_CONFIG_PATH.open(encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=SafeLoader)
    if not isinstance(config, dict):
        raise ValueError("인증 설정 형식이 올바르지 않습니다.")
    cookie = config.setdefault("cookie", {})
    env_key = os.getenv("AUTH_COOKIE_KEY", "").strip()
    if not env_key and _has_streamlit_secrets():
        env_key = str(st.secrets.get("AUTH_COOKIE_KEY", "")).strip()
    if env_key:
        cookie["key"] = env_key
    if not str(cookie.get("key") or "").strip():
        raise ValueError("cookie.key 또는 AUTH_COOKIE_KEY 가 필요합니다.")
    return config


def build_authenticator(auth_config: dict[str, Any]) -> stauth.Authenticate:
    """비밀번호는 이미 bcrypt 해시이므로 auto_hash=False 로 이중 해시를 막습니다."""
    cookie = auth_config["cookie"]
    return stauth.Authenticate(
        auth_config["credentials"],
        cookie["name"],
        cookie["key"],
        float(cookie.get("expiry_days") or 1),
        auto_hash=False,
        login_sleep_time=0,
    )


def _get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    """Call the FastAPI backend and return parsed JSON."""
    url = f"{backend_api_url()}{path}"
    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.RequestException:
        raise RuntimeError("백엔드 API에 연결하지 못했습니다.") from None
    if response.status_code == 404:
        raise NoDataError("선택한 측정소와 기간에 조회 가능한 데이터가 없습니다.")
    if response.status_code == 422:
        raise ValueError("입력값이 허용된 형식이 아닙니다.")
    if response.status_code == 429:
        raise RuntimeError("요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.")
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError("백엔드 API에서 데이터를 불러오지 못했습니다.") from None
    return payload


def create_prediction_figure(payload: dict[str, Any]) -> go.Figure:
    """Plot TimescaleDB observations and the latest T+24 forecast."""
    history = pd.DataFrame(payload.get("history") or [])
    prediction = payload.get("prediction") or {}
    figure = go.Figure()
    if history.empty or not prediction:
        figure.add_annotation(
            text="표시할 시계열 데이터가 없습니다.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        figure.update_layout(template="plotly_white", title="PM2.5 실측값과 T+24 예측")
        return figure

    history["measured_at"] = pd.to_datetime(history["measured_at"], utc=True)
    figure.add_trace(
        go.Scatter(
            x=history["measured_at"],
            y=history["pm25"],
            mode="lines",
            name="실측 PM2.5",
            line={"color": "#2563EB"},
        )
    )
    latest_time = history["measured_at"].iloc[-1]
    latest_value = history["pm25"].iloc[-1]
    target_time = pd.Timestamp(prediction["target_time"])
    predicted = float(prediction["predicted_pm25"])
    figure.add_trace(
        go.Scatter(
            x=[latest_time, target_time],
            y=[latest_value, predicted],
            mode="lines+markers",
            name="T+24 예측",
            line={"color": "#DC2626", "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[target_time, target_time],
            y=[prediction["lower_bound"], prediction["upper_bound"]],
            mode="lines",
            line={"color": "rgba(220, 38, 38, 0.45)", "width": 8},
            name="경험적 오차 범위",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title="PM2.5 실측값과 T+24 예측",
        xaxis_title="측정 시각",
        yaxis_title="농도 (µg/m³)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return figure


def moe_pm25_status(pm25: float | None) -> tuple[str, str]:
    """Return the MOE grade label and marker color for a PM2.5 value."""
    if pm25 is None or not math.isfinite(pm25) or pm25 < 0:
        return "데이터 없음", "#6B7280"
    for upper_bound, label, color in MOE_PM25_GRADES:
        if pm25 <= upper_bound:
            return label, color
    return "매우나쁨", "#DC2626"


def _parse_pm25(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 500.0:
        return None
    return number


def _secure_marker_popup(station_name: str, pm25_text: str, status: str) -> str:
    """Build a Folium HTML popup with every dynamic field escaped."""
    safe_name = html.escape(station_name, quote=True)
    safe_pm25 = html.escape(pm25_text, quote=True)
    safe_status = html.escape(status, quote=True)
    return (
        "<div style='min-width:150px;font-family:sans-serif;font-size:13px'>"
        f"<strong>{safe_name}</strong><br/>"
        f"현재 PM2.5: {safe_pm25}<br/>"
        f"대기질 상태: {safe_status}"
        "</div>"
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest_station_pm25(station_name: str, as_of: date) -> float | None:
    """Return the newest PM2.5 reading for one whitelist-validated station."""
    safe_station = validate_station_name(station_name)
    payload = _get_json(
        "/api/v1/air-quality",
        params={
            "station_name": safe_station,
            "start_date": (as_of - timedelta(days=2)).isoformat(),
            "end_date": as_of.isoformat(),
            "pollutant": "PM2.5",
        },
    )
    if not isinstance(payload, dict):
        raise RuntimeError("대기질 응답 형식이 올바르지 않습니다.")
    metadata = payload.get("metadata") or {}
    if metadata.get("station_name") != safe_station:
        raise RuntimeError("선택한 측정소와 다른 데이터가 반환되었습니다.")
    latest_value: float | None = None
    for row in payload.get("data") or []:
        if not isinstance(row, dict):
            continue
        parsed = _parse_pm25(row.get("value"))
        if parsed is not None:
            latest_value = parsed
    return latest_value


def collect_map_station_data(
    station_names: list[str],
    *,
    as_of: date,
) -> list[dict[str, Any]]:
    """Collect sanitized map rows; skip invalid names and isolate API failures."""
    observations: list[dict[str, Any]] = []
    for raw_name in station_names:
        try:
            station_name = validate_station_name(str(raw_name))
        except (TypeError, ValueError):
            LOGGER.warning("Skipped a station name that failed whitelist validation")
            continue
        coordinates = SEOUL_STATION_COORDINATES.get(station_name)
        if coordinates is None:
            continue
        try:
            pm25 = fetch_latest_station_pm25(station_name, as_of)
        except NoDataError:
            pm25 = None
        except (RuntimeError, ValueError):
            LOGGER.error("Map observation fetch failed for a validated station")
            pm25 = None
        status, color = moe_pm25_status(pm25)
        observations.append(
            {
                "station_name": station_name,
                "latitude": coordinates[0],
                "longitude": coordinates[1],
                "pm25": pm25,
                "status": status,
                "color": color,
            }
        )
    return observations


def render_secure_air_quality_map(
    current_data: list[dict[str, Any]],
    *,
    selected_station: str | None = None,
) -> folium.Map:
    """Render a Seoul map with XSS-safe, MOE color-coded station markers."""
    station_map = folium.Map(
        location=list(SEOUL_MAP_CENTER),
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )
    selected = None
    if selected_station is not None:
        try:
            selected = validate_station_name(selected_station)
        except (TypeError, ValueError):
            selected = None

    for row in current_data:
        if not isinstance(row, dict):
            continue
        try:
            station_name = validate_station_name(str(row.get("station_name", "")))
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            continue
        pm25 = _parse_pm25(row.get("pm25"))
        status, color = moe_pm25_status(pm25)
        if isinstance(row.get("status"), str) and row.get("status"):
            try:
                status = str(row["status"])
            except (TypeError, ValueError):
                pass
        if isinstance(row.get("color"), str) and row.get("color", "").startswith("#"):
            color = str(row["color"])
        pm25_text = "측정값 없음" if pm25 is None else f"{pm25:.1f} µg/m³"
        is_selected = station_name == selected
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=14 if is_selected else 10,
            color=color,
            weight=3 if is_selected else 2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=html.escape(station_name, quote=True),
            popup=folium.Popup(
                _secure_marker_popup(station_name, pm25_text, status),
                max_width=260,
            ),
        ).add_to(station_map)
    return station_map


def create_shap_figure(
    rows: list[dict[str, Any]],
    *,
    value_column: str,
    title: str,
) -> go.Figure:
    frame = pd.DataFrame(rows)
    if frame.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="표시할 SHAP 기여도가 없습니다.",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        figure.update_layout(template="plotly_white", title=title)
        return figure
    frame = frame.sort_values(value_column)
    colors = [
        "#DC2626" if float(value) > 0 else "#2563EB"
        for value in frame[value_column]
    ]
    figure = go.Figure(
        go.Bar(
            x=frame[value_column],
            y=frame["feature"],
            orientation="h",
            marker_color=colors,
        )
    )
    figure.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="SHAP 기여도",
        yaxis_title="특성",
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


@st.cache_data(ttl=60, show_spinner=False)
def fetch_available_stations() -> list[str]:
    payload = _get_json("/api/v1/stations")
    if not isinstance(payload, dict):
        raise RuntimeError("측정소 목록 응답 형식이 올바르지 않습니다.")
    stations: list[str] = []
    for row in payload.get("stations", []):
        if not isinstance(row, dict) or "station_name" not in row:
            continue
        stations.append(validate_station_name(str(row["station_name"])))
    if not stations:
        raise NoDataError("TimescaleDB에 조회 가능한 측정소가 없습니다.")
    return stations


@st.cache_data(ttl=60, show_spinner=False)
def fetch_dashboard_data(
    station_name: str,
    start_date: date,
    end_date: date,
    alert_threshold: float,
) -> dict[str, Any]:
    payload = _get_json(
        f"/api/v1/stations/{quote(station_name, safe='')}/dashboard",
        params={
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "alert_threshold": alert_threshold,
        },
    )
    if not isinstance(payload, dict):
        raise RuntimeError("백엔드 API 응답 형식이 올바르지 않습니다.")
    if payload.get("station_name") != station_name:
        raise RuntimeError("선택한 측정소와 다른 데이터가 반환되었습니다.")
    if not payload.get("history") or not payload.get("prediction"):
        raise NoDataError("선택한 측정소와 기간에 조회 가능한 데이터가 없습니다.")
    return payload


def render_license_footer() -> None:
    st.sidebar.divider()
    st.sidebar.markdown(LICENSE_TEXT)


def render_sidebar() -> tuple[tuple[date, date], str, float] | None:
    st.sidebar.header("조회 조건")
    try:
        stations = fetch_available_stations()
    except NoDataError as exc:
        st.sidebar.warning(display_text(exc))
        render_license_footer()
        return None
    except (RuntimeError, ValueError) as exc:
        st.sidebar.error(display_text(exc))
        render_license_footer()
        return None

    today = date.today()
    default_end = date(2026, 2, 28)
    default_start = default_end - timedelta(days=7)
    today = date.today()

    selected_dates = st.sidebar.date_input(
        "조회 기간",
        value=(default_start, default_end),
        min_value=date(2024, 12, 31),
        max_value=today,
    )
    station_input = st.sidebar.selectbox(
        "측정소명",
        options=stations,
        index=stations.index("종로구") if "종로구" in stations else 0,
        help="FastAPI `/api/v1/stations`에서 조회한 TimescaleDB 측정소입니다.",
    )
    st.sidebar.text_input("분석 항목", value="PM2.5", disabled=True)
    st.sidebar.text_input("예측 시계열", value="T+24시간", disabled=True)
    alert_threshold = st.sidebar.number_input(
        "고농도 알림 기준 (µg/m³)",
        min_value=0.0,
        max_value=500.0,
        value=35.0,
        step=1.0,
    )

    try:
        date_range = validate_date_range(selected_dates)
        safe_station = validate_station_name(str(station_input))
        validated_threshold = float(alert_threshold)
        if not 0.0 <= validated_threshold <= 500.0:
            raise ValueError("알림 기준은 0~500 범위여야 합니다.")
    except (TypeError, ValueError) as exc:
        st.sidebar.error(display_text(exc))
        render_license_footer()
        return None

    st.sidebar.caption(
        f"선택: {display_text(safe_station)} · PM2.5 · T+24시간"
    )
    render_license_footer()
    return date_range, safe_station, validated_threshold


def render_shap_section(payload: dict[str, Any]) -> None:
    """Keep the two SHAP charts aligned by placing the alert above them."""
    st.subheader("방지시설 가동 권장")
    st.info(display_text(payload.get("recommendation", "권장 정보가 없습니다.")))

    shap_column, global_column = st.columns(2, gap="large")
    with shap_column:
        st.subheader("이번 예측의 고농도 기여 요인")
        st.plotly_chart(
            create_shap_figure(
                list(payload.get("instance_shap_positive") or []),
                value_column="shap_value",
                title="양(+)의 instance TreeSHAP",
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    with global_column:
        st.subheader("전역 평균 |SHAP| 중요도")
        st.plotly_chart(
            create_shap_figure(
                list(payload.get("global_shap") or []),
                value_column="importance",
                title="전역 평균 |SHAP| 중요도",
            ),
            width="stretch",
            config={"displaylogo": False},
        )


def render_esg_vision_tab() -> None:
    """B2B ESG closed-loop vision, anomaly drill, and disclosure preview."""
    st.subheader("폐쇄 루프 환경 관제 아키텍처")
    st.caption(
        "공공 대기질(Macro)과 사업장 IoT(Micro)를 한 예측 엔진에 결합해 "
        "방지시설을 필요할 때만 가동합니다."
    )

    public_column, plant_column, control_column = st.columns(3, gap="large")
    with public_column:
        with st.container(border=True):
            st.markdown("**① 공공데이터 · Macro**")
            st.markdown("에어코리아 · 기상청 ASOS")
            st.caption("권역 배경농도, 기상, T+24 예측 입력")
    with plant_column:
        with st.container(border=True):
            st.markdown("**② 공장 내부 IoT · Micro**")
            st.markdown("공정별 분진 · 가스 센서")
            st.caption("국소 오염원, 설비 가동 상태")
    with control_column:
        with st.container(border=True):
            st.markdown("**③ 선별적 자동 제어**")
            st.markdown("스크러버 · 집진기 출력 조절")
            st.caption("고농도 구간만 가동률 상향")

    st.markdown(
        "공공데이터 + 공장 IoT  →  **중앙 AI 예측 시스템**  →  "
        "방지시설 선별적 자동 제어"
    )
    st.info(
        "외부 농도가 낮고 공정 배출이 안정적이면 방지시설을 대기 모드로 두어 "
        "전력을 줄입니다. 국소 오염이 급증하면 해당 공정 설비만 올려 "
        "불필요한 전 공장 가동 없이 배출을 억제합니다."
    )

    st.divider()
    st.subheader("이상 탐지 시뮬레이션")
    with st.container(border=True):
        st.caption("공공 관측망만으로는 보이지 않는 사업장 내부 급증을 재현합니다.")
        simulation_on = st.toggle(
            "시뮬레이션 가동: 공장 내부 국소 오염원 급증 상황",
            value=False,
        )
        if simulation_on:
            public_metric, plant_metric = st.columns(2, gap="large")
            with public_metric:
                st.metric(
                    "공공 관측소 수치",
                    "12.4 µg/m³",
                    "정상 · 좋음",
                    delta_color="off",
                )
                st.success("외부 대기질은 환경부 좋음 구간입니다.")
            with plant_metric:
                st.metric(
                    "사업장 IoT 수치",
                    "86.0 µg/m³",
                    "위험 · 3번 공정",
                    delta_color="off",
                )
                st.error("공정 내부 농도가 매우나쁨 구간입니다.")
            st.error(
                "경고: 외부 대기질은 정상이나 사업장 내 3번 공정의 오염도가 "
                "급증했습니다. 스크러버(Scrubber) 가동률을 80%로 상향 조정합니다."
            )
        else:
            st.info(
                "시뮬레이션을 켜면 공공 관측소는 정상, 사업장 IoT는 위험인 "
                "국소 오염 시나리오와 자동 제어 권고가 표시됩니다."
            )

    st.divider()
    st.subheader("ESG 성과 요약")
    st.caption("이번 달 폐쇄 루프 제어로 산정한 공시 참고 지표입니다.")
    reduction_column, power_column, credit_column = st.columns(3, gap="large")
    reduction_column.metric("누적 오염물질 저감량", "1,245 kg", "+8.2%")
    power_column.metric("불필요 방지시설 제어 전력 절감량", "340 kWh", "+5.4%")
    credit_column.metric("탄소 배출권 환산 가치", "$1,200", "+$150")

    report_buffer = StringIO()
    pd.DataFrame(
        {
            "항목": [
                "누적 오염물질 저감량",
                "전력 절감량",
                "탄소 배출권 환산 가치",
                "제어 방식",
            ],
            "값": [
                "1245 kg",
                "340 kWh",
                "1200 USD",
                "공공 Macro + 사업장 IoT Micro 폐쇄 루프",
            ],
            "기간": ["2026-07"] * 4,
        }
    ).to_csv(report_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        "📥 ESG 공시용 리포트 다운로드 (PDF)",
        data=report_buffer.getvalue().encode("utf-8-sig"),
        file_name="lotus_ent_esg_disclosure_preview.csv",
        mime="text/csv",
        help="공시 초안용 CSV 미리보기입니다. 운영 데이터가 아닌 시뮬레이션 값입니다.",
    )
    st.caption(
        f"미리보기 생성 시각: {display_text(datetime.now().strftime('%Y-%m-%d %H:%M'))} · "
        "실제 공시 문서가 아닌 내부 검토용 산출물입니다."
    )


def render_dashboard(filters: tuple[tuple[date, date], str, float]) -> None:
    date_range, station, alert_threshold = filters
    with st.spinner("선택한 측정소의 실측·예측 데이터를 조회하는 중입니다."):
        try:
            payload = fetch_dashboard_data(
                station,
                date_range[0],
                date_range[1],
                alert_threshold,
            )
        except NoDataError as exc:
            st.warning(display_text(exc))
            return
        except (RuntimeError, ValueError) as exc:
            st.error(display_text(exc))
            return

    current = payload["current"]
    prediction = payload["prediction"]
    performance = payload["performance"]
    ensemble_metrics = performance["metrics"]["ensemble"]

    st.subheader("실시간 대기질 분석 및 AI 예측")
    st.caption(
        f"{display_text(station)} · {date_range[0].isoformat()} ~ "
        f"{date_range[1].isoformat()} · "
        "TimescaleDB · 전역 다중 측정소 LSTM/XGBoost T+24 예측"
    )
    if payload.get("model_scope_warning"):
        st.warning(display_text(payload["model_scope_warning"]))

    current_kpi, grade_kpi, forecast_kpi, target_kpi = st.columns(4)
    current_kpi.metric(
        "현재 PM2.5",
        f"{float(current['pm25']):.1f} µg/m³",
    )
    grade_kpi.metric("위험 등급", display_text(current["risk_grade"]))
    forecast_kpi.metric(
        "T+24 예측 PM2.5",
        f"{float(prediction['predicted_pm25']):.1f} µg/m³",
    )
    target_kpi.metric(
        "예측 목표 시각",
        pd.Timestamp(prediction["target_time"]).strftime("%m-%d %H시"),
    )

    st.subheader("실제값과 예측값")
    st.plotly_chart(
        create_prediction_figure(payload),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "오차 범위는 테스트 잔차 절댓값의 90백분위 기반이며 "
        "통계적 95% 신뢰구간이 아닙니다."
    )

    render_shap_section(payload)

    st.subheader("모델 성능 평가")
    rmse_column, mae_column, r2_column, weight_column = st.columns(4)
    rmse_column.metric("Test RMSE", f"{float(ensemble_metrics['rmse']):.2f}")
    mae_column.metric("Test MAE", f"{float(ensemble_metrics['mae']):.2f}")
    r2_column.metric("Test R²", f"{float(ensemble_metrics['r2']):.3f}")
    weight_column.metric(
        "LSTM 앙상블 비중",
        f"{float(performance['ensemble_lstm_weight']) * 100:.1f}%",
    )
    trained_stations = [
        display_text(name)
        for name in (performance.get("station_categories") or [])
        if str(name).strip()
    ]
    if trained_stations:
        st.caption("전역 모델 학습 측정소: " + ", ".join(trained_stations))
    if bool(performance["production_ready"]):
        st.success(display_text(performance["assessment"]))
    else:
        st.error(display_text(performance["assessment"]))
    cross_validation = pd.DataFrame(performance["time_series_cv"])
    cross_validation.index = [
        f"Fold {index + 1}" for index in range(len(cross_validation))
    ]
    st.dataframe(cross_validation, width="stretch")

    st.subheader("서울 측정소 실시간 지도")
    try:
        available_stations = set(fetch_available_stations())
        mapped_stations = [
            name
            for name in SEOUL_STATION_COORDINATES
            if name in available_stations
        ] or list(SEOUL_STATION_COORDINATES)
        map_data = collect_map_station_data(
            mapped_stations,
            as_of=date.today(),
        )
        if not map_data:
            st.info("지도에 표시할 측정소 좌표가 없습니다.")
        else:
            st_folium(
                render_secure_air_quality_map(
                    map_data,
                    selected_station=station,
                ),
                width=None,
                height=420,
                returned_objects=[],
                key="air-quality-map",
            )
            st.caption(
                "마커 색상은 환경부 PM2.5 기준입니다. "
                "파랑 좋음(0–15), 초록 보통(16–35), 주황 나쁨(36–75), 빨강 매우나쁨(76+)."
            )
    except (RuntimeError, ValueError) as exc:
        st.error(display_text(exc))
    except Exception:
        LOGGER.error("Secure air-quality map rendering failed")
        st.error("지도를 불러오지 못했습니다. 관리자에게 문의해 주세요.")


def _lookup_user(auth_config: dict[str, Any], username: str) -> dict[str, Any] | None:
    users = (auth_config.get("credentials") or {}).get("usernames") or {}
    return users.get(username) or users.get(username.lower())


def _complete_login(username: str, profile: dict[str, Any]) -> None:
    st.session_state["authentication_status"] = True
    st.session_state["username"] = username.lower()
    st.session_state["name"] = profile.get("name") or username
    st.session_state["email"] = profile.get("email")
    st.session_state["roles"] = profile.get("roles") or []
    st.session_state["logout"] = None


def render_password_login_form(auth_config: dict[str, Any]) -> None:
    """라이브러리 쿠키 복원이 실패해도 아이디·비밀번호로 로그인할 수 있게 합니다."""
    with st.form("air_xai_password_login"):
        st.subheader("로그인")
        username = st.text_input("아이디", autocomplete="username")
        password = st.text_input("비밀번호", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("로그인")
    if not submitted:
        st.info("로그인 창에서 아이디와 비밀번호를 입력해 주세요. 로그인 전에는 데이터가 표시되지 않습니다.")
        return
    profile = _lookup_user(auth_config, username.strip())
    hashed = str((profile or {}).get("password") or "")
    if (
        profile is None
        or not hashed
        or not bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    ):
        st.session_state["authentication_status"] = False
        st.error("아이디 또는 비밀번호가 잘못되었습니다.")
        return
    _complete_login(username.strip(), profile)
    st.rerun()


def render_login_screen(
    authenticator: stauth.Authenticate,
    auth_config: dict[str, Any],
) -> None:
    """쿠키 복원 또는 로그인 폼을 그립니다. 만료·구 아이디 쿠키는 무시합니다."""
    if not is_authenticated():
        st.title("대기질 예측 · ESG 관제")
        st.caption("권한이 있는 사용자만 대시보드에 접근할 수 있습니다.")
    try:
        authenticator.login(
            location="main",
            max_login_attempts=5,
            fields={
                "Form name": "로그인",
                "Username": "아이디",
                "Password": "비밀번호",
                "Login": "로그인",
                "Captcha": "캡차",
            },
            key="LotusLogin",
        )
    except LoginError:
        st.session_state["authentication_status"] = None
        st.session_state["username"] = None
        st.session_state["logout"] = True
        render_password_login_form(auth_config)
        return
    except Exception:
        LOGGER.exception("streamlit-authenticator login widget failed")
        render_password_login_form(auth_config)
        return
    status = st.session_state.get("authentication_status")
    if status is True:
        return
    if status is False:
        st.error("아이디 또는 비밀번호가 잘못되었습니다.")
        return
    st.info("로그인 창에서 아이디와 비밀번호를 입력해 주세요. 로그인 전에는 데이터가 표시되지 않습니다.")


def render_auth_sidebar(
    authenticator: stauth.Authenticate,
    auth_config: dict[str, Any],
) -> None:
    """로그인 성공 후에만 사용자 정보와 로그아웃을 사이드바에 표시합니다."""
    require_authentication()
    display_name = escape_html(st.session_state.get("name") or current_username())
    roles = ", ".join(current_roles(auth_config)) or "user"
    st.sidebar.header("계정")
    st.sidebar.write(f"사용자: {display_name}")
    st.sidebar.caption(f"아이디: {escape_html(current_username())}")
    st.sidebar.caption(f"권한: {escape_html(roles)}")
    authenticator.logout(button_name="로그아웃", location="sidebar", key="LotusLogout")


def render_secure_tools(auth_config: dict[str, Any]) -> None:
    """XSS·업로드·경로 이탈 가드가 적용된 운영 도구입니다."""
    require_authentication()
    st.subheader("안전한 검색")
    raw_query = st.text_input(
        "측정소 또는 메모 검색",
        max_chars=80,
        help="한글, 영문, 숫자, 공백, . _ - 만 허용합니다.",
    )
    if raw_query:
        try:
            query = validate_search_query(raw_query)
        except SecurityError as exc:
            st.error(escape_html(exc))
        else:
            # 사용자 입력을 HTML로 넣을 때도 escape 후에만 unsafe_allow_html 을 켭니다.
            safe_markdown(f"검색어: {query}", unsafe_allow_html=True)
            if SAFE_IDENTIFIER_PATTERN.fullmatch(query):
                statement, params = parameterized_station_lookup(query)
                st.caption("DB 조회는 f-string SQL이 아니라 파라미터 바인딩만 사용합니다.")
                st.code(str(statement) + f"\nparams={params}", language="sql")

    st.subheader("파일 업로드")
    st.caption("관리자만 업로드할 수 있습니다. 허용 확장자 .csv / .xlsx, 최대 10MB.")
    if "admin" not in current_roles(auth_config):
        st.info("업로드는 admin 권한이 필요합니다.")
        return

    require_role(auth_config, "admin")
    uploaded = st.file_uploader(
        "관측 파일",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        return
    try:
        saved = save_uploaded_file(uploaded, username=current_username())
    except SecurityError as exc:
        st.error(escape_html(exc))
        return
    st.success(f"저장 완료: {escape_html(saved.name)}")
    st.caption(f"경로가 data/uploads 안에 고정되었습니다: {escape_html(saved)}")


def render_authenticated_app(auth_config: dict[str, Any]) -> None:
    require_authentication()
    st.title("대기질 예측 · ESG 관제")
    st.caption("공공 관측망과 사업장 제어를 연결하는 B2B 환경 운영 콘솔")
    filters = render_sidebar()
    monitoring_tab, esg_tab, security_tab = st.tabs(
        [
            "실시간 관제 & AI 예측",
            "B2B ESG 리포트 & IoT 비전",
            "보안 도구",
        ]
    )
    with monitoring_tab:
        if filters is None:
            st.warning("입력값을 확인한 뒤 다시 시도해 주세요.")
        else:
            render_dashboard(filters)
    with esg_tab:
        render_esg_vision_tab()
    with security_tab:
        render_secure_tools(auth_config)


def main() -> None:
    st.set_page_config(
        page_title="Air Quality ESG",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    try:
        auth_config = load_auth_config()
        authenticator = build_authenticator(auth_config)
        if not is_authenticated():
            render_login_screen(authenticator, auth_config)
        if not is_authenticated():
            return
        render_auth_sidebar(authenticator, auth_config)
        render_authenticated_app(auth_config)
    except FileNotFoundError as exc:
        st.error(escape_html(exc))
    except NoDataError as exc:
        LOGGER.info("Dashboard has no available station data")
        st.warning(display_text(exc))
    except RuntimeError as exc:
        LOGGER.error("Dashboard backend request failed")
        st.error(display_text(exc))
        st.code(
            "python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
        )
    except Exception as exc:
        LOGGER.exception("Dashboard rendering failed")
        st.error("화면을 불러오지 못했습니다. 관리자에게 문의해 주세요.")
        st.caption(escape_html(exc))


if __name__ == "__main__":
    main()
