from __future__ import annotations

from typing import Final


DEFAULT_MODEL_ARTIFACT: Final[str] = "pm25_retrained_v4"
CANDIDATE_MODEL_ARTIFACTS: Final[tuple[str, ...]] = (
    "pm25_t24_ensemble",
    "pm25_retrained_v4",
)
