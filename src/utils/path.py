

from __future__ import annotations

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DATA_ROOT: Final = (PROJECT_ROOT / "data").resolve()
MODELS_ROOT: Final = (PROJECT_ROOT / "models").resolve()
STORAGE_ROOTS: Final = {
    "data": DATA_ROOT,
    "models": MODELS_ROOT,
}


def resolve_under_root(
    storage: str,
    relative_path: str | Path,
) -> Path:
    """Resolve a relative path and guarantee it remains below its root."""
    if storage not in STORAGE_ROOTS:
        raise ValueError("Unknown storage area")
    untrusted_path = Path(relative_path)
    if untrusted_path.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    root = STORAGE_ROOTS[storage]
    resolved = (root / untrusted_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path must remain inside the {storage} directory"
        ) from None
    return resolved


def resolve_data_path(relative_path: str | Path) -> Path:
    return resolve_under_root("data", relative_path)


def resolve_model_path(relative_path: str | Path) -> Path:
    return resolve_under_root("models", relative_path)
