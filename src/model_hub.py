from __future__ import annotations

import logging
import os
import re
from typing import Final

from huggingface_hub import snapshot_download

from src.model_registry import DEFAULT_MODEL_ARTIFACT
from src.utils.path import resolve_model_path


LOGGER = logging.getLogger(__name__)
_ARTIFACT_PATTERN: Final = re.compile(r"^[0-9A-Za-z_-]{1,50}$")
_HF_REPO_ID_PATTERN: Final = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98})/[A-Za-z0-9._-]{1,100}$"
)


def download_model_from_hf(artifact_name: str = DEFAULT_MODEL_ARTIFACT) -> None:
    """Hugging Face Hub에서 모델 번들을 다운로드하여 로컬 폴더에 저장합니다."""
    if not _ARTIFACT_PATTERN.fullmatch(artifact_name):
        raise ValueError("Invalid model artifact name")

    if os.getenv("RENDER", "").strip().lower() != "true":
        LOGGER.info("Not running on Render; using local model files")
        return

    repo_id = os.getenv("HF_REPO_ID", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    if not repo_id:
        raise RuntimeError("HF_REPO_ID is unset on Render")
    if not _HF_REPO_ID_PATTERN.fullmatch(repo_id):
        raise RuntimeError("HF_REPO_ID is not a valid Hugging Face model id")

    local_dir = resolve_model_path(artifact_name)
    local_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading serving artifacts from Hugging Face repo %s", repo_id)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        token=hf_token,
        ignore_patterns=["*.md", ".gitattributes"],
    )
    LOGGER.info("Hugging Face serving artifacts downloaded")
