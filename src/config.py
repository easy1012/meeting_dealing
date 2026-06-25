"""
src/config.py — 전역 설정 및 경로 관리
========================================
환경변수를 .env에서 로드하며, 코드에 시크릿을 하드코딩하지 않습니다.
# TODO(security): 프로덕션 환경에서는 KMS/Vault 등 시크릿 관리 솔루션을 사용하세요.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings  # type: ignore[import]

# 프로젝트 루트: meeting_dealing/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RSC_DIR = PROJECT_ROOT / "rsc"
DATA_DIR = RSC_DIR / "data"
OUTPUT_DIR_DEFAULT = RSC_DIR / "output"

# .env 로드 (상위 디렉토리도 탐색)
_env_path = PROJECT_ROOT / ".env"
if not _env_path.exists():
    _env_path = PROJECT_ROOT.parent / ".env"
load_dotenv(_env_path)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """애플리케이션 전체 설정.

    .env 파일 또는 환경변수에서 값을 읽습니다.
    HF_TOKEN 등 시크릿은 절대 기본값으로 하드코딩하지 않습니다.
    """

    # ── HuggingFace ──────────────────────────────────────────
    hf_token: str = ""
    """HuggingFace Access Token. pyannote 화자분리에 필수."""

    # ── 디바이스 ─────────────────────────────────────────────
    device: str = "cuda"
    """추론 디바이스: 'cuda' | 'mps' | 'cpu'"""

    # ── Whisper STT ──────────────────────────────────────────
    whisper_model_size: str = "large-v3"
    whisper_language: str | None = "ko"
    """STT 언어 힌트. None이면 자동 감지."""

    # ── sLLM ─────────────────────────────────────────────────
    sllm_model_id: str = "google/gemma-3-4b-it"
    """로컬 추론에 사용할 sLLM 모델 ID (HuggingFace Hub)."""
    sllm_max_new_tokens: int = 1024
    sllm_temperature: float = 0.1

    # ── 화자 분리 ─────────────────────────────────────────────
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None

    # ── 경로 ─────────────────────────────────────────────────
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR_DEFAULT

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"

    @field_validator("hf_token")
    @classmethod
    def _warn_if_empty_token(cls, v: str) -> str:
        if not v:
            logger.warning(
                "HF_TOKEN이 설정되지 않았습니다. "
                "pyannote 화자분리 모델은 HuggingFace 토큰이 필요합니다. "
                ".env 파일에 HF_TOKEN=hf_xxx 형태로 설정하세요."
            )
        return v

    @field_validator("device")
    @classmethod
    def _auto_device(cls, v: str) -> str:
        """CUDA/MPS 사용 불가 시 CPU로 폴백."""
        if v == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA 불가 — CPU로 폴백합니다.")
                    return "cpu"
            except ImportError:
                return "cpu"
        if v == "mps":
            try:
                import torch
                if not torch.backends.mps.is_available():
                    logger.warning("MPS 불가 — CPU로 폴백합니다.")
                    return "cpu"
            except ImportError:
                return "cpu"
        return v


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """싱글톤 설정 인스턴스 반환."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        # 출력 디렉토리 생성
        _settings_instance.output_dir.mkdir(parents=True, exist_ok=True)
    return _settings_instance
