"""
src/diarization.py — 화자 분리 모듈
=====================================
pyannote/speaker-diarization-3.1 을 사용해 오디오 파일에서
화자 세그먼트를 추출합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """화자 분리 단위 세그먼트."""
    start: float       # 시작 시간 (초)
    end: float         # 종료 시간 (초)
    speaker: str       # 화자 ID (예: "SPEAKER_00")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"[{self.speaker}] {self.start:.2f}s – {self.end:.2f}s ({self.duration:.2f}s)"


class DiarizationPipeline:
    """pyannote 화자 분리 파이프라인 래퍼.

    Args:
        settings: 전역 설정 객체
    """

    def __init__(self, settings=None) -> None:
        from .config import get_settings
        self.settings = settings or get_settings()
        self._pipeline = None

    def _load_pipeline(self):
        """파이프라인 지연 로딩 (첫 실행 시에만 모델 다운로드)."""
        if self._pipeline is not None:
            return

        logger.info("화자 분리 모델 로딩: %s", self.settings.diarization_model)

        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise ImportError(
                "pyannote.audio 패키지가 설치되지 않았습니다. "
                "`pip install pyannote.audio` 를 실행하세요."
            ) from e

        if not self.settings.hf_token:
            raise ValueError(
                "HF_TOKEN이 없습니다. .env 파일에 HF_TOKEN=hf_xxx 를 추가하세요. "
                "pyannote/speaker-diarization-3.1 라이선스 동의도 필요합니다: "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        self._pipeline = Pipeline.from_pretrained(
            self.settings.diarization_model,
        )
        import torch
        self._pipeline.to(torch.device(self.settings.device))
        logger.info("화자 분리 모델 로딩 완료.")

    def run(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> List[SpeakerSegment]:
        """오디오 파일에서 화자 세그먼트를 추출합니다.

        Args:
            audio_path: 분석할 오디오 파일 경로 (wav / mp3 등)
            min_speakers: 최소 화자 수 (None이면 자동 감지)
            max_speakers: 최대 화자 수 (None이면 자동 감지)

        Returns:
            화자 세그먼트 리스트 (시간 순 정렬)
        """
        audio_path = self._validate_audio_path(audio_path)
        self._load_pipeline()

        logger.info("화자 분리 실행: %s", audio_path.name)

        # torchaudio 2.9+가 torchcodec에 의존하므로, soundfile(libsndfile)로 직접 로드합니다.
        # soundfile은 WAV / FLAC / OGG / AIFF 등을 ffmpeg 없이 지원합니다.
        # MP3 / M4A 등은 지원하지 않으므로 WAV로 변환 후 사용을 권장합니다.
        import numpy as np
        import torch
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd

        _TARGET_SR = 16_000

        try:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(
                f"오디오 파일을 읽을 수 없습니다: {audio_path.name}\n"
                "MP3·M4A 포맷은 지원되지 않습니다. WAV / FLAC / OGG 파일을 사용하세요.\n"
                f"(원본 오류: {exc})"
            ) from exc

        # (time, channels) → mono: 채널 평균
        if data.ndim == 2:
            data = data.mean(axis=1)

        # 리샘플링 (원본 SR ≠ 16 kHz 인 경우)
        if sr != _TARGET_SR:
            g = gcd(sr, _TARGET_SR)
            data = resample_poly(data, _TARGET_SR // g, sr // g).astype(np.float32)

        # (time,) → (1, time) torch.Tensor
        waveform = torch.from_numpy(data).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": _TARGET_SR}

        kwargs: dict = {}
        _min = min_speakers or self.settings.min_speakers
        _max = max_speakers or self.settings.max_speakers
        if _min is not None:
            kwargs["min_speakers"] = _min
        if _max is not None:
            kwargs["max_speakers"] = _max

        diarization = self._pipeline(audio_input, **kwargs)

        # pyannote 최신 버전은 DiarizeOutput 객체를 반환합니다.
        # 구 버전(legacy=True)은 Annotation을 바로 반환합니다.
        if hasattr(diarization, "speaker_diarization"):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization  # Annotation 직접 반환(구 API)

        segments: List[SpeakerSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                )
            )

        segments.sort(key=lambda s: s.start)
        logger.info("화자 분리 완료 — %d 세그먼트 / 화자: %s",
                    len(segments),
                    list({s.speaker for s in segments}))
        return segments

    def _validate_audio_path(self, audio_path: str | Path) -> Path:
        """경로 탐색 공격 방지: rsc/data/ 하위 파일만 허용.
        # TODO(security): 업로드 파일 처리 시 확장자 화이트리스트 검증 추가.
        """
        from .config import get_settings
        cfg = get_settings()

        path = Path(audio_path).resolve()
        allowed_root = cfg.data_dir.resolve()

        # 경로 탐색 방지: data_dir 하위에 있는지 엄격 검증
        if not str(path).startswith(str(allowed_root) + "/"):
            raise ValueError(
                f"허용되지 않은 파일 경로입니다: {path}\n"
                f"오디오 파일은 {allowed_root}/ 하위에 있어야 합니다."
            )

        if not path.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {path}")

        return path
