"""
src/transcription.py — STT + 화자 매칭 모듈
=============================================
HuggingFace transformers + Whisper 를 사용해 오디오를 전사하고,
화자 분리 세그먼트와 타임스탬프를 매칭합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """화자 한 차례의 발언 단위."""
    speaker: str          # 화자 ID (예: "SPEAKER_00")
    start: float          # 시작 시간 (초)
    end: float            # 종료 시간 (초)
    text: str             # 전사 텍스트

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"[{self.speaker} {self.start:.1f}s-{self.end:.1f}s] {preview!r}"


class TranscriptionEngine:
    """Whisper 기반 STT + 화자 세그먼트 매칭.

    Args:
        settings: 전역 설정 객체
    """

    def __init__(self, settings=None) -> None:
        from .config import get_settings
        self.settings = settings or get_settings()
        self._pipe = None

    def _load_model(self) -> None:
        """Whisper 파이프라인 지연 로딩."""
        if self._pipe is not None:
            return

        model_id = f"openai/whisper-{self.settings.whisper_model_size}"
        logger.info("Whisper 모델 로딩: %s (device=%s)", model_id, self.settings.device)

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            dtype = torch.float16 if self.settings.device != "cpu" else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self.settings.device)

            processor = AutoProcessor.from_pretrained(model_id)

            generate_kwargs: dict = {"task": "transcribe"}
            if self.settings.whisper_language:
                generate_kwargs["language"] = self.settings.whisper_language

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=dtype,
                device=self.settings.device,
                generate_kwargs=generate_kwargs,
            )
        except ImportError as e:
            raise ImportError(
                "transformers / torch 패키지가 필요합니다. requirements.txt를 설치하세요."
            ) from e

        logger.info("Whisper 모델 로딩 완료.")

    def transcribe(self, audio_path: str | Path) -> List[dict]:
        """오디오 파일을 전사합니다.

        Returns:
            HuggingFace pipeline 출력 청크 리스트
            각 청크: {"text": str, "timestamp": (start_sec, end_sec)}
        """
        self._load_model()
        audio_path = Path(audio_path)
        logger.info("STT 실행: %s", audio_path.name)

        result = self._pipe(str(audio_path))
        chunks = result.get("chunks", [])
        logger.info("STT 완료 — %d 청크", len(chunks))
        return chunks

    def align_with_diarization(
        self,
        chunks: List[dict],
        segments: "List[SpeakerSegment]",  # noqa: F821
    ) -> List[Turn]:
        """STT 청크와 화자 세그먼트를 시간 기반으로 정렬·병합합니다.

        각 STT 청크를 오버랩이 가장 큰 화자 세그먼트에 배정합니다.
        """
        from .diarization import SpeakerSegment

        turns: List[Turn] = []
        unknown = "SPEAKER_UNKNOWN"

        for chunk in chunks:
            ts = chunk.get("timestamp", (None, None))
            chunk_start, chunk_end = ts if ts else (0.0, 0.0)
            text = chunk.get("text", "").strip()

            if not text:
                continue

            # chunk_start/end가 None인 경우 안전 처리
            chunk_start = float(chunk_start or 0.0)
            chunk_end = float(chunk_end or chunk_start + 0.1)

            # 오버랩이 가장 큰 화자 찾기
            best_speaker = unknown
            best_overlap = -1.0
            for seg in segments:
                overlap = min(chunk_end, seg.end) - max(chunk_start, seg.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg.speaker

            # 직전 Turn과 같은 화자이면 합치기
            if turns and turns[-1].speaker == best_speaker:
                turns[-1].text += " " + text
                turns[-1].end = chunk_end
            else:
                turns.append(Turn(
                    speaker=best_speaker,
                    start=chunk_start,
                    end=chunk_end,
                    text=text,
                ))

        logger.info("화자-전사 정렬 완료 — %d Turn", len(turns))
        return turns

    def run(
        self,
        audio_path: str | Path,
        segments: "Optional[List[SpeakerSegment]]" = None,  # noqa: F821
    ) -> List[Turn]:
        """전사 + 화자 매칭 통합 실행.

        Args:
            audio_path: 오디오 파일 경로
            segments: 화자 분리 세그먼트 (None이면 화자 정보 없이 전사만 수행)
        """
        chunks = self.transcribe(audio_path)

        if segments:
            return self.align_with_diarization(chunks, segments)

        # 화자 분리 없이 전사만 반환
        turns = []
        for chunk in chunks:
            ts = chunk.get("timestamp", (None, None))
            s, e = ts if ts else (0.0, 0.0)
            text = chunk.get("text", "").strip()
            if text:
                turns.append(Turn(
                    speaker="SPEAKER_00",
                    start=float(s or 0.0),
                    end=float(e or 0.0),
                    text=text,
                ))
        return turns
