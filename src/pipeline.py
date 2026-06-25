"""
src/pipeline.py — 회의 전투력 측정 메인 파이프라인
=====================================================
화자 분리 → STT → sLLM 정리 → 스코어링 → 리포트 를 순서대로 실행합니다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과 집계."""
    audio_path: Path
    segments: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    cleaned: Optional[object] = None
    score: Optional[object] = None
    report_path: Optional[Path] = None
    elapsed_sec: float = 0.0


class MeetingPipeline:
    """회의 분석 파이프라인.

    Args:
        settings: 전역 설정. None이면 get_settings()로 자동 로드.
        skip_diarization: True이면 화자 분리 건너뜀 (단일 화자 가정)
        skip_llm: True이면 sLLM 정리 건너뜀 (스코어링만 규칙 기반으로)
        save_json: True이면 JSON 리포트 자동 저장
    """

    def __init__(
        self,
        settings=None,
        skip_diarization: bool = False,
        skip_llm: bool = False,
        save_json: bool = True,
    ) -> None:
        from .config import get_settings

        self.settings = settings or get_settings()
        self.skip_diarization = skip_diarization
        self.skip_llm = skip_llm
        self.save_json = save_json

        # 모듈 지연 초기화
        self._diarizer = None
        self._transcriber = None
        self._cleaner = None
        self._scorer = None
        self._reporter = None

    # ── 프로퍼티 (지연 초기화) ───────────────────────────────────────────────

    @property
    def diarizer(self):
        if self._diarizer is None:
            from .diarization import DiarizationPipeline
            self._diarizer = DiarizationPipeline(self.settings)
        return self._diarizer

    @property
    def transcriber(self):
        if self._transcriber is None:
            from .transcription import TranscriptionEngine
            self._transcriber = TranscriptionEngine(self.settings)
        return self._transcriber

    @property
    def cleaner(self):
        if self._cleaner is None:
            from .transcript_cleaner import TranscriptCleaner
            self._cleaner = TranscriptCleaner(self.settings)
        return self._cleaner

    @property
    def scorer(self):
        if self._scorer is None:
            from .scorer import MeetingScorer
            self._scorer = MeetingScorer()
        return self._scorer

    @property
    def reporter(self):
        if self._reporter is None:
            from .report import ReportGenerator
            self._reporter = ReportGenerator(self.settings)
        return self._reporter

    # ── 파이프라인 실행 ──────────────────────────────────────────────────────

    def run(
        self,
        audio_path: str | Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        print_report: bool = True,
    ) -> PipelineResult:
        """전체 파이프라인 실행.

        Args:
            audio_path: 분석할 오디오 파일 경로
            min_speakers: 최소 화자 수 힌트
            max_speakers: 최대 화자 수 힌트
            print_report: True이면 콘솔에 리포트 출력

        Returns:
            PipelineResult — 중간 결과 및 최종 스코어 포함
        """
        t0 = time.time()
        audio_path = Path(audio_path)
        result = PipelineResult(audio_path=audio_path)

        logger.info("=" * 50)
        logger.info("▶ 파이프라인 시작: %s", audio_path.name)
        logger.info("=" * 50)

        # ── Step 1: 화자 분리 ──────────────────────────────────────────────
        if not self.skip_diarization:
            logger.info("[Step 1/4] 화자 분리...")
            result.segments = self.diarizer.run(
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        else:
            logger.info("[Step 1/4] 화자 분리 건너뜀.")

        # ── Step 2: STT ───────────────────────────────────────────────────
        logger.info("[Step 2/4] 음성 인식(STT)...")
        result.turns = self.transcriber.run(
            audio_path,
            segments=result.segments if result.segments else None,
        )

        if not result.turns:
            logger.warning("전사 결과가 없습니다. 파이프라인을 종료합니다.")
            return result

        # ── Step 3: sLLM 전사 정리 ────────────────────────────────────────
        if not self.skip_llm:
            logger.info("[Step 3/4] sLLM 전사 정리...")
            result.cleaned = self.cleaner.clean(result.turns)
        else:
            from .transcript_cleaner import CleanedTranscript
            result.cleaned = CleanedTranscript(raw_turns=result.turns)
            logger.info("[Step 3/4] sLLM 건너뜀.")

        # ── Step 4: 스코어링 ─────────────────────────────────────────────
        logger.info("[Step 4/4] 스코어링...")
        result.score = self.scorer.score(result.turns, result.cleaned)

        # ── 리포트 ───────────────────────────────────────────────────────
        if print_report:
            self.reporter.print_console(result.score, audio_path.name)

        if self.save_json and result.score:
            result.report_path = self.reporter.save_json(
                result.score, audio_path.name
            )

        result.elapsed_sec = time.time() - t0
        logger.info("✅ 파이프라인 완료 — 소요 시간: %.1fs", result.elapsed_sec)
        return result

    def run_text_only(
        self,
        transcript_text: str,
        speakers: Optional[List[str]] = None,
        print_report: bool = True,
    ) -> PipelineResult:
        """텍스트 전사본만으로 분석 (오디오 없이 테스트용).

        Args:
            transcript_text: "[SPEAKER_00] 발언 내용\\n[SPEAKER_01] 발언 내용" 형식
            speakers: 화자 ID 목록 (None이면 자동 파싱)
            print_report: 콘솔 출력 여부
        """
        from .transcription import Turn
        from .transcript_cleaner import CleanedTranscript

        import re
        turns = []
        pattern = re.compile(r"\[([^\]]+)\]\s*(.+)")
        for i, line in enumerate(transcript_text.strip().splitlines()):
            m = pattern.match(line.strip())
            if m:
                speaker, text = m.group(1), m.group(2)
                turns.append(Turn(
                    speaker=speaker,
                    start=float(i * 5),
                    end=float(i * 5 + 4),
                    text=text.strip(),
                ))

        result = PipelineResult(audio_path=Path("text_input"))
        result.turns = turns

        if not self.skip_llm:
            result.cleaned = self.cleaner.clean(turns)
        else:
            result.cleaned = CleanedTranscript(raw_turns=turns)

        result.score = self.scorer.score(turns, result.cleaned)

        if print_report:
            self.reporter.print_console(result.score, "텍스트 입력")

        return result
