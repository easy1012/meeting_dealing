"""
main.py — 회의 전투력 측정 CLI 진입점
========================================
사용법:
  python main.py --audio rsc/data/meeting.wav
  python main.py --audio rsc/data/meeting.wav --max-speakers 4 --skip-llm
  python main.py --text  # 텍스트 입력 테스트 모드
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # 외부 라이브러리 로그 레벨 억제
    for noisy in ("transformers", "torch", "pyannote", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    import click

    @click.command()
    @click.option(
        "--audio", "-a",
        type=click.Path(exists=False),
        default=None,
        help="분석할 오디오 파일 경로 (wav/mp3)",
    )
    @click.option(
        "--min-speakers", type=int, default=None,
        help="최소 화자 수 힌트 (생략 시 자동 감지)",
    )
    @click.option(
        "--max-speakers", type=int, default=None,
        help="최대 화자 수 힌트 (생략 시 자동 감지)",
    )
    @click.option(
        "--skip-diarization", is_flag=True, default=False,
        help="화자 분리 건너뜀 (단일 화자 테스트)",
    )
    @click.option(
        "--skip-llm", is_flag=True, default=False,
        help="sLLM 정리 건너뜀 (규칙 기반 스코어링만)",
    )
    @click.option(
        "--no-save", is_flag=True, default=False,
        help="JSON 리포트 저장 안 함",
    )
    @click.option(
        "--text", is_flag=True, default=False,
        help="텍스트 전사본 직접 입력 테스트 모드",
    )
    @click.option(
        "--verbose", "-v", is_flag=True, default=False,
        help="상세 로그 출력",
    )
    def cli(
        audio, min_speakers, max_speakers,
        skip_diarization, skip_llm, no_save, text, verbose,
    ):
        """🏆 회의 전투력 측정 — Meeting Combat Power Analyzer"""
        setup_logging(verbose)

        from src.pipeline import MeetingPipeline

        pipeline = MeetingPipeline(
            skip_diarization=skip_diarization,
            skip_llm=skip_llm,
            save_json=not no_save,
        )

        if text:
            # 텍스트 입력 테스트 모드
            _demo_text_mode(pipeline)
            return

        if not audio:
            click.echo("❌ 오디오 파일을 지정하세요: --audio rsc/data/meeting.wav")
            click.echo("   또는 텍스트 테스트: --text")
            sys.exit(1)

        result = pipeline.run(
            audio_path=audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        if result.report_path:
            click.echo(f"\n💾 JSON 리포트 저장됨: {result.report_path}")
        click.echo(f"⏱ 소요 시간: {result.elapsed_sec:.1f}초")

    cli()


def _demo_text_mode(pipeline) -> None:
    """데모용 텍스트 입력 테스트."""
    print("\n[텍스트 테스트 모드] 아래 형식으로 입력하세요.")
    print("[SPEAKER_00] 발언 내용")
    print("[SPEAKER_01] 발언 내용")
    print("(입력 완료: 빈 줄 두 번)\n")

    lines = []
    empty_count = 0
    while True:
        line = input()
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
        else:
            empty_count = 0
            lines.append(line)

    if not lines:
        # 샘플 데모 실행
        print("\n[샘플 데모 실행]")
        sample = """[SPEAKER_00] 오늘 회의 주제는 Q3 마케팅 전략입니다. 어 우선 현황부터 살펴볼까요?
[SPEAKER_01] 네 맞습니다. 현재 전환율이 3.2%인데 목표는 5%입니다. 어떻게 개선할 수 있을까요?
[SPEAKER_00] SNS 광고 예산을 늘리는 방향이 효과적일 것 같습니다. 담당자를 정해야 할 것 같은데요.
[SPEAKER_02] 그... 제가 SNS 운영 맡겠습니다. 다음 주까지 계획서 드리겠습니다.
[SPEAKER_01] 좋습니다. 그럼 저는 데이터 분석을 맡겠습니다. 결론적으로 예산 30% 증액으로 가는 건가요?
[SPEAKER_00] 네, 결론은 예산 30% 증액, SNS 강화, 다음 주 금요일 중간 보고입니다."""
        pipeline.run_text_only(sample, print_report=True)
    else:
        pipeline.run_text_only("\n".join(lines), print_report=True)


if __name__ == "__main__":
    main()
