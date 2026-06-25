"""
src/report.py — 리포트 생성 모듈
==================================
회의 분석 결과를 콘솔(Rich 테이블)과 JSON 파일로 출력합니다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .scorer import MeetingScore

logger = logging.getLogger(__name__)

# ── 등급 매핑 ────────────────────────────────────────────────────────────────
def _grade(score: float) -> str:
    """점수 → 등급 문자열."""
    if score >= 85:
        return "S"
    elif score >= 70:
        return "A"
    elif score >= 55:
        return "B"
    elif score >= 40:
        return "C"
    else:
        return "D"


def _bar(value: float, width: int = 20) -> str:
    """0~100 값을 ASCII 바 차트로 표시."""
    filled = int(value / 100 * width)
    return "█" * filled + "░" * (width - filled)


class ReportGenerator:
    """회의 분석 리포트 생성기."""

    def __init__(self, settings=None) -> None:
        from .config import get_settings
        self.settings = settings or get_settings()

    def print_console(self, score: MeetingScore, audio_name: str = "회의") -> None:
        """Rich 라이브러리로 콘솔에 리포트 출력."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich import box
            self._print_rich(score, audio_name)
        except ImportError:
            # Rich 없으면 기본 출력
            self._print_plain(score, audio_name)

    def _print_rich(self, score: MeetingScore, audio_name: str) -> None:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box

        console = Console()

        # 헤더
        console.print()
        console.print(Panel(
            f"[bold cyan]🏆 회의 전투력 측정 리포트[/bold cyan]\n"
            f"[dim]파일: {audio_name} | 분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}[/dim]",
            box=box.DOUBLE_EDGE,
        ))

        # 회의 개요
        if score.topic:
            console.print(f"\n[bold yellow]📋 회의 주제:[/bold yellow] {score.topic}")
        if score.conclusion:
            console.print(f"[bold green]✅ 결론:[/bold green] {score.conclusion}")
        if score.action_items:
            console.print("[bold red]🎯 액션 아이템:[/bold red]")
            for item in score.action_items:
                console.print(f"  • {item}")

        console.print(
            f"\n[bold]회의 전체 효율:[/bold] "
            f"{_bar(score.meeting_efficiency)} "
            f"[bold]{score.meeting_efficiency:.1f}점[/bold] "
            f"({_grade(score.meeting_efficiency)} 등급)"
        )

        # 화자별 점수 테이블
        table = Table(
            title="\n🎤 화자별 전투력 점수",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("순위", justify="center", width=4)
        table.add_column("화자", width=12)
        table.add_column("참여율\n(20%)", justify="right")
        table.add_column("집중도\n(25%)", justify="right")
        table.add_column("상호작용\n(20%)", justify="right")
        table.add_column("결론기여\n(25%)", justify="right")
        table.add_column("효율성\n(10%)", justify="right")
        table.add_column("총점", justify="right", style="bold")
        table.add_column("등급", justify="center")

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for ss in score.speaker_scores:
            d = ss.as_dict()
            s = d["scores"]
            medal = medals.get(ss.rank, f"#{ss.rank}")
            grade = _grade(ss.total)
            grade_style = {
                "S": "bold yellow", "A": "bold green",
                "B": "green", "C": "yellow", "D": "red",
            }.get(grade, "white")

            table.add_row(
                medal,
                ss.speaker,
                f"{s['participation']:.1f}",
                f"{s['focus']:.1f}",
                f"{s['interaction']:.1f}",
                f"{s['contribution']:.1f}",
                f"{s['efficiency']:.1f}",
                f"[bold]{ss.total:.1f}[/bold]",
                f"[{grade_style}]{grade}[/{grade_style}]",
            )

        console.print(table)
        console.print()

    def _print_plain(self, score: MeetingScore, audio_name: str) -> None:
        """Rich 없을 때 기본 텍스트 출력."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  회의 전투력 측정 리포트 — {audio_name}")
        print(sep)
        print(f"  주제    : {score.topic}")
        print(f"  결론    : {score.conclusion}")
        print(f"  효율    : {score.meeting_efficiency:.1f}점 ({_grade(score.meeting_efficiency)} 등급)")
        if score.action_items:
            print(f"  액션    : {', '.join(score.action_items)}")
        print(f"\n{'화자':12} {'참여':>6} {'집중':>6} {'상호':>6} {'결론':>6} {'효율':>6} {'총점':>7} {'등급':>4}")
        print("-" * 60)
        for ss in score.speaker_scores:
            d = ss.as_dict()
            s = d["scores"]
            print(
                f"{ss.speaker:12} "
                f"{s['participation']:>6.1f} "
                f"{s['focus']:>6.1f} "
                f"{s['interaction']:>6.1f} "
                f"{s['contribution']:>6.1f} "
                f"{s['efficiency']:>6.1f} "
                f"{ss.total:>7.1f} "
                f"{_grade(ss.total):>4}"
            )
        print(sep + "\n")

    def save_json(
        self,
        score: MeetingScore,
        audio_name: str = "meeting",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """결과를 JSON 파일로 저장.

        Args:
            score: 스코어 결과
            audio_name: 오디오 파일명 (확장자 제외)
            output_dir: 저장 경로 (None이면 settings.output_dir 사용)

        Returns:
            저장된 JSON 파일 경로
        """
        out_dir = output_dir or self.settings.output_dir
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(audio_name).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{stem}_{ts}_report.json"

        data = score.as_dict()
        data["meta"] = {
            "audio": audio_name,
            "generated_at": datetime.now().isoformat(),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("JSON 리포트 저장: %s", out_path)
        return out_path
