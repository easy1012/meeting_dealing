"""
src/scorer.py — 회의 전투력 스코어링 엔진
==========================================
화자별 5개 지표를 계산해 '회의 전투력' 점수를 산출합니다.

지표 및 가중치:
  1. 발언 참여율 (20%) — 전체 발언 시간 대비 비율
  2. 발언 집중도 (25%) — 주제 이탈 없는 발언 비율 (sLLM 평가)
  3. 상호작용 점수 (20%) — 다른 화자 의견에 반응/질문 빈도
  4. 결론 기여도 (25%) — 액션 아이템·결론 발언 포함 여부
  5. 발언 효율성 (10%) — 불필요한 반복·필러 비율 역산
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .transcript_cleaner import CleanedTranscript
from .transcription import Turn

logger = logging.getLogger(__name__)

# ── 가중치 ───────────────────────────────────────────────────────────────────
WEIGHTS = {
    "participation": 0.20,
    "focus":         0.25,
    "interaction":   0.20,
    "contribution":  0.25,
    "efficiency":    0.10,
}

# 필러 워드 패턴 (한국어)
_FILLER_PATTERN = re.compile(
    r"\b(어+|음+|그+|저+|아+|뭐+|그냥|이제|사실|솔직히|뭐랄까|어떻게 보면)\b",
    re.IGNORECASE,
)

# 상호작용 패턴 (질문, 동의, 반응)
_INTERACTION_PATTERN = re.compile(
    r"(맞죠|그렇죠|그런가요|어떻게 생각|의견|질문|맞습니까|어떤가요|"
    r"동의|반대|좋은 의견|좋아요|그거|그 부분|\?)",
    re.IGNORECASE,
)

# 결론/액션 키워드
_CONCLUSION_PATTERN = re.compile(
    r"(결론|정리|액션|해야|해줘|해주세요|담당|기한|마감|다음 주|다음 번|"
    r"진행|처리|검토|확인|완료|보고)",
    re.IGNORECASE,
)


@dataclass
class SpeakerScore:
    """화자 한 명의 스코어 상세."""
    speaker: str
    participation: float = 0.0   # 발언 참여율
    focus: float = 0.0           # 발언 집중도
    interaction: float = 0.0     # 상호작용 점수
    contribution: float = 0.0    # 결론 기여도
    efficiency: float = 0.0      # 발언 효율성
    total: float = 0.0           # 가중 합산 점수 (0~100)
    rank: int = 0                # 전투력 순위

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "scores": {
                "participation": round(self.participation * 100, 1),
                "focus":         round(self.focus * 100, 1),
                "interaction":   round(self.interaction * 100, 1),
                "contribution":  round(self.contribution * 100, 1),
                "efficiency":    round(self.efficiency * 100, 1),
            },
            "total": round(self.total, 1),
            "rank": self.rank,
        }


@dataclass
class MeetingScore:
    """회의 전체 스코어 결과."""
    speaker_scores: List[SpeakerScore] = field(default_factory=list)
    meeting_efficiency: float = 0.0   # 회의 전체 효율 점수
    topic: str = ""
    conclusion: str = ""
    action_items: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "topic": self.topic,
            "conclusion": self.conclusion,
            "action_items": self.action_items,
            "meeting_efficiency": round(self.meeting_efficiency, 1),
            "speakers": [s.as_dict() for s in self.speaker_scores],
        }


class MeetingScorer:
    """회의 전투력 스코어링 엔진."""

    def score(
        self,
        turns: List[Turn],
        cleaned: CleanedTranscript,
    ) -> MeetingScore:
        """화자별 전투력 점수 계산.

        Args:
            turns: 원본 화자-발언 Turn 리스트
            cleaned: sLLM이 정리한 전사 결과

        Returns:
            MeetingScore — 화자별 점수 + 회의 전체 점수
        """
        speakers = list({t.speaker for t in turns})
        result = MeetingScore(
            topic=cleaned.topic,
            conclusion=cleaned.conclusion,
            action_items=cleaned.action_items,
        )

        p_scores = self._participation(turns, speakers)
        f_scores = self._focus(turns, speakers, cleaned)
        i_scores = self._interaction(turns, speakers)
        c_scores = self._contribution(turns, speakers)
        e_scores = self._efficiency(turns, speakers)

        speaker_scores = []
        for sp in speakers:
            p = p_scores.get(sp, 0.0)
            f = f_scores.get(sp, 0.0)
            i = i_scores.get(sp, 0.0)
            c = c_scores.get(sp, 0.0)
            e = e_scores.get(sp, 0.0)

            total = (
                p * WEIGHTS["participation"] +
                f * WEIGHTS["focus"] +
                i * WEIGHTS["interaction"] +
                c * WEIGHTS["contribution"] +
                e * WEIGHTS["efficiency"]
            ) * 100

            speaker_scores.append(SpeakerScore(
                speaker=sp,
                participation=p,
                focus=f,
                interaction=i,
                contribution=c,
                efficiency=e,
                total=total,
            ))

        # 순위 부여 (높을수록 1등)
        speaker_scores.sort(key=lambda s: s.total, reverse=True)
        for rank, ss in enumerate(speaker_scores, start=1):
            ss.rank = rank

        result.speaker_scores = speaker_scores

        # 회의 전체 효율: 화자 평균 집중도 × 발언 효율
        if speaker_scores:
            avg_focus = sum(s.focus for s in speaker_scores) / len(speaker_scores)
            avg_eff = sum(s.efficiency for s in speaker_scores) / len(speaker_scores)
            result.meeting_efficiency = (avg_focus * 0.6 + avg_eff * 0.4) * 100

        return result

    # ── 지표 계산 메서드 ─────────────────────────────────────────────────────

    def _participation(
        self, turns: List[Turn], speakers: List[str]
    ) -> Dict[str, float]:
        """발언 참여율: 각 화자의 발언 시간 / 전체 발언 시간."""
        total_time = sum(t.duration for t in turns) or 1.0
        speaker_time: Dict[str, float] = defaultdict(float)
        for t in turns:
            speaker_time[t.speaker] += t.duration

        return {sp: speaker_time[sp] / total_time for sp in speakers}

    def _focus(
        self,
        turns: List[Turn],
        speakers: List[str],
        cleaned: CleanedTranscript,
    ) -> Dict[str, float]:
        """발언 집중도: sLLM 평가 결과 사용, 없으면 결론 키워드 기반 추정."""
        if cleaned.focus_scores:
            # sLLM 평가 결과 사용 (화자 ID가 다를 수 있어 부분 매칭)
            scores = {}
            for sp in speakers:
                # 정확히 일치하거나 번호가 같은 키 탐색
                matched = cleaned.focus_scores.get(sp)
                if matched is None:
                    # SPEAKER_00 → 00 추출 후 매칭
                    for k, v in cleaned.focus_scores.items():
                        if sp.split("_")[-1] == k.split("_")[-1]:
                            matched = v
                            break
                scores[sp] = matched if matched is not None else 0.5
            return scores

        # sLLM 평가 없을 때 키워드 기반 휴리스틱
        logger.debug("집중도: sLLM 평가 없음 — 키워드 기반 추정 사용")
        focus_hits: Dict[str, int] = defaultdict(int)
        focus_total: Dict[str, int] = defaultdict(int)

        if cleaned.topic:
            topic_words = set(cleaned.topic.split())
        else:
            topic_words = set()

        for t in turns:
            focus_total[t.speaker] += 1
            words = set(t.text.split())
            if topic_words & words:
                focus_hits[t.speaker] += 1

        return {
            sp: focus_hits[sp] / max(focus_total[sp], 1)
            for sp in speakers
        }

    def _interaction(
        self, turns: List[Turn], speakers: List[str]
    ) -> Dict[str, float]:
        """상호작용 점수: 발언 중 질문/반응 패턴 포함 비율."""
        hits: Dict[str, int] = defaultdict(int)
        totals: Dict[str, int] = defaultdict(int)

        for t in turns:
            totals[t.speaker] += 1
            if _INTERACTION_PATTERN.search(t.text):
                hits[t.speaker] += 1

        return {
            sp: hits[sp] / max(totals[sp], 1)
            for sp in speakers
        }

    def _contribution(
        self, turns: List[Turn], speakers: List[str]
    ) -> Dict[str, float]:
        """결론 기여도: 결론/액션 키워드 포함 발언 비율."""
        hits: Dict[str, int] = defaultdict(int)
        totals: Dict[str, int] = defaultdict(int)

        for t in turns:
            totals[t.speaker] += 1
            if _CONCLUSION_PATTERN.search(t.text):
                hits[t.speaker] += 1

        return {
            sp: min(hits[sp] / max(totals[sp], 1) * 3, 1.0)  # 3배 증폭 후 클램핑
            for sp in speakers
        }

    def _efficiency(
        self, turns: List[Turn], speakers: List[str]
    ) -> Dict[str, float]:
        """발언 효율성: 필러 워드 비율의 역수 (1 - filler_ratio)."""
        filler_ratio: Dict[str, float] = {}

        for sp in speakers:
            sp_turns = [t for t in turns if t.speaker == sp]
            all_text = " ".join(t.text for t in sp_turns)
            words = all_text.split()
            if not words:
                filler_ratio[sp] = 0.0
                continue
            filler_count = sum(
                1 for w in words if _FILLER_PATTERN.match(w)
            )
            filler_ratio[sp] = filler_count / len(words)

        return {
            sp: max(0.0, 1.0 - filler_ratio.get(sp, 0.0) * 5)
            for sp in speakers
        }
