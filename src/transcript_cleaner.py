"""
src/transcript_cleaner.py — sLLM 기반 전사 정리 모듈
======================================================
Gemma-3 4B(로컬) 또는 설정된 sLLM으로 회의 전사본을 정리합니다.
  - 필러(어, 음, 그...) 제거
  - 주제 분류
  - 핵심 발언 추출
  - 액션 아이템 감지
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .transcription import Turn

logger = logging.getLogger(__name__)

# ── 프롬프트 템플릿 ──────────────────────────────────────────────────────────

_CLEAN_PROMPT = """\
당신은 회의 전사 정리 전문가입니다. 아래 회의 전사본을 읽고 다음 작업을 수행하세요.

[전사본]
{transcript}

[작업]
1. 불필요한 필러 워드(어, 음, 그, 저, 아 등)와 반복 표현을 제거하세요.
2. 각 화자의 핵심 발언을 한 줄로 요약하세요.
3. 전체 회의 주제를 3단어 이내로 추출하세요.
4. 결론 및 액션 아이템을 목록으로 정리하세요. 없으면 "없음"으로 표시하세요.

[출력 형식 - 반드시 아래 JSON 형식으로만 응답하세요]
{{
  "topic": "<회의 주제>",
  "cleaned_turns": [
    {{"speaker": "<화자ID>", "summary": "<핵심 발언 요약>"}},
    ...
  ],
  "action_items": ["<액션1>", "<액션2>"],
  "conclusion": "<회의 결론 한 문장>"
}}
"""

_FOCUS_PROMPT = """\
아래 회의 전사본에서 각 화자의 주제 집중도를 평가하세요.
주제: {topic}

[전사본]
{transcript}

각 화자가 주제와 관련된 발언을 한 비율(0.0~1.0)을 JSON으로 반환하세요.
{{"SPEAKER_00": 0.85, "SPEAKER_01": 0.72, ...}}
"""


@dataclass
class CleanedTranscript:
    """sLLM이 정리한 전사 결과."""
    topic: str = ""
    conclusion: str = ""
    action_items: List[str] = field(default_factory=list)
    cleaned_turns: List[dict] = field(default_factory=list)
    focus_scores: dict = field(default_factory=dict)  # {speaker: 0.0~1.0}
    raw_turns: List[Turn] = field(default_factory=list)


class TranscriptCleaner:
    """sLLM 기반 전사 정리기.

    Args:
        settings: 전역 설정 객체
    """

    def __init__(self, settings=None) -> None:
        from .config import get_settings
        self.settings = settings or get_settings()
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """sLLM 지연 로딩."""
        if self._model is not None:
            return

        model_id = self.settings.sllm_model_id
        logger.info("sLLM 로딩: %s", model_id)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=self.settings.hf_token or None,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.settings.hf_token or None,
            )
        except ImportError as e:
            raise ImportError("transformers 패키지가 필요합니다.") from e

        logger.info("sLLM 로딩 완료: %s", model_id)

    def _generate(self, prompt: str) -> str:
        """sLLM으로 텍스트 생성."""
        self._load_model()

        import torch

        messages = [{"role": "user", "content": prompt}]
        # chat template 적용 (Gemma, Phi 등 공통)
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.settings.sllm_max_new_tokens,
                temperature=self.settings.sllm_temperature,
                do_sample=self.settings.sllm_temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # 입력 부분 제거
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _turns_to_text(self, turns: List[Turn]) -> str:
        """Turn 리스트를 텍스트 전사본으로 변환."""
        lines = []
        for t in turns:
            lines.append(f"[{t.speaker}] ({t.start:.1f}s-{t.end:.1f}s) {t.text}")
        return "\n".join(lines)

    def _parse_json_safe(self, text: str) -> Optional[dict]:
        """LLM 출력에서 JSON 블록 추출 (파싱 실패 시 None 반환)."""
        import json
        # 코드 블록 제거
        text = re.sub(r"```json\s*|\s*```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # JSON 부분만 추출 시도
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.warning("sLLM JSON 파싱 실패. 원문: %s", text[:200])
        return None

    def clean(self, turns: List[Turn]) -> CleanedTranscript:
        """회의 전사 정리 및 구조화.

        Args:
            turns: 화자별 발언 Turn 리스트

        Returns:
            CleanedTranscript — 정리된 전사, 주제, 결론, 액션 아이템 포함
        """
        result = CleanedTranscript(raw_turns=turns)

        if not turns:
            logger.warning("전사 데이터가 비어있습니다.")
            return result

        transcript_text = self._turns_to_text(turns)

        # 1단계: 전사 정리 + 주제/결론/액션 추출
        logger.info("sLLM: 전사 정리 중...")
        prompt = _CLEAN_PROMPT.format(transcript=transcript_text)
        raw_output = self._generate(prompt)

        parsed = self._parse_json_safe(raw_output)
        if parsed:
            result.topic = parsed.get("topic", "")
            result.conclusion = parsed.get("conclusion", "")
            result.action_items = parsed.get("action_items", [])
            result.cleaned_turns = parsed.get("cleaned_turns", [])
        else:
            # 파싱 실패 시 원본 유지
            result.topic = "파싱 실패"
            result.cleaned_turns = [
                {"speaker": t.speaker, "summary": t.text[:100]}
                for t in turns
            ]

        # 2단계: 화자별 주제 집중도 평가
        if result.topic and result.topic != "파싱 실패":
            logger.info("sLLM: 주제 집중도 평가 중...")
            focus_prompt = _FOCUS_PROMPT.format(
                topic=result.topic,
                transcript=transcript_text,
            )
            focus_output = self._generate(focus_prompt)
            focus_parsed = self._parse_json_safe(focus_output)
            if focus_parsed and isinstance(focus_parsed, dict):
                # 0~1 범위로 클램핑
                result.focus_scores = {
                    k: max(0.0, min(1.0, float(v)))
                    for k, v in focus_parsed.items()
                    if isinstance(v, (int, float))
                }

        logger.info("전사 정리 완료 — 주제: %s / 액션: %d개",
                    result.topic, len(result.action_items))
        return result
