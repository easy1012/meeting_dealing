"""
app.py — 회의 전투력 측정 Streamlit 대시보드
=============================================
실행: streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── 프로젝트 루트 경로 추가 ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 페이지 설정 (반드시 첫 번째 st 호출) ─────────────────────────────────────
st.set_page_config(
    page_title="회의 전투력 측정기",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 로깅 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for noisy in ("transformers", "torch", "pyannote", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ── CSS 커스텀 스타일 ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;600;700;900&display=swap');

  html, body, [class*="css"] {
    font-family: 'Pretendard', 'Noto Sans KR', sans-serif !important;
  }

  /* 전체 배경 */
  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f172a 50%, #0d1117 100%);
    color: #e2e8f0;
  }

  /* 사이드바 */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1a1f2e 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
  }

  /* 헤더 영역 */
  .hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(168,85,247,0.1) 50%, rgba(236,72,153,0.1) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
  }
  .hero-banner h1 {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
  }
  .hero-banner p {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
  }

  /* 카드 공통 */
  .metric-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(8px);
  }
  .metric-card:hover {
    border-color: rgba(99,102,241,0.5);
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(99,102,241,0.2);
  }
  .metric-card .value {
    font-size: 2.5rem;
    font-weight: 900;
    line-height: 1;
  }
  .metric-card .label {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* 등급 뱃지 */
  .grade-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .grade-S { background: linear-gradient(90deg,#f59e0b,#fbbf24); color:#000; }
  .grade-A { background: linear-gradient(90deg,#6ee7b7,#34d399); color:#000; }
  .grade-B { background: linear-gradient(90deg,#60a5fa,#3b82f6); color:#fff; }
  .grade-C { background: linear-gradient(90deg,#fbbf24,#f59e0b); color:#000; }
  .grade-D { background: linear-gradient(90deg,#f87171,#ef4444); color:#fff; }

  /* 순위 카드 */
  .rank-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.95) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
  }
  .rank-card:hover {
    border-color: rgba(99,102,241,0.5);
    box-shadow: 0 8px 32px rgba(99,102,241,0.2);
  }
  .rank-card.rank-1 { border-left: 4px solid #fbbf24; }
  .rank-card.rank-2 { border-left: 4px solid #94a3b8; }
  .rank-card.rank-3 { border-left: 4px solid #b45309; }

  .rank-medal { font-size: 2.5rem; line-height: 1; }
  .rank-speaker { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; }
  .rank-score  { font-size: 2.2rem; font-weight: 900; }
  .rank-score.high   { background: linear-gradient(90deg,#a78bfa,#f472b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
  .rank-score.medium { color: #60a5fa; }
  .rank-score.low    { color: #94a3b8; }

  /* 진행 바 */
  .progress-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.3rem;
  }
  .progress-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    transition: width 0.8s ease;
  }

  /* 전사 타임라인 */
  .turn-block {
    background: rgba(30,41,59,0.6);
    border-left: 3px solid;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn { from{opacity:0;transform:translateX(-8px)} to{opacity:1;transform:translateX(0)} }

  .turn-speaker { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing:0.06em; }
  .turn-text    { font-size: 0.95rem; color: #cbd5e1; margin-top: 0.25rem; line-height: 1.5; }
  .turn-time    { font-size: 0.72rem; color: #475569; margin-top: 0.2rem; }

  /* 액션 아이템 */
  .action-item {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* 섹션 구분선 */
  .section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.4), transparent);
  }

  /* 스텝 진행 표시 */
  .step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    background: rgba(30,41,59,0.5);
    border-radius: 10px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
  }
  .step-done   { color: #34d399; }
  .step-active { color: #fbbf24; animation: pulse 1.5s infinite; }
  .step-wait   { color: #475569; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

  /* 업로드 영역 */
  [data-testid="stFileUploadDropzone"] {
    background: rgba(99,102,241,0.05) !important;
    border: 2px dashed rgba(99,102,241,0.35) !important;
    border-radius: 16px !important;
  }

  /* 버튼 */
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,0.4) !important;
  }

  /* Streamlit 기본 요소 다크 테마 보정 */
  .stTextArea textarea {
    background: rgba(30,41,59,0.8) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
  }
  .stSelectbox > div > div {
    background: rgba(30,41,59,0.8) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
  }
  [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
  .stRadio label { color: #94a3b8 !important; }
  hr { border-color: rgba(99,102,241,0.15) !important; }

  /* 숨기기 */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

SPEAKER_COLORS = [
    "#6366f1", "#f472b6", "#34d399", "#fbbf24",
    "#60a5fa", "#f87171", "#a78bfa", "#2dd4bf",
]

def _speaker_color(speaker: str) -> str:
    idx = int(re.sub(r"\D", "", speaker) or "0") % len(SPEAKER_COLORS)
    return SPEAKER_COLORS[idx]

def _grade(score: float) -> str:
    if score >= 85: return "S"
    elif score >= 70: return "A"
    elif score >= 55: return "B"
    elif score >= 40: return "C"
    else: return "D"

def _grade_badge(score: float) -> str:
    g = _grade(score)
    return f'<span class="grade-badge grade-{g}">{g}</span>'

def _score_color(score: float) -> str:
    if score >= 70: return "high"
    elif score >= 45: return "medium"
    else: return "low"

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

def _progress_html(value: float, color: str = "#6366f1") -> str:
    pct = min(max(value, 0), 100)
    return f"""
    <div class="progress-bar-wrap">
      <div class="progress-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>"""


# ── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ 분석 설정")
    st.markdown("---")

    input_mode = st.radio(
        "입력 방식",
        ["🎤 오디오 파일", "📝 텍스트 직접 입력"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**화자 분리 설정**")
    skip_diarization = st.toggle("화자 분리 건너뜀", value=False,
        help="오디오를 단일 화자로 처리합니다.")
    min_spk = st.number_input("최소 화자 수", min_value=1, max_value=10, value=2,
        disabled=skip_diarization)
    max_spk = st.number_input("최대 화자 수", min_value=1, max_value=10, value=6,
        disabled=skip_diarization)

    st.markdown("---")
    st.markdown("**모델 설정**")
    skip_llm = st.toggle("sLLM 건너뜀 (빠른 테스트)", value=False,
        help="LLM 없이 규칙 기반 스코어링만 실행합니다.")
    save_json = st.toggle("JSON 결과 저장", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#475569;font-size:0.8rem;line-height:1.6;">
    <b style="color:#6366f1;">파이프라인</b><br>
    1️⃣ 화자 분리 (pyannote)<br>
    2️⃣ 음성 인식 (Whisper)<br>
    3️⃣ 정리 (Gemma-3 4B)<br>
    4️⃣ 스코어링 & 리포트
    </div>
    """, unsafe_allow_html=True)


# ── 히어로 배너 ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <h1>⚔️ 회의 전투력 측정기</h1>
  <p>AI가 회의 음성을 분석해 화자별 기여도, 집중도, 상호작용을 실시간으로 평가합니다</p>
</div>
""", unsafe_allow_html=True)


# ── 입력 영역 ─────────────────────────────────────────────────────────────────

audio_file = None
text_input = None

if "🎤" in input_mode:
    st.markdown('<div class="section-header">🎤 오디오 업로드</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader(
        "회의 녹음 파일을 업로드하세요",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="wav / mp3 / m4a / flac / ogg 지원",
        label_visibility="collapsed",
    )
    if audio_file:
        st.audio(audio_file)
        col1, col2 = st.columns(2)
        col1.metric("파일명", audio_file.name)
        col2.metric("크기", f"{audio_file.size / 1024:.1f} KB")

else:
    st.markdown('<div class="section-header">📝 텍스트 입력</div>', unsafe_allow_html=True)

    SAMPLE = """[SPEAKER_00] 오늘 회의 주제는 Q3 마케팅 전략입니다. 어 우선 현황부터 살펴볼까요?
[SPEAKER_01] 네 맞습니다. 현재 전환율이 3.2%인데 목표는 5%입니다. 어떻게 개선할 수 있을까요?
[SPEAKER_00] SNS 광고 예산을 늘리는 방향이 효과적일 것 같습니다. 담당자를 정해야 할 것 같은데요.
[SPEAKER_02] 제가 SNS 운영 맡겠습니다. 다음 주까지 계획서 드리겠습니다.
[SPEAKER_01] 좋습니다. 그럼 저는 데이터 분석을 맡겠습니다. 어떻게 생각하세요?
[SPEAKER_00] 결론은 예산 30% 증액, SNS 강화, 다음 주 금요일 중간 보고입니다."""

    use_sample = st.checkbox("샘플 데이터 사용", value=True)
    text_input = st.text_area(
        "전사본 입력 ([SPEAKER_00] 발언내용 형식)",
        value=SAMPLE if use_sample else "",
        height=220,
        placeholder="[SPEAKER_00] 발언 내용\n[SPEAKER_01] 발언 내용",
        label_visibility="collapsed",
    )

# ── 분석 버튼 ─────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("⚔️ 전투력 측정 시작", use_container_width=True)


# ── 분석 실행 ─────────────────────────────────────────────────────────────────

if run_btn:
    has_input = (audio_file is not None) if "🎤" in input_mode else bool(text_input and text_input.strip())

    if not has_input:
        st.error("⚠️ 분석할 데이터를 먼저 입력하세요.")
        st.stop()

    # 진행 상황 UI
    st.markdown("---")
    st.markdown('<div class="section-header">🔄 분석 진행 중</div>', unsafe_allow_html=True)

    step_container = st.empty()
    status_bar = st.progress(0, text="초기화 중...")

    def update_steps(done: list[int], active: int | None, total: int = 4):
        steps = [
            ("🎙️", "화자 분리 (pyannote)"),
            ("📝", "음성 인식 (Whisper)"),
            ("🧠", "sLLM 전사 정리 (Gemma-3)"),
            ("📊", "스코어링 & 리포트"),
        ]
        html = ""
        for i, (icon, label) in enumerate(steps, start=1):
            if i in done:
                cls, symbol = "step-done", "✅"
            elif i == active:
                cls, symbol = "step-active", "⏳"
            else:
                cls, symbol = "step-wait", "○"
            html += f'<div class="step-indicator"><span class="{cls}">{symbol}</span> {icon} {label}</div>'
        step_container.markdown(html, unsafe_allow_html=True)

    try:
        from src.pipeline import MeetingPipeline

        pipeline = MeetingPipeline(
            skip_diarization=skip_diarization or ("📝" in input_mode),
            skip_llm=skip_llm,
            save_json=save_json,
        )

        result = None

        if "🎤" in input_mode:
            # 오디오 파일 → 임시 경로 저장 후 실행
            # 경로 탐색 방지: 허용된 temp 경로만 사용
            update_steps([], 1)
            status_bar.progress(5, "화자 분리 중...")

            suffix = Path(audio_file.name).suffix.lower()
            allowed_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
            if suffix not in allowed_exts:
                st.error(f"허용되지 않은 파일 형식입니다: {suffix}")
                st.stop()

            # 오디오를 rsc/data/에 임시 저장 (경로 샌드박스 유지)
            data_dir = _ROOT / "rsc" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(audio_file.name).stem)
            tmp_path = data_dir / f"_upload_{safe_name}{suffix}"

            with open(tmp_path, "wb") as f:
                f.write(audio_file.read())

            update_steps([1], 2)
            status_bar.progress(25, "음성 인식(STT) 중...")

            update_steps([1, 2], 3)
            status_bar.progress(55, "sLLM 정리 중...")

            result = pipeline.run(
                audio_path=tmp_path,
                min_speakers=int(min_spk),
                max_speakers=int(max_spk),
                print_report=False,
            )
            # 임시 파일 정리
            try: tmp_path.unlink()
            except: pass

        else:
            # 텍스트 전사본 모드
            update_steps([], 1); status_bar.progress(20, "텍스트 파싱 중...")
            update_steps([1, 2], 3); status_bar.progress(50, "sLLM 정리 중...")
            result = pipeline.run_text_only(text_input, print_report=False)

        update_steps([1, 2, 3, 4], None); status_bar.progress(100, "완료!")

    except Exception as e:
        st.error(f"❌ 분석 중 오류가 발생했습니다:\n```\n{e}\n```")
        st.info("💡 모델이 설치되어 있지 않으면 `pip install -r requirements.txt` 후 재시도하세요.")
        st.stop()

    score = result.score
    if not score:
        st.warning("분석 결과가 없습니다.")
        st.stop()

    # ── 결과 저장 (세션) ─────────────────────────────────────────────────────
    st.session_state["last_result"] = result
    st.session_state["last_score"] = score

    st.markdown("---")
    st.success(f"✅ 분석 완료! (소요: {result.elapsed_sec:.1f}초)")


# ── 결과 표시 ─────────────────────────────────────────────────────────────────

if "last_score" not in st.session_state:
    # 결과 없을 때 안내
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#334155;">
      <div style="font-size:5rem;">⚔️</div>
      <div style="font-size:1.3rem;font-weight:600;margin-top:1rem;color:#64748b;">
        오디오 파일 또는 텍스트를 입력하고 분석을 시작하세요
      </div>
      <div style="font-size:0.9rem;margin-top:0.5rem;color:#475569;">
        AI가 화자 분리 → 음성 인식 → sLLM 정리 → 전투력 점수 순서로 분석합니다
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

score = st.session_state["last_score"]
result = st.session_state["last_result"]

# ── 섹션 1: 회의 개요 ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 회의 개요</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    eff = score.meeting_efficiency
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">회의 전체 효율</div>
      <div class="value" style="background:linear-gradient(90deg,#a78bfa,#f472b6);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;">{eff:.1f}</div>
      <div style="margin-top:0.5rem;">{_grade_badge(eff)}</div>
      {_progress_html(eff)}
    </div>
    """, unsafe_allow_html=True)

with col2:
    n_spk = len(score.speaker_scores)
    n_actions = len(score.action_items)
    st.markdown(f"""
    <div class="metric-card">
      <div style="display:flex;justify-content:space-around;align-items:center;gap:1rem;">
        <div>
          <div class="value" style="color:#60a5fa;">{n_spk}</div>
          <div class="label">참여 화자</div>
        </div>
        <div style="width:1px;height:60px;background:rgba(255,255,255,0.1);"></div>
        <div>
          <div class="value" style="color:#34d399;">{n_actions}</div>
          <div class="label">액션 아이템</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="height:100%;">
      <div class="label">분석 시간</div>
      <div class="value" style="color:#fbbf24;font-size:1.8rem;">{result.elapsed_sec:.1f}s</div>
    </div>
    """, unsafe_allow_html=True)

# 주제 / 결론
if score.topic or score.conclusion:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if score.topic:
        c1.markdown(f"""
        <div class="metric-card" style="text-align:left;">
          <div class="label">📋 회의 주제</div>
          <div style="font-size:1.2rem;font-weight:700;color:#a78bfa;margin-top:0.4rem;">
            {score.topic}
          </div>
        </div>
        """, unsafe_allow_html=True)
    if score.conclusion:
        c2.markdown(f"""
        <div class="metric-card" style="text-align:left;">
          <div class="label">✅ 결론</div>
          <div style="font-size:1.05rem;font-weight:600;color:#34d399;margin-top:0.4rem;">
            {score.conclusion}
          </div>
        </div>
        """, unsafe_allow_html=True)

# 액션 아이템
if score.action_items:
    st.markdown('<div class="section-header" style="margin-top:1.5rem;">🎯 액션 아이템</div>', unsafe_allow_html=True)
    action_html = ""
    for item in score.action_items:
        action_html += f'<div class="action-item">⚡ <span style="color:#e2e8f0;">{item}</span></div>'
    st.markdown(action_html, unsafe_allow_html=True)


# ── 섹션 2: 화자 전투력 순위 ─────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:2rem;">🏆 화자 전투력 순위</div>', unsafe_allow_html=True)

METRIC_LABELS = {
    "participation": ("참여율", "20%"),
    "focus":         ("집중도", "25%"),
    "interaction":   ("상호작용", "20%"),
    "contribution":  ("결론기여", "25%"),
    "efficiency":    ("효율성", "10%"),
}

for ss in score.speaker_scores:
    rank_cls = f"rank-{ss.rank}" if ss.rank <= 3 else ""
    medal = MEDALS.get(ss.rank, f"#{ss.rank}")
    sc_cls = _score_color(ss.total)
    color = _speaker_color(ss.speaker)
    d = ss.as_dict()
    scores_d = d["scores"]

    # 지표 바 HTML
    bars_html = ""
    for key, (label, weight) in METRIC_LABELS.items():
        val = scores_d.get(key, 0)
        bars_html += f"""
        <div style="margin-bottom:0.5rem;">
          <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#64748b;margin-bottom:2px;">
            <span>{label} <span style="color:#475569;">({weight})</span></span>
            <span style="color:#94a3b8;font-weight:600;">{val:.1f}</span>
          </div>
          {_progress_html(val, color)}
        </div>"""

    st.markdown(f"""
    <div class="rank-card {rank_cls}">
      <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1.2rem;">
        <div class="rank-medal">{medal}</div>
        <div style="flex:1;">
          <div class="rank-speaker" style="color:{color};">{ss.speaker}</div>
          <div style="font-size:0.8rem;color:#475569;">순위 #{ss.rank}</div>
        </div>
        <div>
          <div class="rank-score {sc_cls}">{ss.total:.1f}</div>
          <div style="text-align:right;margin-top:0.2rem;">{_grade_badge(ss.total)}</div>
        </div>
      </div>
      {bars_html}
    </div>
    """, unsafe_allow_html=True)


# ── 섹션 3: 전사 타임라인 ─────────────────────────────────────────────────────
if result.turns:
    st.markdown('<div class="section-header" style="margin-top:2rem;">💬 발언 타임라인</div>', unsafe_allow_html=True)

    with st.expander("전체 전사본 보기", expanded=False):
        timeline_html = ""
        for turn in result.turns:
            color = _speaker_color(turn.speaker)
            timeline_html += f"""
            <div class="turn-block" style="border-color:{color};">
              <div class="turn-speaker" style="color:{color};">{turn.speaker}</div>
              <div class="turn-text">{turn.text}</div>
              <div class="turn-time">⏱ {turn.start:.1f}s — {turn.end:.1f}s</div>
            </div>"""
        st.markdown(timeline_html, unsafe_allow_html=True)


# ── 섹션 4: 레이더 차트 + 참여율 파이 ────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:2rem;">📈 시각화</div>', unsafe_allow_html=True)

try:
    import plotly.graph_objects as go
    import plotly.express as px

    chart_col1, chart_col2 = st.columns(2)

    # 레이더 차트
    with chart_col1:
        categories = ["참여율", "집중도", "상호작용", "결론기여", "효율성"]
        fig_radar = go.Figure()

        for ss in score.speaker_scores:
            d = ss.as_dict()["scores"]
            vals = [
                d["participation"], d["focus"], d["interaction"],
                d["contribution"], d["efficiency"]
            ]
            # 레이더는 닫혀야 함
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=ss.speaker,
                line_color=_speaker_color(ss.speaker),
                fillcolor=_speaker_color(ss.speaker).replace("#", "rgba(") + ",0.12)",
                opacity=0.85,
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="rgba(99,102,241,0.15)",
                                tickcolor="#475569", tickfont=dict(color="#64748b", size=10)),
                angularaxis=dict(gridcolor="rgba(99,102,241,0.2)",
                                 linecolor="rgba(99,102,241,0.3)",
                                 tickfont=dict(color="#94a3b8", size=11)),
                bgcolor="rgba(15,23,42,0)",
            ),
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(15,23,42,0)",
            plot_bgcolor="rgba(15,23,42,0)",
            title=dict(text="화자별 역량 레이더", font=dict(color="#a78bfa", size=14)),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 발언 시간 도넛 차트
    with chart_col2:
        from collections import defaultdict
        spk_time: dict = defaultdict(float)
        for t in result.turns:
            spk_time[t.speaker] += t.duration

        labels = list(spk_time.keys())
        values = [spk_time[l] for l in labels]
        colors_pie = [_speaker_color(l) for l in labels]

        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors_pie,
                        line=dict(color="#0f172a", width=3)),
            textinfo="label+percent",
            textfont=dict(color="#e2e8f0", size=12),
            hovertemplate="<b>%{label}</b><br>발언 시간: %{value:.1f}s<br>비율: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            title=dict(text="화자별 발언 시간 비율", font=dict(color="#a78bfa", size=14)),
            paper_bgcolor="rgba(15,23,42,0)",
            plot_bgcolor="rgba(15,23,42,0)",
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[dict(
                text=f"<b>{len(labels)}</b><br>화자",
                x=0.5, y=0.5, font_size=16, showarrow=False,
                font=dict(color="#e2e8f0"),
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

except ImportError:
    st.info("📊 차트를 보려면 `pip install plotly` 를 실행하세요.")


# ── 섹션 5: JSON 다운로드 ─────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:2rem;">💾 결과 내보내기</div>', unsafe_allow_html=True)

json_data = score.as_dict()
json_data["meta"] = {
    "generated_at": datetime.now().isoformat(),
    "elapsed_sec": result.elapsed_sec,
}

dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    st.download_button(
        label="📥 JSON 다운로드",
        data=json.dumps(json_data, ensure_ascii=False, indent=2),
        file_name=f"meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

with dl_col2:
    # 텍스트 요약 다운로드
    summary_lines = [
        f"# 회의 전투력 측정 결과",
        f"생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## 회의 개요",
        f"- 주제: {score.topic}",
        f"- 결론: {score.conclusion}",
        f"- 회의 효율: {score.meeting_efficiency:.1f}점 ({_grade(score.meeting_efficiency)}등급)",
        f"",
        f"## 액션 아이템",
    ] + [f"- {a}" for a in score.action_items] + [
        f"",
        f"## 화자 전투력 순위",
    ]
    for ss in score.speaker_scores:
        d = ss.as_dict()
        summary_lines.append(
            f"{MEDALS.get(ss.rank,'#'+str(ss.rank))} {ss.speaker}: {ss.total:.1f}점 ({_grade(ss.total)}등급)"
        )

    st.download_button(
        label="📄 텍스트 요약 다운로드",
        data="\n".join(summary_lines),
        file_name=f"meeting_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

if result.report_path:
    st.caption(f"💾 JSON 자동 저장됨: `{result.report_path}`")
