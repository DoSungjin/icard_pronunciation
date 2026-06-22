"""
icard · 발음 코치 (Streamlit 화면)
================================================================
영어/한국어 발음 평가를 한 앱에서. 한국어 모드는 사전 DB와 연동된다.

실행:
  pip install -r requirements.txt
  python dictionary_db.py build 사전1.tsv 사전2.tsv 사전3.tsv   # 한국어 모드 전 1회
  streamlit run pronunciation_app.py

필요: Streamlit 1.40+ (st.audio_input), 같은 폴더에 pronunciation_eval.py,
      pronunciation_ko.py, dictionary_db.py
"""

import io
import math
import os

import numpy as np
import streamlit as st

from pronunciation_eval import (
    PhonemeRecognizer, reference_phonemes, align, score, SAMPLE_RATE, MODEL_ID,
)
from pronunciation_ko import korean_reference_phonemes, KO_MODEL, KO_MODEL_L2
import dictionary_db as dic

st.set_page_config(page_title="icard · 발음 코치", page_icon="🎙️", layout="centered")

# ---------------------------------------------------------------------------
# 디자인 토큰 (CSS는 일반 문자열 — f-string 아님)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
:root{ --ink:#1b1e2b; --muted:#6b7280; --line:#e7e8ef;
  --brand:#4338ca; --ok:#0e9f6e; --sub:#d97706; --del:#e11d48; --ins:#7c3aed; }
html, body, [class*="css"]{ color:var(--ink); }
h1,h2,h3{ font-family:'Space Grotesk',sans-serif !important; }
.hero .logo{ font-family:'Space Grotesk'; font-weight:700; font-size:1.6rem; letter-spacing:-.02em; }
.hero .logo .dot{ color:var(--brand); }
.hero .sub{ color:var(--muted); font-size:.95rem; margin-top:.1rem; }
.wordcard{ background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:1.4rem 1.5rem; text-align:center; margin:.4rem 0 1rem; }
.wordcard .label{ color:var(--muted); font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; }
.wordcard .word{ font-family:'Space Grotesk'; font-weight:700; font-size:2.6rem; line-height:1.1; margin:.1rem 0 .2rem; }
.wordcard .pron{ color:var(--muted); font-family:'JetBrains Mono',monospace; margin-bottom:.5rem; }
.tiles{ display:flex; flex-wrap:wrap; gap:.4rem; justify-content:center; }
.tile{ font-family:'JetBrains Mono',monospace; font-weight:700; min-width:2.4rem;
  padding:.5rem .6rem; border-radius:10px; text-align:center; border:1px solid var(--line); background:#fafafe; }
.tile .said{ display:block; font-size:.62rem; font-weight:500; margin-top:.15rem; opacity:.85; }
.tile.ref{ color:var(--muted); }
.tile.ok{ background:#e7f7f1; border-color:#bfeada; color:#087a55; }
.tile.sub{ background:#fdf3e6; border-color:#f3d9ad; color:#a85d06; }
.tile.del{ background:#fdeaee; border-color:#f6c3ce; color:#b3173a; text-decoration:line-through; }
.chip.ins{ font-family:'JetBrains Mono',monospace; font-weight:700; font-size:.75rem; padding:.45rem .5rem;
  border-radius:10px; background:#f1ebfd; border:1px solid #d9c8f7; color:#5b21b6; align-self:center; }
.verdict{ font-family:'Space Grotesk'; font-weight:600; font-size:1.05rem; margin:.2rem 0 .6rem; }
.legend{ color:var(--muted); font-size:.78rem; margin-top:.6rem; display:flex; gap:1rem; flex-wrap:wrap; justify-content:center; }
.sw{ display:inline-block; width:.7rem; height:.7rem; border-radius:3px; vertical-align:middle; margin-right:.25rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="logo">icard <span class="dot">·</span> 발음 코치</div>
  <div class="sub">단어를 소리 내어 읽으면, 바로 코칭해 드려요.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="음소 인식 모델 준비 중...")
def get_recognizer(mid: str) -> PhonemeRecognizer:
    return PhonemeRecognizer(mid)


def bytes_to_audio(data: bytes) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sr
        new_len = int(math.floor(len(audio) * ratio))
        audio = np.interp(np.linspace(0, len(audio), new_len, endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return audio


def render_reference(phs):
    tiles = "".join(f'<div class="tile ref">{p}</div>' for p in phs)
    return f'<div class="tiles">{tiles}</div>'


def render_result_tiles(alignment):
    cells = []
    for op, ref_ph, hyp_ph in alignment:
        if op == "match":
            cells.append(f'<div class="tile ok">{ref_ph}</div>')
        elif op == "sub":
            cells.append(f'<div class="tile sub">{ref_ph}<span class="said">들림: {hyp_ph}</span></div>')
        elif op == "del":
            cells.append(f'<div class="tile del">{ref_ph}<span class="said">빠짐</span></div>')
        elif op == "ins":
            cells.append(f'<div class="chip ins">+{hyp_ph}</div>')
    return f'<div class="tiles">{"".join(cells)}</div>'


LEGEND = """
<div class="legend">
  <span><span class="sw" style="background:#0e9f6e"></span>정확</span>
  <span><span class="sw" style="background:#d97706"></span>다른 소리로</span>
  <span><span class="sw" style="background:#e11d48"></span>빠뜨림</span>
  <span><span class="sw" style="background:#7c3aed"></span>추가됨</span>
</div>
"""


def evaluate(reference, audio, recognizer):
    hypothesis, conf = recognizer.recognize(audio)
    alignment = align(reference, hypothesis)
    result = score(alignment, conf)
    return hypothesis, alignment, result


def pick_focus(alignment):
    """가장 신경 쓸 음소 하나만 고른다 (대치 우선, 없으면 탈락)."""
    for op, ref_ph, hyp_ph in alignment:
        if op == "sub":
            return ("sub", ref_ph, hyp_ph)
    for op, ref_ph, hyp_ph in alignment:
        if op == "del":
            return ("del", ref_ph, None)
    return None


def render_focus_tile(focus):
    op, ref_ph, hyp_ph = focus
    if op == "sub":
        return (f'<div class="tiles"><div class="tile sub">{ref_ph}'
                f'<span class="said">들림: {hyp_ph}</span></div></div>')
    return (f'<div class="tiles"><div class="tile del">{ref_ph}'
            f'<span class="said">살짝 빠졌어요</span></div></div>')


def show_result_user(hypothesis, alignment):
    """학습자 화면: 칭찬 + 신경 쓸 소리 하나만."""
    if not hypothesis:
        st.warning("소리가 잘 안 들렸어요. 다시 한 번 또박또박 말해볼까요?")
        return
    focus = pick_focus(alignment)
    if focus is None:
        st.markdown('<div class="verdict">참 잘했어요! 🎉 완벽해요.</div>',
                    unsafe_allow_html=True)
        return
    st.markdown('<div class="verdict">참 잘했어요! 🎉</div>', unsafe_allow_html=True)
    st.markdown("이 소리만 조금 더 신경 써봐요 👇")
    st.markdown(render_focus_tile(focus), unsafe_allow_html=True)


def show_result_admin(reference, hypothesis, alignment, result):
    """관리자 화면: 정확한 데이터 전부."""
    acc = result["accuracy"]
    if not hypothesis:
        st.error("소리를 인식하지 못했습니다.")
        return
    verdict = ("훌륭해요! 거의 원어민 수준이에요." if acc >= 90
               else "좋아요. 표시된 음소만 더 다듬어볼까요?" if acc >= 60
               else "조금 더 연습해봐요. 어디서 어긋났는지 아래에 표시했어요.")
    st.markdown(f'<div class="verdict">{verdict}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    c1.metric("정확도", f"{acc}%", f"{result['correct']}/{result['total']} 음소")
    if result["confidence"] is not None:
        c2.metric("인식 확신도", result["confidence"])
    st.progress(min(int(acc), 100) / 100)
    st.markdown("###### 음소별 결과")
    st.markdown(render_result_tiles(alignment), unsafe_allow_html=True)
    st.markdown(LEGEND, unsafe_allow_html=True)
    with st.expander("자세히 보기"):
        st.write(f"**정답 발음** : `{' '.join(reference)}`")
        st.write(f"**내 발음** : `{' '.join(hypothesis) if hypothesis else '(없음)'}`")
        st.write(f"대치 {result['substitutions']} · 탈락 {result['deletions']} · 삽입 {result['insertions']}")


# ---------------------------------------------------------------------------
# 사이드바: 모드 선택
# ---------------------------------------------------------------------------
LANG_FIELDS = {
    "영어": ("en_equiv", "en_gloss"), "일본어": ("ja_equiv", "ja_gloss"),
    "중국어": ("zh_equiv", "zh_gloss"), "베트남어": ("vi_equiv", "vi_gloss"),
    "타이어": ("th_equiv", "th_gloss"), "인도네시아어": ("id_equiv", "id_gloss"),
    "러시아어": ("ru_equiv", "ru_gloss"), "몽골어": ("mn_equiv", "mn_gloss"),
    "아랍어": ("ar_equiv", "ar_gloss"), "프랑스어": ("fr_equiv", "fr_gloss"),
    "스페인어": ("es_equiv", "es_gloss"),
}

with st.sidebar:
    admin = st.toggle("관리자 모드", value=False,
                      help="정확도·음소 분석 등 상세 데이터를 봅니다.")
    mode = st.radio("학습 언어", ["영어", "한국어"], horizontal=True)
    if mode == "영어":
        model_id, use_ipa = MODEL_ID, False
        if admin:
            model_id = st.text_input("음소 인식 모델", value=MODEL_ID)
            use_ipa = st.toggle("모델이 IPA로 출력함", value=False)
    else:
        model_id, db_path = KO_MODEL, dic.DB_PATH
        if admin:
            model_id = st.selectbox("음소 인식 모델", [KO_MODEL, KO_MODEL_L2])
            db_path = st.text_input("사전 DB 경로", value=dic.DB_PATH)
        meaning_lang = st.selectbox("내 언어 (뜻 표시)", list(LANG_FIELDS.keys()))


# ===========================================================================
# 영어 모드
# ===========================================================================
if mode == "영어":
    # NGSL 레벨 모드 (english_words 테이블이 적재돼 있으면 활성화)
    ew_levels = []
    try:
        import english_words as ew
        ew_levels = ew.levels()
    except Exception:
        ew_levels = []

    WORD_BANK = {
        "기초 단어": ["apple", "water", "school", "friend", "beautiful"],
        "최소대립쌍 (R/L · B/V · 모음)": ["rice", "lice", "ban", "van", "sit", "seat", "light", "right"],
    }
    pick_opts = (["NGSL 레벨"] if ew_levels else []) + ["단어 은행", "직접 입력"]
    pick = st.radio("연습 모드", pick_opts, horizontal=True, label_visibility="collapsed")

    if pick == "NGSL 레벨":
        lv_label = st.selectbox("레벨", [f"레벨 {l} · {n}단어" for l, n in ew_levels])
        lv = int(lv_label.split()[1])
        words = ew.words_by_level(lv)
        opts = {w["word"]: w for w in words}
        sel = st.selectbox("단어", list(opts.keys()))
        chosen = opts[sel]
        word = chosen["word"]
        if chosen.get("definition"):
            st.caption(f'뜻: {chosen["definition"]}')
    elif pick == "단어 은행":
        cat = st.selectbox("카테고리", list(WORD_BANK.keys()))
        word = st.selectbox("단어", WORD_BANK[cat])
    else:
        word = st.text_input("연습할 단어", value="apple").strip()

    reference = None
    if word:
        try:
            reference = reference_phonemes(word, as_ipa=use_ipa)
            ref_html = render_reference(reference) if admin else ""
            st.markdown(f'<div class="wordcard"><div class="label">이렇게 발음해보세요</div>'
                        f'<div class="word">{word.lower()}</div>{ref_html}</div>',
                        unsafe_allow_html=True)
        except ValueError:
            st.warning(f"'{word}' 는 CMUdict 사전에 없어요. 다른 단어로 시도해보세요.")

    if reference:
        audio_value = st.audio_input("🎙️ 마이크로 단어를 발음하세요")
        if audio_value is not None:
            with st.spinner("발음 분석 중..."):
                rec = get_recognizer(model_id)
                hyp, al, res = evaluate(reference, bytes_to_audio(audio_value.getvalue()), rec)
            if admin:
                show_result_admin(reference, hyp, al, res)
            else:
                show_result_user(hyp, al)


# ===========================================================================
# 한국어 모드 (사전 연동)
# ===========================================================================
else:
    if not os.path.exists(db_path):
        st.info("사전 DB가 없습니다. 먼저 사전을 적재하세요:\n\n"
                "`python dictionary_db.py build 사전1.tsv 사전2.tsv 사전3.tsv`")
        st.stop()

    query = st.text_input("단어 검색 (한국어)", value="").strip()
    entry = None
    if query:
        results = dic.prefix_korean(query, db_path) or dic.search(query, db_path)
        if not results:
            st.warning("검색 결과가 없어요.")
        else:
            def _label(r):
                hn = f"({r.get('homonym_no')})" if r.get("homonym_no") else ""
                d = (r.get("definition") or "")[:18]
                return f"{r.get('headword','')}{hn} · {r.get('pos','')} · {d}"
            idx = st.selectbox("단어 선택", range(len(results)),
                               format_func=lambda i: _label(results[i]))
            entry = results[idx]

    if entry:
        pron = entry.get("pronunciation") or entry.get("headword")
        reference = korean_reference_phonemes(pronunciation=pron) if pron else []
        ref_html = render_reference(reference) if admin else ""
        st.markdown(f'<div class="wordcard"><div class="label">이렇게 발음해보세요</div>'
                    f'<div class="word">{entry.get("headword","")}</div>'
                    f'<div class="pron">[{entry.get("pronunciation","")}]</div>'
                    f'{ref_html}</div>', unsafe_allow_html=True)

        # 뜻: 한국어 정의 + 선택 언어 대역어
        st.markdown(f"**뜻풀이** · {entry.get('definition','')}")
        eq, gl = LANG_FIELDS[meaning_lang]
        if entry.get(eq):
            st.markdown(f"**{meaning_lang}** · {entry.get(eq,'')} — {entry.get(gl,'')}")
        if admin:
            with st.expander("모든 언어 보기"):
                for lang, (e, g) in LANG_FIELDS.items():
                    if entry.get(e):
                        st.write(f"- **{lang}**: {entry.get(e,'')} — {entry.get(g,'')}")

        if reference:
            audio_value = st.audio_input("🎙️ 마이크로 단어를 발음하세요")
            if audio_value is not None:
                with st.spinner("발음 분석 중..."):
                    rec = get_recognizer(model_id)
                    hyp, al, res = evaluate(reference, bytes_to_audio(audio_value.getvalue()), rec)
                if admin:
                    show_result_admin(reference, hyp, al, res)
                else:
                    show_result_user(hyp, al)
