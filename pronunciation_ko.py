"""
icard - 한국어 발음 평가 백엔드
================================================================
영어 엔진(pronunciation_eval.py)의 정렬·채점·음소 인식기를 그대로 재사용하고,
한국어 고유 부분인 "정답 발음 음소열 생성"만 추가한다.

정답 발음을 얻는 두 경로:
  1) 사전의 '발음' 컬럼이 있으면 그대로 사용 (이미 발음 규칙이 적용된 형태)
     예: "가ː" → ㄱ ㅏ      ← g2p 불필요, 가장 정확
  2) 표기만 있으면 g2pk로 발음 규칙(연음·경음화 등)을 적용한 뒤 자모 분해
     예: "맑았는데" → "말간는데" → ㅁ ㅏ ㄹ ㄱ ...

음향 모델 (서울대 SLP랩):
  slplab/wav2vec2-xls-r-300m_phone-mfa_korean        # 원어민 한국어 음소
  slplab/wav2vec2-xls-r_Korean_ASR_by_foreigners     # 외국인의 한국어(L2) — 학습자 평가에 적합

주의:
  음향 모델이 출력하는 음소 기호 집합과 아래 자모 표현이 일치해야 정렬이 맞다.
  pronunciation_eval.py 의 --inspect 로 모델 vocab을 먼저 확인하고,
  필요하면 JAMO_* 표현을 모델 기호에 맞춰 매핑하라. (영어의 ARPAbet vs IPA 문제와 동일)
"""

import re

# 영어 엔진에서 검증된 로직 재사용
from pronunciation_eval import align, score, PhonemeRecognizer  # noqa: F401

KO_MODEL = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"
KO_MODEL_L2 = "slplab/wav2vec2-xls-r_Korean_ASR_by_foreigners"

# 한글 자모 분해 테이블 (유니코드 한글 음절: U+AC00 ~ U+D7A3)
CHOSEONG = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
            "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNGSEONG = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
             "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONGSEONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
             "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
             "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

_HANGUL_BASE = 0xAC00


def hangul_to_jamo(text: str) -> list:
    """한글 문자열을 자모 음소 리스트로 분해. 한글이 아닌 문자는 무시."""
    out = []
    for ch in text:
        code = ord(ch)
        if _HANGUL_BASE <= code <= 0xD7A3:
            s = code - _HANGUL_BASE
            cho, jung, jong = s // 588, (s % 588) // 28, s % 28
            out.append(CHOSEONG[cho])
            out.append(JUNGSEONG[jung])
            if jong:
                out.append(JONGSEONG[jong])
    return out


def _clean_pron(pron: str) -> str:
    """사전 발음 표기 정리: 장음(ː), 공백, 부가기호 제거."""
    pron = pron.replace("ː", "").replace(":", "")
    pron = re.sub(r"[\s.\-()]", "", pron)
    return pron


def korean_reference_phonemes(word: str = None, pronunciation: str = None,
                              backend: str = "g2pk") -> list:
    """한국어 정답 음소열(자모) 생성.

    pronunciation (사전 '발음' 컬럼)이 있으면 그것을 우선 사용한다.
    없으면 word를 backend로 발음 변환 후 분해한다.
    """
    if pronunciation:
        return hangul_to_jamo(_clean_pron(pronunciation))

    if not word:
        raise ValueError("word 또는 pronunciation 중 하나는 필요합니다.")

    if backend == "none":
        # 발음 규칙 미적용 (표기 그대로 분해) — 테스트/폴백용
        return hangul_to_jamo(word)

    # 발음 규칙 적용: g2pk (또는 가벼운 대안 g2pkk)
    pron = _apply_g2p(word, backend)
    return hangul_to_jamo(pron)


_g2p_cache = {}


def _apply_g2p(word: str, backend: str) -> str:
    if backend in _g2p_cache:
        return _g2p_cache[backend](word)
    if backend == "g2pk":
        from g2pk import G2p
        g2p = G2p()
    elif backend == "g2pkk":
        from g2pkk import G2p   # mecab 불필요한 경량 포크
        g2p = G2p()
    else:
        raise ValueError(f"알 수 없는 backend: {backend}")
    _g2p_cache[backend] = g2p
    return g2p(word)


# ---------------------------------------------------------------------------
# 평가 (영어와 동일한 흐름)
# ---------------------------------------------------------------------------
def evaluate_korean(recognizer, audio, word=None, pronunciation=None,
                    backend="g2pk"):
    """오디오를 한국어 정답 발음과 비교해 채점 결과 반환."""
    reference = korean_reference_phonemes(word, pronunciation, backend)
    hypothesis, conf = recognizer.recognize(audio)
    alignment = align(reference, hypothesis)
    result = score(alignment, conf)
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "alignment": alignment,
        "result": result,
    }


if __name__ == "__main__":
    # 네트워크/마이크 없이 정답 생성 로직만 빠르게 확인
    print("가  ->", korean_reference_phonemes(pronunciation="가ː"))
    print("값  ->", korean_reference_phonemes(word="값", backend="none"))
    print("학교 ->", korean_reference_phonemes(word="학교", backend="none"),
          "(g2pk 적용 시 학꾜로 바뀜)")
