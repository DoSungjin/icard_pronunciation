"""
icard - 발음 평가 프로토타입 (Pronunciation Evaluation)
================================================================
파이프라인:
  1) 마이크 녹음 (16kHz)
  2) wav2vec2 음소 인식  -> 사용자가 실제로 낸 음소 시퀀스
  3) CMUdict(ARPAbet)    -> 목표 단어의 '정답' 음소 시퀀스
  4) 시퀀스 정렬(edit distance) -> 음소별 대치/탈락/삽입 판정
  5) 채점 + 피드백 출력

설치:
  pip install torch transformers sounddevice soundfile numpy pronouncing

권장 모델 (한국인 영어 학습자용, 서울대 SLP랩):
  slplab/wav2vec2-large-robust-L2-english-phoneme-recognition

사용법:
  python pronunciation_eval.py            # 대화형 루프
  python pronunciation_eval.py --inspect  # 모델이 출력하는 음소 기호 확인
"""

import argparse
import sys
import re

import numpy as np

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODEL_ID = "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition"
# 대안(다국어 IPA 출력): "facebook/wav2vec2-lv-60-espeak-cv-ft"

SAMPLE_RATE = 16000          # wav2vec2 계열은 16kHz 고정
DEFAULT_RECORD_SECONDS = 3.0

# ARPAbet -> IPA 매핑 (모델이 IPA를 출력할 경우 기준 발음을 변환해 비교하기 위함)
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


# ---------------------------------------------------------------------------
# 1) 녹음
# ---------------------------------------------------------------------------
def record_audio(seconds: float = DEFAULT_RECORD_SECONDS,
                 samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """마이크에서 mono float32 오디오를 녹음해서 반환."""
    import sounddevice as sd
    print(f"  ● 녹음 중... ({seconds:.0f}초) 지금 말하세요!")
    audio = sd.rec(int(seconds * samplerate),
                   samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    print("  ✓ 녹음 완료")
    return audio.flatten()


def load_audio_file(path: str, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """WAV/오디오 파일을 16kHz mono float32로 로드 (테스트용)."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != samplerate:
        # 간단 리샘플 (의존성 최소화). 정밀 작업이면 librosa.resample 권장.
        import math
        ratio = samplerate / sr
        new_len = int(math.floor(len(audio) * ratio))
        audio = np.interp(
            np.linspace(0, len(audio), new_len, endpoint=False),
            np.arange(len(audio)), audio,
        ).astype("float32")
    return audio


# ---------------------------------------------------------------------------
# 2) 음소 인식 (wav2vec2)
# ---------------------------------------------------------------------------
class PhonemeRecognizer:
    def __init__(self, model_id: str = MODEL_ID):
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        print(f"[모델 로딩] {model_id} ...")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id)
        self.model.eval()
        print("[모델 로딩] 완료")

    def recognize(self, audio: np.ndarray):
        """오디오 -> (음소 리스트, 평균 confidence)."""
        torch = self.torch
        inputs = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits  # [1, T, V]

        # CTC argmax 디코딩
        pred_ids = torch.argmax(logits, dim=-1)
        decoded = self.processor.batch_decode(pred_ids)[0]

        # GOP 유사 신호: 프레임별 최대 posterior의 평균 (대략적 확신도)
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs.max(dim=-1).values.mean())

        phonemes = tokenize_phonemes(decoded)
        return phonemes, confidence

    def vocab(self):
        return self.processor.tokenizer.get_vocab()


def tokenize_phonemes(decoded: str) -> list:
    """디코딩 문자열을 음소 토큰 리스트로 변환.
    음소 모델은 보통 공백으로 음소를 구분한다."""
    decoded = decoded.strip()
    if " " in decoded:
        toks = decoded.split()
    else:
        # 공백 구분이 없으면 글자 단위로 (IPA 모델 대비)
        toks = list(decoded)
    return [normalize_phoneme(t) for t in toks if t.strip()]


def normalize_phoneme(p: str) -> str:
    """강세 숫자 제거, 소문자 통일 등 비교용 정규화."""
    p = p.strip()
    p = re.sub(r"\d+$", "", p)   # ARPAbet 강세 표기(AH0, IY1) 제거
    return p.lower()


# ---------------------------------------------------------------------------
# 3) 기준 발음 (CMUdict / ARPAbet)
# ---------------------------------------------------------------------------
def reference_phonemes(word: str, as_ipa: bool = False) -> list:
    """단어 -> 기준 음소 시퀀스. pronouncing(CMUdict) 사용."""
    import pronouncing
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        raise ValueError(f"'{word}' 단어가 CMUdict에 없습니다.")
    arpabet = phones[0].split()  # 첫 번째 발음 변형 사용
    if as_ipa:
        out = []
        for p in arpabet:
            base = re.sub(r"\d+$", "", p)
            out.append(ARPABET_TO_IPA.get(base, base.lower()))
        return out
    return [normalize_phoneme(p) for p in arpabet]


# ---------------------------------------------------------------------------
# 4) 시퀀스 정렬 (Needleman-Wunsch / edit distance + backtrace)
# ---------------------------------------------------------------------------
def align(reference: list, hypothesis: list):
    """reference(정답)와 hypothesis(사용자)를 정렬.
    반환: [(op, ref_ph, hyp_ph), ...]  op ∈ {match, sub, del, ins}"""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # 탈락 (deletion)
                dp[i][j - 1] + 1,        # 삽입 (insertion)
                dp[i - 1][j - 1] + cost  # 일치/대치
            )

    # backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1] \
                and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("match", reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", reference[i - 1], None))   # 사용자가 빠뜨림
            i -= 1
        else:
            ops.append(("ins", None, hypothesis[j - 1]))  # 사용자가 더 넣음
            j -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# 5) 채점 + 피드백
# ---------------------------------------------------------------------------
def score(alignment, confidence: float = None) -> dict:
    total = sum(1 for op, _, _ in alignment if op != "ins")  # 기준 음소 수
    correct = sum(1 for op, _, _ in alignment if op == "match")
    subs = sum(1 for op, _, _ in alignment if op == "sub")
    dels = sum(1 for op, _, _ in alignment if op == "del")
    inss = sum(1 for op, _, _ in alignment if op == "ins")
    accuracy = (correct / total * 100) if total else 0.0
    return {
        "accuracy": round(accuracy, 1),
        "correct": correct, "total": total,
        "substitutions": subs, "deletions": dels, "insertions": inss,
        "confidence": round(confidence, 3) if confidence is not None else None,
    }


def print_feedback(word, reference, hypothesis, alignment, result):
    print("\n" + "=" * 52)
    print(f"  단어: {word.upper()}")
    print(f"  정답 발음 : {' '.join(reference)}")
    print(f"  내 발음   : {' '.join(hypothesis) if hypothesis else '(인식 안 됨)'}")
    print("-" * 52)
    symbols = {"match": "✓", "sub": "✗", "del": "−", "ins": "+"}
    for op, ref_ph, hyp_ph in alignment:
        if op == "match":
            print(f"   {symbols[op]} {ref_ph}")
        elif op == "sub":
            print(f"   {symbols[op]} {ref_ph}  →  '{hyp_ph}' 로 발음함 (대치)")
        elif op == "del":
            print(f"   {symbols[op]} {ref_ph}  →  빠뜨림 (탈락)")
        elif op == "ins":
            print(f"   {symbols[op]} (불필요한 음소 '{hyp_ph}' 추가)")
    print("-" * 52)
    print(f"  정확도: {result['accuracy']}%  "
          f"(정답 {result['correct']}/{result['total']}, "
          f"대치 {result['substitutions']}, 탈락 {result['deletions']}, "
          f"삽입 {result['insertions']})")
    if result["confidence"] is not None:
        print(f"  인식 확신도: {result['confidence']}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def evaluate_once(recognizer, word, audio, as_ipa=False):
    reference = reference_phonemes(word, as_ipa=as_ipa)
    hypothesis, conf = recognizer.recognize(audio)
    alignment = align(reference, hypothesis)
    result = score(alignment, conf)
    print_feedback(word, reference, hypothesis, alignment, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seconds", type=float, default=DEFAULT_RECORD_SECONDS)
    parser.add_argument("--ipa", action="store_true",
                        help="모델이 IPA를 출력할 때 (예: facebook espeak 모델)")
    parser.add_argument("--inspect", action="store_true",
                        help="모델 음소 vocab만 출력하고 종료")
    parser.add_argument("--file", help="마이크 대신 오디오 파일로 테스트")
    args = parser.parse_args()

    recognizer = PhonemeRecognizer(args.model)

    if args.inspect:
        vocab = sorted(recognizer.vocab().keys())
        print(f"\n모델이 출력하는 음소 기호 ({len(vocab)}개):")
        print(" ".join(vocab))
        print("\n→ 위 기호가 IPA처럼 보이면 --ipa 옵션을 쓰세요.")
        return

    if args.file:
        word = input("목표 단어: ").strip()
        audio = load_audio_file(args.file)
        evaluate_once(recognizer, word, audio, as_ipa=args.ipa)
        return

    print("\n[발음 평가] 단어를 입력하고 발음하세요. (종료: q)\n")
    while True:
        word = input("목표 단어 > ").strip()
        if word.lower() in ("q", "quit", "exit", ""):
            break
        try:
            reference_phonemes(word, as_ipa=args.ipa)  # 사전 존재 확인
        except ValueError as e:
            print(f"  ! {e}\n")
            continue
        input("  (Enter를 누르면 녹음 시작) ")
        audio = record_audio(args.seconds)
        try:
            evaluate_once(recognizer, word, audio, as_ipa=args.ipa)
        except Exception as e:
            print(f"  ! 평가 중 오류: {e}\n")


if __name__ == "__main__":
    main()
