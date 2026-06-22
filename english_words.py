"""
icard - 영어 학습 어휘 적재 모듈
================================================================
NGSL(고빈도 학습어) CSV -> SQLite 'english_words' 테이블.
선택적으로 WordNet으로 영어 뜻풀이를 보강한다.

데이터 받기:
  NGSL 1.2 (빈도 포함):
    https://www.newgeneralservicelist.com/s/NGSL_12_stats.csv
  (헤더: Lemma, SFI Rank, SFI, Adjusted Frequency per Million (U))

사용:
  python english_words.py build data/NGSL_12_stats.csv   # 적재 (레벨 자동 분할)
  python english_words.py wordnet                         # 영어 뜻 보강(선택)
  python english_words.py levels                          # 레벨별 단어 수
  python english_words.py level 1                         # 레벨 1 단어 미리보기

발음 정답은 CMUdict(pronouncing)로 별도 조회되므로 여기서 다루지 않는다.
"""

import csv
import sqlite3
import sys

csv.field_size_limit(10 ** 7)

DB_PATH = "icard_dict.db"     # 한국어 사전과 같은 DB 파일, 테이블만 분리
TABLE = "english_words"

WORD_KEYS = ("lemma", "word", "headword", "ngsl")
RANK_KEYS = ("sfi rank", "rank")


def _find(header, keys):
    low = [h.strip().lower() for h in header]
    for i, h in enumerate(low):
        if h in keys:
            return i
    for i, h in enumerate(low):           # 부분일치 폴백
        if any(k in h for k in keys):
            return i
    return None


def build_english(input_path, db_path=DB_PATH, band_size=500, delimiter=","):
    """NGSL CSV -> english_words 테이블. rank 기준으로 레벨(밴드) 자동 분할."""
    with open(input_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        wi = _find(header, WORD_KEYS)
        ri = _find(header, RANK_KEYS)
        if wi is None:
            raise ValueError("단어(Lemma) 컬럼을 찾지 못했습니다. 헤더를 확인하세요.")
        rows = []
        for n, row in enumerate(reader, 1):
            if not row or wi >= len(row):
                continue
            word = row[wi].strip()
            if not word:
                continue
            try:
                rank = int(row[ri]) if (ri is not None and ri < len(row)
                                        and row[ri].strip()) else n
            except ValueError:
                rank = n
            level = (rank - 1) // band_size + 1
            rows.append((word, rank, level))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    cur.execute(f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, word TEXT, "
                f"rank INTEGER, level INTEGER, definition TEXT, pos TEXT)")
    cur.executemany(f"INSERT INTO {TABLE} (word, rank, level) VALUES (?,?,?)", rows)
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_ew_word ON {TABLE}(word)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_ew_level ON {TABLE}(level)")
    conn.commit()
    conn.close()
    top_level = rows[-1][2] if rows else 0
    print(f"✓ 적재: {len(rows)}단어 → {db_path} ({TABLE}), "
          f"레벨 1~{top_level} (밴드 {band_size}단어)")
    return len(rows)


def enrich_wordnet(db_path=DB_PATH):
    """WordNet 첫 번째 의미로 정의/품사 보강 (선택). nltk 필요."""
    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(f"SELECT id, word FROM {TABLE}").fetchall()
    upd = []
    for id_, word in rows:
        ss = wn.synsets(word)
        if ss:
            upd.append((ss[0].definition(), ss[0].pos(), id_))
    cur.executemany(f"UPDATE {TABLE} SET definition=?, pos=? WHERE id=?", upd)
    conn.commit()
    conn.close()
    print(f"✓ WordNet 정의 보강: {len(upd)}/{len(rows)}단어")


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------
def _conn(db_path=DB_PATH):
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def levels(db_path=DB_PATH):
    """레벨별 (레벨, 단어수) 목록."""
    c = _conn(db_path)
    rows = c.execute(f"SELECT level, COUNT(*) AS n FROM {TABLE} "
                     f"GROUP BY level ORDER BY level").fetchall()
    c.close()
    return [(r["level"], r["n"]) for r in rows]


def words_by_level(level, db_path=DB_PATH, limit=500):
    c = _conn(db_path)
    rows = c.execute(f"SELECT * FROM {TABLE} WHERE level=? ORDER BY rank LIMIT ?",
                     (level, limit)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def search_english(prefix, db_path=DB_PATH, limit=30):
    c = _conn(db_path)
    rows = c.execute(f"SELECT * FROM {TABLE} WHERE word LIKE ? ORDER BY rank LIMIT ?",
                     (f"{prefix}%", limit)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build_english(sys.argv[2])
    elif cmd == "wordnet":
        enrich_wordnet()
    elif cmd == "levels":
        for lv, n in levels():
            print(f"레벨 {lv}: {n}단어")
    elif cmd == "level":
        for r in words_by_level(int(sys.argv[2]))[:20]:
            print(r["rank"], r["word"], "-", (r.get("definition") or ""))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
