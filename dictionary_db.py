"""
icard - 다국어 사전 적재 / 조회 모듈
================================================================
국립국어원 한국어기초사전(다국어 학습용) TSV/CSV를 SQLite로 적재하고,
한국어 표제어 / 영어 대역어 / 전문검색(FTS)으로 빠르게 조회한다.

표준 라이브러리만 사용 (csv, sqlite3). 별도 설치 불필요.

사용:
  # 1) 적재 (최초 1회)
  python dictionary_db.py build  내사전.tsv

  # 2) 조회
  python dictionary_db.py ko 가              # 한국어 표제어로
  python dictionary_db.py en edge            # 영어 대역어로 역조회
  python dictionary_db.py search 둘레        # 전문검색
"""

import csv
import sqlite3
import sys

# csv 필드에 긴 용례가 들어가므로 상한을 올린다
csv.field_size_limit(10 ** 7)

DB_PATH = "icard_dict.db"

# 원본 한글 헤더 -> SQLite 컬럼명(영문 snake_case) 매핑
COLUMN_MAP = {
    "표제어": "headword",
    "동형어 번호": "homonym_no",
    "구분": "entry_type",
    "품사": "pos",
    "고유어 여부": "native_yn",
    "원어": "origin",
    "발음": "pronunciation",
    "활용": "conjugation",
    "파생어": "derivative",
    "☞(가 보라)": "see_also",
    "어휘 등급": "level",
    "의미 범주": "sense_category",
    "주제 및 상황 범주": "topic_category",
    "전체 참고": "overall_ref",
    "검색용 이형태": "search_variants",
    "관련어": "related",
    "의미 참고": "sense_ref",
    "다중 매체 정보": "multimedia",
    "문형": "sentence_pattern",
    "문형 참고": "sentence_pattern_ref",
    "뜻풀이": "definition",
    "용례": "examples",
    "몽골어 대역어": "mn_equiv", "몽골어 대역어 뜻풀이": "mn_gloss",
    "아랍어 대역어": "ar_equiv", "아랍어 대역어 뜻풀이": "ar_gloss",
    "중국어 대역어": "zh_equiv", "중국어 대역어 뜻풀이": "zh_gloss",
    "베트남어 대역어": "vi_equiv", "베트남어 대역어 뜻풀이": "vi_gloss",
    "타이어 대역어": "th_equiv", "타이어 대역어 뜻풀이": "th_gloss",
    "인도네시아어 대역어": "id_equiv", "인도네시아어 대역어 뜻풀이": "id_gloss",
    "러시아어 대역어": "ru_equiv", "러시아어 대역어 뜻풀이": "ru_gloss",
    "영어 대역어": "en_equiv", "영어 대역어 뜻풀이": "en_gloss",
    "일본어 대역어": "ja_equiv", "일본어 대역어 뜻풀이": "ja_gloss",
    "프랑스어 대역어": "fr_equiv", "프랑스어 대역어 뜻풀이": "fr_gloss",
    "스페인어 대역어": "es_equiv", "스페인어 대역어 뜻풀이": "es_gloss",
}


def _slug(name, idx):
    """매핑에 없는 헤더는 안전한 컬럼명으로 변환."""
    return COLUMN_MAP.get(name.strip(), f"col_{idx}")


# ---------------------------------------------------------------------------
# 적재
# ---------------------------------------------------------------------------
def _sniff_delimiter(path):
    """첫 줄을 보고 탭/쉼표 구분자를 추정 (텍스트 파일용)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        first = f.readline()
    return "\t" if first.count("\t") >= first.count(",") else ","


def _is_excel(path):
    return path.lower().endswith((".xls", ".xlsx", ".xlsm"))


def _iter_rows(path, delimiter=None):
    """파일 형식(csv/tsv/xls/xlsx)에 상관없이 행(list[str])을 순서대로 yield.
    첫 행이 헤더다. 엑셀은 pandas로 읽는다(.xls→xlrd, .xlsx→openpyxl 필요)."""
    if _is_excel(path):
        import pandas as pd
        df = pd.read_excel(path, header=None, dtype=str, keep_default_na=False)
        for row in df.itertuples(index=False, name=None):
            yield ["" if v is None else str(v) for v in row]
    else:
        delim = delimiter or _sniff_delimiter(path)
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f, delimiter=delim, quotechar='"'):
                yield row


def build_db(input_paths, db_path=DB_PATH, delimiter=None):
    """여러 파일(csv/tsv/xls/xlsx)을 하나의 SQLite로 적재 (이어붙임).

    파일마다 컬럼 순서가 달라도 헤더 '이름' 기준으로 정렬해 넣으므로 안전하다.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    # 1) 모든 파일의 헤더를 읽어 표준 컬럼 집합(첫 등장 순서의 합집합) 구성
    file_cols = {}
    canonical, seen = [], set()
    for path in input_paths:
        header = next(_iter_rows(path, delimiter))
        cols = [_slug(h, i) for i, h in enumerate(header)]
        file_cols[path] = cols
        for c in cols:
            if c not in seen:
                seen.add(c)
                canonical.append(c)

    # 2) 테이블 생성
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS entries")
    col_defs = ", ".join(f'"{c}" TEXT' for c in canonical)
    cur.execute(f"CREATE TABLE entries (id INTEGER PRIMARY KEY, {col_defs})")
    placeholders = ", ".join("?" for _ in canonical)
    insert_sql = (f'INSERT INTO entries ({", ".join(chr(34)+c+chr(34) for c in canonical)}) '
                  f"VALUES ({placeholders})")

    # 3) 파일별로 이름 기반 적재
    total = 0
    for path in input_paths:
        cols = file_cols[path]
        n = 0
        batch = []
        rows = _iter_rows(path, delimiter)
        next(rows)  # 헤더 건너뜀
        for row in rows:
            row = (row + [""] * len(cols))[:len(cols)]
            d = dict(zip(cols, row))                  # 이 파일의 헤더→값
            batch.append([d.get(c, "") for c in canonical])
            if len(batch) >= 1000:
                cur.executemany(insert_sql, batch)
                n += len(batch)
                batch = []
        if batch:
            cur.executemany(insert_sql, batch)
            n += len(batch)
        print(f"  + {path}: {n}행")
        total += n

    # 4) 인덱스
    for col in ("headword", "en_equiv", "level", "pos"):
        if col in canonical:
            cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{col} ON entries("{col}")')

    conn.commit()
    conn.close()
    print(f"✓ 적재 완료: 총 {total}행 → {db_path}")
    return total


def info(db_path=DB_PATH):
    """적재 결과 검증용: 건수 + 채워진 컬럼 + 샘플 출력."""
    conn = _connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(entries)").fetchall()]
    print(f"총 표제어: {total}")
    print(f"컬럼({len(cols)}): {', '.join(cols)}")
    print("\n샘플 2건:")
    for r in conn.execute("SELECT * FROM entries LIMIT 2").fetchall():
        r = dict(r)
        print(f"  - {r.get('headword','')} [{r.get('pos','')}] "
              f"발음:{r.get('pronunciation','')} / 영어:{r.get('en_equiv','')}")
    conn.close()


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------
def _connect(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def lookup_korean(word, db_path=DB_PATH):
    """한국어 표제어로 조회 (동형어 모두 반환)."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM entries WHERE headword = ? ORDER BY homonym_no", (word,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def lookup_by_english(term, db_path=DB_PATH, limit=20):
    """영어 대역어로 역조회 (영어→한국어). 부분일치."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM entries WHERE en_equiv LIKE ? LIMIT ?",
        (f"%{term}%", limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def prefix_korean(prefix, db_path=DB_PATH, limit=20):
    """표제어 접두 검색 (자동완성용). headword 인덱스를 타서 빠름."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM entries WHERE headword LIKE ? "
        "ORDER BY headword, homonym_no LIMIT ?", (f"{prefix}%", limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search(query, db_path=DB_PATH, limit=20):
    """부분일치 검색 — 한국어 뜻풀이/표제어/영어 대역어에서 substring 매칭.
    한국어는 FTS 토크나이저가 부적합하므로 LIKE 기반이 안전하다."""
    conn = _connect(db_path)
    like = f"%{query}%"
    rows = conn.execute(
        "SELECT * FROM entries WHERE headword LIKE ? OR definition LIKE ? "
        "OR en_equiv LIKE ? OR en_gloss LIKE ? LIMIT ?",
        (like, like, like, like, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 출력 + CLI
# ---------------------------------------------------------------------------
def _show(rows):
    if not rows:
        print("  결과 없음")
        return
    for r in rows:
        hw = r.get("headword", "")
        hn = r.get("homonym_no", "")
        pos = r.get("pos", "")
        pron = r.get("pronunciation", "")
        print(f"\n■ {hw}{('('+hn+')') if hn else ''}  [{pos}]  발음:{pron}")
        print(f"  뜻풀이 : {r.get('definition','')}")
        if r.get("en_equiv"):
            print(f"  영어   : {r.get('en_equiv','')} — {r.get('en_gloss','')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "build":
        build_db(sys.argv[2:])   # 여러 파일 한 번에: build a.tsv b.tsv c.tsv
    elif cmd == "ko":
        _show(lookup_korean(sys.argv[2]))
    elif cmd == "en":
        _show(lookup_by_english(sys.argv[2]))
    elif cmd == "search":
        _show(search(sys.argv[2]))
    elif cmd == "prefix":
        _show(prefix_korean(sys.argv[2]))
    elif cmd == "info":
        info()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()