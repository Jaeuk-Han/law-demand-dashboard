# -*- coding: utf-8 -*-
"""
Law Bucketizer (파일별 분리 저장 + JSON 동시 저장)
- 각 입력 파일을 열어 그 파일 내부 레코드만 법률별로 분리 저장합니다.
- 결과: OUTPUT/<원본파일명>/<data_type>/<법률명>.jsonl 및 .json
"""
import os, re, json, csv, glob
from collections import defaultdict
from typing import Dict, Any, Iterable, List, Optional

# ========= [CONFIG] =========
INPUT_DIR  = "./data/temp"
OUTPUT_DIR = "./final/data_buckets"
FILE_GLOB  = "*.json"

CANONICAL_LAWS = [
    "개인정보 보호법",
    "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "아동복지법",
    "자본시장과 금융투자업에 관한 법률",
    "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "전자금융거래법",
    "전자증권의 발행 및 유통에 관한 법률",
    "금융소비자보호법",
    "중대재해 처벌 등에 관한 법률",
]

LAW_ALIAS = {
    "개보법": "개인정보 보호법",
    "개인정보보호법": "개인정보 보호법",
    "정보통신망법": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "정보통신망": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "자본시장법": "자본시장과 금융투자업에 관한 법률",
    "금융투자업법": "자본시장과 금융투자업에 관한 법률",
    "특정금융정보법": "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "특금법": "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "전자금융거래법": "전자금융거래법",
    "전자증권법": "전자증권의 발행 및 유통에 관한 법률",
    "금융소비자보호법": "금융소비자보호법",
    "금소법": "금융소비자보호법",
    "금소보법": "금융소비자보호법",
    "아동복지법": "아동복지법",
    "아복법": "아동복지법",
    "중대재해처벌법": "중대재해 처벌 등에 관한 법률",
    "중처법": "중대재해 처벌 등에 관한 법률",
}

POSSIBLE_LAW_FIELDS = ["law", "law_name", "법률명", "법률", "target_law"]
POSSIBLE_TITLE_FIELDS = ["billName", "title", "기사제목"]
POSSIBLE_TEXT_FIELDS  = ["summary", "content", "text", "본문", "cleaned_text"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= [UTILS] =========
def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip()

def detect_data_type(filename: str) -> str:
    base = os.path.basename(filename)
    if "_social_processed" in base:
        return "sns"
    elif "_processed" in base:
        return "news"
    return "news"

def candidates_from_filename(filename: str) -> List[str]:
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"_social_processed$|_processed$", "", name)
    parts = [p for p in name.split("_") if "법" in p]
    return [LAW_ALIAS.get(x, x) for x in parts]

LAW_REGEXES = [
    r"([가-힣0-9·\-\s]+?에 관한 법률)",
    r"([가-힣0-9·\-\s]+?보호법)",
    r"([가-힣0-9·\-\s]+?거래법)",
    r"([가-힣0-9·\-\s]+?소비자보호법)",
    r"([가-힣0-9·\-\s]+?복지법)",
    r"([가-힣0-9·\-\s]+?법)(?:안| 일부개정법률안| 전부개정법률안| 개정안)?",
]

def extract_law_candidates_from_text(text: str) -> List[str]:
    if not text:
        return []
    found = []
    for pat in LAW_REGEXES:
        for m in re.finditer(pat, text):
            cand = m.group(1).strip()
            if 2 <= len(cand) <= 60:
                found.append(cand)
    return list(dict.fromkeys(found))

def normalize_law_name(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = LAW_ALIAS.get(raw.strip(), raw.strip())
    if raw in CANONICAL_LAWS:
        return raw
    for law in CANONICAL_LAWS:
        if raw in law or law in raw:
            return law
    return None

def iter_json_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.strip().startswith("["):
            for rec in json.load(f):
                if isinstance(rec, dict):
                    yield rec
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        yield rec
                except:
                    continue

def decide_law_for_record(rec: Dict[str, Any], fname_cands: List[str]) -> Optional[str]:
    for k in POSSIBLE_LAW_FIELDS:
        if k in rec and rec[k]:
            norm = normalize_law_name(str(rec[k]))
            if norm:
                return norm
    bag = " ".join(str(rec.get(k, "")) for k in POSSIBLE_TITLE_FIELDS + POSSIBLE_TEXT_FIELDS)
    for cand in extract_law_candidates_from_text(bag):
        norm = normalize_law_name(cand)
        if norm:
            return norm
    for cand in fname_cands:
        norm = normalize_law_name(cand)
        if norm:
            return norm
    return None

def save_jsonl_and_json(path_base: str, records: List[Dict[str, Any]]):
    """동시에 jsonl + json 저장"""
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    jsonl_path = path_base + ".jsonl"
    json_path = path_base + ".json"

    # JSONL 저장
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # JSON (리스트 전체)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# ========= [MAIN] =========
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB)))
    print(f"[INFO] Found {len(files)} files")

    for fp in files:
        data_type = detect_data_type(fp)
        fname_cands = candidates_from_filename(fp)
        base = os.path.splitext(os.path.basename(fp))[0]
        base_s = sanitize(base)

        buckets = defaultdict(list)
        total, assigned = 0, 0

        for rec in iter_json_records(fp):
            total += 1
            law = decide_law_for_record(rec, fname_cands)
            if law:
                assigned += 1
                rec_out = dict(rec)
                rec_out["_bucket_law"] = law
                rec_out["_data_type"] = data_type
                rec_out["_source_file"] = os.path.basename(fp)
                buckets[law].append(rec_out)

        for law, items in buckets.items():
            law_s = sanitize(law)
            subdir = os.path.join(OUTPUT_DIR, base_s, data_type)
            base_path = os.path.join(subdir, law_s)
            save_jsonl_and_json(base_path, items)

        print(f"- {os.path.basename(fp)} | {data_type} | total={total} | assigned={assigned} | laws={len(buckets)}")

    print("\n[DONE] 파일별 버킷화 및 JSON 동시 저장 완료!")
    print(f"[OUTPUT] {OUTPUT_DIR}")
