# -*- coding: utf-8 -*-
"""
의안명(billName)에서 법률명(lawName)을 추출하고, 같은 법률명에 속한 의안들을 묶어 저장합니다.
결과는 CSV와 JSON 두 가지 포맷으로 저장됩니다.

Usage:
    python extract_law_groups_v2.py --in billinfo_2025.json --out ./out_dir

입력 JSON 형식:
[
  {
    "billId": "PRC_AAA...",
    "billNo": "2101234",
    "billName": "개인정보 보호법 일부개정법률안(이강일의원 등 14인)",
    "summary": "..."
  },
  ...
]

출력 파일:
- bills_with_law_extracted.csv          : 의안별 추출 결과
- bill_groups_with_summary.json         : 법률명별 그룹 (요약 포함)
- bill_groups.csv                       : 법률명별 그룹 요약
"""

import json, re, unicodedata, argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


# --------------------------
# 문자열 정규화 함수
# --------------------------
def normalize(s: str) -> str:
    """괄호, 특수문자, 중복공백 제거"""
    s = unicodedata.normalize("NFC", s or "")
    s = (s.replace("「","").replace("」","")
           .replace("『","").replace("』","")
           .replace("“","").replace("”","")
           .replace('"',''))
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------
# 정규식 패턴 정의
# --------------------------
paren_tail_re = re.compile(r"\s*(\([^()]*\)\s*)+$")  # 괄호 꼬리 제거
amend_phrase_re = re.compile(r"(?:일부|전부|일괄|타법)?개정법률안")  # 개정법률안 패턴
law_tail_re = re.compile(r"(?P<name>.+?(?:법률|법))\s*$")  # ...법 or ...법률
lawbill_tail_re = re.compile(r"(?P<name>.+?법안)\s*$")  # ...법안


# --------------------------
# 핵심 함수: 법률명 추출
# --------------------------
def extract_law_name(bill_name: str) -> str | None:
    """billName에서 법률명(lawName) 추출"""
    original = normalize(bill_name)
    core = paren_tail_re.sub("", original)

    # 1) 개정법률안 패턴
    m = amend_phrase_re.search(core)
    if m:
        prefix = normalize(core[:m.start()])
        m2 = law_tail_re.search(prefix)
        if m2:
            return m2.group("name")
        return prefix or None

    # 2) 법안 패턴
    m = lawbill_tail_re.search(core)
    if m:
        return m.group("name")

    # 3) 일반 법률명 패턴
    m = law_tail_re.search(core)
    if m:
        return m.group("name")

    return None


# --------------------------
# 실행 로직
# --------------------------
def run(input_path: str, out_dir: str) -> dict:
    p = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(p.read_text(encoding="utf-8"))

    rows = []
    missing = 0
    for it in data:
        bill_id = it.get("billId")
        bill_no = it.get("billNo")
        bill_name = it.get("billName", "")
        summary = it.get("summary", "")
        law = extract_law_name(bill_name) or ""
        if not law:
            missing += 1
        rows.append({
            "billId": bill_id,
            "billNo": bill_no,
            "billName": bill_name,
            "lawName": law,
            "summary": summary
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "bills_with_law_extracted.csv", index=False, encoding="utf-8-sig")

    # 그룹핑
    groups = defaultdict(list)
    for r in rows:
        if r["lawName"]:
            groups[r["lawName"]].append({
                "billId": r["billId"],
                "billNo": r["billNo"],
                "billName": r["billName"],
                "summary": r["summary"]
            })

    grouped = [{"lawName": law, "count": len(lst), "bills": lst}
               for law, lst in groups.items()]
    grouped_sorted = sorted(grouped, key=lambda x: (-x["count"], x["lawName"]))

    # JSON & CSV 저장
    with (out / "bill_groups_with_summary.json").open("w", encoding="utf-8") as f:
        json.dump(grouped_sorted, f, ensure_ascii=False, indent=2)

    pd.DataFrame([{"lawName": g["lawName"], "count": g["count"]}
                  for g in grouped_sorted]).to_csv(out / "bill_groups.csv",
                                                   index=False, encoding="utf-8-sig")

    return {
        "total": len(rows),
        "no_law_extracted": missing,
        "out_dir": str(out.resolve())
    }


# --------------------------
# CLI 진입점
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="입력 JSON 파일 경로")
    ap.add_argument("--out", dest="out", required=True, help="출력 디렉토리 경로")
    args = ap.parse_args()
    stats = run(args.inp, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
