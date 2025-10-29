# -*- coding: utf-8 -*-
import os, json, argparse, csv
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter

def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data

def safe_str(x):
    return "" if x is None else str(x)

def normalize_keywords(v) -> str:
    """list면 join, 문자열이면 strip. 없으면 빈 문자열."""
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(map(lambda s: safe_str(s).strip(), v))
    return safe_str(v).strip()

def get_topic_keywords(group: List[Dict[str, Any]]) -> str:
    """토픽 그룹 내에서 첫 번째 비어있지 않은 keywords를 대표로 사용."""
    for r in group:
        kw = normalize_keywords(r.get("keywords"))
        if kw:
            return kw
    return ""

def aggregate_topic_top_law(
    records: List[Dict[str, Any]],
    bills_key: str = "bills",
    top_n_bills: int = 20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    반환:
    - results_json: 토픽별 최다 법률 + 해당 법률의 의안 집계 상세(JSON용)
    - rows_topic_toplaw_csv: topic별 최다 법률 요약(CSV용)
    - rows_topic_bills_csv: (topic×최다법률) 의안별 상세(CSV용, summary 포함)
    ※ keywords: 토픽 대표 키워드(그 토픽 그룹의 첫 유효값)
    """
    # 1) 토픽 그룹핑
    topic_groups = defaultdict(list)
    for rec in records:
        topic = safe_str(rec.get("topic"))
        topic_groups[topic].append(rec)

    results_json, rows_topic_toplaw_csv, rows_topic_bills_csv = [], [], []

    for topic, group in sorted(topic_groups.items(), key=lambda x: x[0]):
        # 대표 topic_name: 최빈값
        topic_name_counts = Counter(
            [safe_str(g.get("topic_name")).strip() for g in group if g.get("topic_name")]
        )
        topic_name = topic_name_counts.most_common(1)[0][0] if topic_name_counts else ""

        # 대표 keywords: 그룹 내 첫 유효값(동일하다고 가정)
        topic_keywords = get_topic_keywords(group)

        # 2) 토픽 내 전체 bill mentions 수집
        law_counter = Counter()
        total_bill_mentions = 0
        per_bill_stats_all = []  # (law, bill_id, bill_name, bill_summary, score)

        for rec in group:
            bills = rec.get(bills_key) or []
            for b in bills:
                law = safe_str(b.get("lawName")).strip()
                bill_id = safe_str(b.get("billId")).strip()
                bill_name = safe_str(b.get("billName")).strip()
                bill_summary = safe_str(b.get("summary")).strip()
                score = float(b.get("score", 0.0))
                if not law:
                    continue
                law_counter[law] += 1
                total_bill_mentions += 1
                per_bill_stats_all.append((law, bill_id, bill_name, bill_summary, score))

        # bills가 전혀 없으면 빈 구조 출력
        if total_bill_mentions == 0 or not law_counter:
            results_json.append({
                "topic": topic,
                "topic_name": topic_name,
                "keywords": topic_keywords,
                "top_law": None,
                "total_records": len(group),
                "total_bill_mentions": 0,
                "bills": []
            })
            rows_topic_toplaw_csv.append({
                "topic": topic,
                "topic_name": topic_name,
                "keywords": topic_keywords,
                "top_law": "",
                "law_mentions": 0,
                "law_ratio": 0.0,
                "total_records": len(group),
                "total_bill_mentions": 0
            })
            continue

        # 3) 최다 법률 선택 (동률 시 사전순)
        max_count = max(law_counter.values())
        candidate_laws = [lw for lw, c in law_counter.items() if c == max_count]
        top_law = sorted(candidate_laws)[0]
        top_law_count = law_counter[top_law]
        law_ratio = top_law_count / total_bill_mentions if total_bill_mentions else 0.0

        # 4) 최다 법률에 속한 의안 집계
        bill_counter = Counter()
        bill_score_sum = defaultdict(float)
        bill_score_max = defaultdict(float)
        bill_name_map: Dict[str, str] = {}
        bill_summary_map: Dict[str, str] = {}

        for law, bill_id, bill_name, bill_summary, score in per_bill_stats_all:
            if law != top_law:
                continue
            key = bill_id if bill_id else bill_name  # billId가 없을 경우 대비

            bill_counter[key] += 1
            bill_score_sum[key] += score
            bill_score_max[key] = max(bill_score_max[key], score)

            # 가장 길이가 긴 이름/요약을 보존(정보량 최대 기준)
            if key not in bill_name_map or len(bill_name) > len(bill_name_map[key]):
                bill_name_map[key] = bill_name
            if bill_summary and (key not in bill_summary_map or len(bill_summary) > len(bill_summary_map[key])):
                bill_summary_map[key] = bill_summary

        # 의안 정렬: 등장횟수 desc, max_score desc, bill_name asc
        sorted_bills = sorted(
            bill_counter.items(),
            key=lambda x: (x[1], bill_score_max[x[0]], bill_name_map.get(x[0], "")),
            reverse=True
        )

        bills_out = []
        for i, (key, cnt) in enumerate(sorted_bills[:top_n_bills], start=1):
            avg_score = bill_score_sum[key] / cnt if cnt else 0.0
            bills_out.append({
                "rank": i,
                "billId_or_key": key,
                "billName": bill_name_map.get(key, ""),
                "billSummary": bill_summary_map.get(key, ""),
                "occurrences": cnt,
                "avg_score": round(avg_score, 6),
                "max_score": round(bill_score_max[key], 6),
            })
            rows_topic_bills_csv.append({
                "topic": topic,
                "topic_name": topic_name,
                "keywords": topic_keywords,  # 토픽별 keywords
                "lawName": top_law,
                "billId_or_key": key,
                "billName": bill_name_map.get(key, ""),
                "bill_summary": bill_summary_map.get(key, ""),
                "occurrences": cnt,
                "avg_score": round(avg_score, 6),
                "max_score": round(bill_score_max[key], 6),
            })

        results_json.append({
            "topic": topic,
            "topic_name": topic_name,
            "keywords": topic_keywords,  # 토픽별 keywords
            "top_law": {
                "lawName": top_law,
                "law_mentions": top_law_count,
                "law_ratio": round(law_ratio, 6),
            },
            "total_records": len(group),
            "total_bill_mentions": total_bill_mentions,
            "bills": bills_out
        })

        rows_topic_toplaw_csv.append({
            "topic": topic,
            "topic_name": topic_name,
            "keywords": topic_keywords,  # 토픽별 keywords
            "top_law": top_law,
            "law_mentions": top_law_count,
            "law_ratio": round(law_ratio, 6),
            "total_records": len(group),
            "total_bill_mentions": total_bill_mentions
        })

    return results_json, rows_topic_toplaw_csv, rows_topic_bills_csv

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="enriched.json 경로 (list 또는 단일 객체)")
    ap.add_argument("--out_dir", required=True, help="출력 디렉토리")
    ap.add_argument("--bills_key", default="bills", help="리트리브 결과가 담긴 키 이름 (기본: bills)")
    ap.add_argument("--top_n_bills", type=int, default=20, help="법률 하위 의안 상위 N개 출력")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_json_list(args.input_path)
    results_json, rows_topic_toplaw_csv, rows_topic_bills_csv = aggregate_topic_top_law(
        records, bills_key=args.bills_key, top_n_bills=args.top_n_bills
    )

    # 저장
    json_out = os.path.join(args.out_dir, "topic_top_law.json")
    csv_toplaw_out = os.path.join(args.out_dir, "topic_top_law.csv")
    csv_bills_out = os.path.join(args.out_dir, "topic_top_law_bills.csv")

    save_json(json_out, results_json)
    save_csv(
        csv_toplaw_out,
        rows_topic_toplaw_csv,
        ["topic", "topic_name", "keywords", "top_law", "law_mentions", "law_ratio", "total_records", "total_bill_mentions"]
    )
    save_csv(
        csv_bills_out,
        rows_topic_bills_csv,
        ["topic", "topic_name", "keywords", "lawName", "billId_or_key", "billName", "bill_summary", "occurrences", "avg_score", "max_score"]
    )

    print("[DONE] Saved:")
    print(" -", json_out)
    print(" -", csv_toplaw_out)
    print(" -", csv_bills_out)

if __name__ == "__main__":
    main()
