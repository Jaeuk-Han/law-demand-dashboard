# src/billinfo_fetch_and_merge_with_cache.py
# -*- coding: utf-8 -*-
"""
BillInfoService2 -> getBillInfoList 수집기
- 기본값: 법률안(B04)만 수집하도록 bill_kind_cd=B04 적용
- 날짜 범위 혹은 대수(국회) 범위로 필터 가능
- 페이지네이션 처리, XML 파싱(xmltodict), CSV 저장(utf-8-sig)
"""

import argparse
import math
import sys
import time
from typing import Dict, List, Any, Optional

import requests
import xmltodict
import pandas as pd

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# 기본 엔드포인트들 (우선 순위대로 시도)
HTTP_ENDPOINTS = [
    "http://apis.data.go.kr/9710000/BillInfoService2/getBillInfoList",
    "http://openapi.assembly.go.kr/openapi/service/BillInfoService2/getBillInfoList",
]
HTTPS_ENDPOINTS = [
    "https://apis.data.go.kr/9710000/BillInfoService2/getBillInfoList",
    "https://openapi.assembly.go.kr/openapi/service/BillInfoService2/getBillInfoList",
]

# 의안종류 코드 (필요시 확장)
KIND_CHOICES = [
    "B01", # 헌법개정안
    "B02", # 예산안
    "B03", # 결산
    "B04", # 법률안  ← 기본값
    "B05", # 동의안
    "B06", # 승인안
    "B07", # 결의안
    "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16"
]


def parse_xml_response(text: str) -> Dict[str, Any]:
    d = xmltodict.parse(text)
    # 표준 형태: response -> header -> resultCode/resultMsg, body -> items/item, totalCount
    return d


def extract_items_and_total(d: Dict[str, Any]) -> (List[Dict[str, Any]], int):
    try:
        body = d["response"]["body"]
        total = int(body.get("totalCount", 0))
        items = body.get("items", {})
        items = items.get("item", [])
        if items is None:
            return [], total
        if isinstance(items, dict):
            items = [items]
        # 값 트리밍
        norm_items = []
        for it in items:
            norm = {}
            for k, v in it.items():
                if isinstance(v, str):
                    norm[k] = v.strip()
                else:
                    norm[k] = v
            norm_items.append(norm)
        return norm_items, total
    except Exception:
        return [], 0


def try_request(session: requests.Session, url: str, params: Dict[str, Any], timeout: float = 20.0) -> Optional[requests.Response]:
    try:
        r = session.get(url, params=params, timeout=timeout, headers={"User-Agent": DEFAULT_UA})
        # 일부 환경에서 "Bad Request.<br><br><br>" 같은 비-XML 응답이 올 수 있음
        if r.status_code == 200 and r.text.strip().startswith("<"):
            return r
        return None
    except requests.RequestException:
        return None


def fetch_all(
    service_key: str,
    start_date: Optional[str],
    end_date: Optional[str],
    start_ord: Optional[int],
    end_ord: Optional[int],
    page_size: int,
    use_https_first: bool,
    bill_kind_cd: str,
    sleep_sec: float = 0.2,
) -> List[Dict[str, Any]]:

    endpoints = (HTTPS_ENDPOINTS + HTTP_ENDPOINTS) if use_https_first else (HTTP_ENDPOINTS + HTTPS_ENDPOINTS)

    # 공통 파라미터
    base_params = {
        "ServiceKey": service_key,     # 주의: 대소문자 'S' 필요
        "numOfRows": page_size,
        "pageNo": 1,
        "bill_kind_cd": bill_kind_cd,  # 기본 B04(법률안)
    }

    # 날짜 필터
    if start_date and end_date:
        base_params["start_propose_date"] = start_date
        base_params["end_propose_date"] = end_date

    # 대수 필터 (둘 다 들어오면 gbn=dae_num + 범위)
    if start_ord and end_ord:
        base_params["gbn"] = "dae_num"
        base_params["start_ord"] = str(start_ord)
        base_params["end_ord"] = str(end_ord)

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_UA, "Accept": "*/*"})

    # 1페이지로 totalCount 파악
    total_count = None
    first_items: List[Dict[str, Any]] = []
    for ep in endpoints:
        r = try_request(session, ep, base_params)
        if r is None:
            continue
        try:
            d = parse_xml_response(r.text)
            items, total = extract_items_and_total(d)
            total_count = total
            first_items = items
            base_url = ep  # 성공한 엔드포인트 고정
            break
        except Exception:
            continue

    if total_count is None:
        print("[ERROR] 모든 엔드포인트에서 유효한 XML 응답을 받지 못했습니다.", file=sys.stderr)
        return []

    print(f"[INFO] totalCount={total_count:,}  pageSize={page_size}")
    if total_count == 0:
        return []

    pages = math.ceil(total_count / page_size)
    all_items = []
    if first_items:
        print(f"[INFO] page 1 ok: items={len(first_items)}")
        all_items.extend(first_items)

    # 나머지 페이지
    for page in range(2, pages + 1):
        params = dict(base_params)
        params["pageNo"] = page
        r = try_request(session, base_url, params)
        if r is None:
            print(f"[WARN] page {page}: 응답 실패(엔드포인트 교체 시도)")
            # 같은 파라미터로 다른 엔드포인트 재시도
            reok = False
            for ep in endpoints:
                r2 = try_request(session, ep, params)
                if r2 is not None:
                    try:
                        d2 = parse_xml_response(r2.text)
                        items2, _ = extract_items_and_total(d2)
                        all_items.extend(items2)
                        print(f"[INFO] page {page} ok via fallback: items={len(items2)}")
                        base_url = ep
                        reok = True
                        break
                    except Exception:
                        pass
            if not reok:
                print(f"[ERROR] page {page}: 모든 엔드포인트 실패. 중단.", file=sys.stderr)
                break
        else:
            try:
                d = parse_xml_response(r.text)
                items, _ = extract_items_and_total(d)
                print(f"[INFO] page {page} ok: items={len(items)}")
                all_items.extend(items)
            except Exception as e:
                print(f"[WARN] page {page}: 파싱 실패: {e}")

        time.sleep(sleep_sec)

    return all_items


def to_dataframe(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()

    # 예상 주요 컬럼 (있으면 사용, 없으면 자동 포함)
    preferred_cols = [
        "billId", "billNo", "billName", "proposeDt", "proposerKind",
        "passGubn", "procStageCd", "committeeName", "generalResult",
        "bill_kind_cd",
    ]

    # 모든 키 수집
    all_keys = set()
    for it in items:
        all_keys.update(it.keys())
    # 우선 순서 + 나머지
    ordered_cols = [c for c in preferred_cols if c in all_keys] + [c for c in sorted(all_keys) if c not in preferred_cols]

    df = pd.DataFrame(items)
    # 공백/None 정리
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # 컬럼 순서 정렬
    df = df.reindex(columns=ordered_cols)
    return df


def main():
    p = argparse.ArgumentParser(description="Fetch bill list (BillInfoService2/getBillInfoList). Default: bill_kind_cd=B04(법률안)")
    p.add_argument("--service-key", required=True, help="공공데이터포털 일반 인증키 (decoding 필요없는 원문)")
    p.add_argument("--start-date", help="시작 제안일 (YYYY-MM-DD)")
    p.add_argument("--end-date", help="종료 제안일 (YYYY-MM-DD)")
    p.add_argument("--start-ord", type=int, help="시작 대수 (예: 21)")
    p.add_argument("--end-ord", type=int, help="종료 대수 (예: 22)")
    p.add_argument("--page-size", type=int, default=100, help="페이지 크기 (기본 100)")
    p.add_argument("--out", required=True, help="출력 CSV 경로")
    p.add_argument("--https-first", action="store_true", help="HTTPS 엔드포인트를 우선 시도 (기본은 HTTP 우선)")
    p.add_argument("--kind", default="B04", choices=KIND_CHOICES, help="의안종류코드(bill_kind_cd). 기본 B04(법률안)")

    args = p.parse_args()

    key_prefix = (args.service_key[:8] + "********") if args.service_key else "(none)"
    print(f"[INFO] using key prefix: {key_prefix}")

    items = fetch_all(
        service_key=args.service_key,
        start_date=args.start_date,
        end_date=args.end_date,
        start_ord=args.start_ord,
        end_ord=args.end_ord,
        page_size=args.page_size,
        use_https_first=args.https_first,
        bill_kind_cd=args.kind,
        sleep_sec=0.15,
    )

    df = to_dataframe(items)
    print(f"[INFO] collected rows: {len(df):,}")

    # CSV 저장 (엑셀 호환 좋게 BOM 포함)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved: {args.out} (rows={len(df):,})")


if __name__ == "__main__":
    main()
