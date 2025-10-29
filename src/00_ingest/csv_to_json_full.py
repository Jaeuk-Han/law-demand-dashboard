
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → JSON/JSONL converter that preserves *all* columns.

Features
- Reads UTF-8/UTF-8-SIG by default, falls back to CP949 if needed
- Keeps every column from the CSV as-is (no dropping)
- Optionally normalizes column names to snake_case
- Optional extra fields (e.g., _data_type=sns) appended to every row
- Saves either JSON array (*.json) or JSON Lines (*.jsonl)
- Safe for very long text fields (newlines preserved, ensure_ascii=False)

Usage
------
python csv_to_json_full.py input.csv output.json
python csv_to_json_full.py input.csv output.jsonl --jsonl
python csv_to_json_full.py input.csv output.json --snake-case-cols --extra _data_type=sns --extra _bucket_law=개인정보 보호법

Notes
-----
- If your CSV has a dedicated datetime column you want to reformat, use --date-col and --date-fmt-out.
- No columns are dropped; NaN becomes null in JSON.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def to_snake(name: str) -> str:
    # Normalize header to snake_case while being friendly to non-ASCII
    s = re.sub(r'\s+', '_', str(name).strip())
    s = re.sub(r'[^\w]+', '_', s, flags=re.UNICODE)  # keep unicode letters/digits/_
    s = re.sub(r'_+', '_', s).strip('_')
    return s.lower()


def read_csv_any(path: Path, encoding: str | None):
    encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "cp949"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", dtype=str, keep_default_na=True)
            return df, enc
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV with tried encodings {encodings}: {last_err}")


def main():
    ap = argparse.ArgumentParser(description="Convert CSV to JSON/JSONL preserving all columns.")
    ap.add_argument("input_csv", default="data/final_combined", type=Path, help="Path to input CSV")
    ap.add_argument("output_path", default="data/final_combined.json", type=Path, help="Output file path (.json or .jsonl)")
    ap.add_argument("--jsonl", action="store_true", help="Write JSON Lines instead of a JSON array")
    ap.add_argument("--encoding", type=str, default=None, help="Force CSV encoding (default: auto-try utf-8/utf-8-sig/cp949)")
    ap.add_argument("--snake-case-cols", action="store_true", help="Normalize column names to snake_case")
    ap.add_argument("--extra", action="append", default=[], help="Extra key=value to add to every record; can be given multiple times")
    ap.add_argument("--date-col", type=str, default=None, help="Column name containing a date to normalize (optional)")
    ap.add_argument("--date-fmt-in", type=str, default=None, help="strftime/strptime format of input date (optional)")
    ap.add_argument("--date-fmt-out", type=str, default="%Y%m%d%H%M%S", help="strftime format for output date (default: %%Y%%m%%d%%H%%M%%S)")

    args = ap.parse_args()

    df, used_enc = read_csv_any(args.input_csv, args.encoding)

    # Optional header normalization
    if args.snake_case_cols:
        df.columns = [to_snake(c) for c in df.columns]

    # Optional date normalization (non-destructive: write back into the same column if detected)
    if args.date_col:
        col = args.date_col
        if col not in df.columns:
            raise SystemExit(f"[ERROR] --date-col '{col}' not found in columns: {list(df.columns)}")
        try:
            if args.date_fmt_in:
                ts = pd.to_datetime(df[col], format=args.date_fmt_in, errors="coerce")
            else:
                ts = pd.to_datetime(df[col], errors="coerce")  # try pandas inference
            df[col] = ts.dt.strftime(args.date_fmt_out)
        except Exception as e:
            raise SystemExit(f"[ERROR] Failed to parse/format date column '{col}': {e}")

    # Replace NaN with None for JSON
    records = df.where(pd.notna(df), None).to_dict(orient="records")

    # Parse extra key=value pairs
    extra = {}
    for kv in args.extra:
        if "=" not in kv:
            raise SystemExit(f"[ERROR] --extra should be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        extra[k] = v

    if extra:
        for r in records:
            for k, v in extra.items():
                r[k] = v

    # Ensure parent directory
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.jsonl or args.output_path.suffix.lower() == ".jsonl":
        with args.output_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[DONE] Wrote {len(records)} JSONL lines → {args.output_path} (CSV encoding: {used_enc})")
    else:
        with args.output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Wrote {len(records)} records as a JSON array → {args.output_path} (CSV encoding: {used_enc})")


if __name__ == "__main__":
    main()
