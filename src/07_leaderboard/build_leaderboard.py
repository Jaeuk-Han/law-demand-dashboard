# -*- coding: utf-8 -*-
"""
Leaderboard v10 — keywords 개선형


- Table: show top 8 keywords (+N), Expand: show ALL keywords
- Stats: GOLD on labels (Share/Mentions/Avg/Max), numbers default color
- Summary processing:
    * Remove leading '제안이유…' header (제안이유/대안의 제안이유/…)
    * Show a cyan/Slate section chip with detected label
    * 6-line clamp + More/Less
- Header(Hero) section: gradient + subtle grid + medal badges + mini stat cards
"""
import json, os, html, datetime, re, argparse, ast
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# ---------------- small utils ----------------
def pick(d: Dict[str, Any], ks: List[str]):
    for k in ks:
        if k in d and d[k] is not None:
            return d[k]
    return None

def bill_id(b: Dict[str, Any]):
    return pick(b, ["billId_or_key","billId","id","key","name"])

def bill_name(b: Dict[str, Any]):
    return pick(b, ["billName","name","title","bill_title"])

def bill_mentions(b: Dict[str, Any]):
    v = pick(b, ["occurrences","bill_mentions","mentions","count"])
    try:
        return int(v)
    except Exception:
        try:
            return float(v)
        except Exception:
            return None

def pct(x, dp=1):
    if x is None: return ""
    try:
        return f"{float(x)*100:.{dp}f}%"
    except Exception:
        return ""

def norm_bills_full(bills: List[Dict[str, Any]], total_bill_mentions: Optional[float]):
    if not bills:
        return []
    ranked = [b for b in bills if isinstance(b.get("rank"), int)]
    bills_sorted = sorted(ranked, key=lambda x: x.get("rank", 10**9)) if ranked \
                   else sorted(bills, key=lambda x: (bill_mentions(x) or -1), reverse=True)
    res = []
    for i, b in enumerate(bills_sorted, start=1):
        m = bill_mentions(b) or 0
        share = (float(m) / float(total_bill_mentions)) if total_bill_mentions else None
        res.append({
            "rank": i if b.get("rank") is None else b.get("rank"),
            "bill_id": bill_id(b),
            "bill_name": bill_name(b),
            "mentions": m,
            "avg_score": pick(b, ["avg_score","avgScore"]),
            "max_score": pick(b, ["max_score","maxScore"]),
            "summary": pick(b, ["billSummary","summary","desc"]),
            "share": share
        })
    return res

# ---------------- keywords ----------------
def normalize_keywords(raw: Any, limit: int = 12) -> List[str]:
    items: List[str] = []
    if raw is None:
        return items
    if isinstance(raw, list):
        items = [str(x) for x in raw]
    elif isinstance(raw, str):
        s = raw.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    items = [str(x) for x in parsed]
                else:
                    items = [s]
            except Exception:
                items = [p.strip() for p in re.split(r'[,\u3001]', s) if p.strip()]
        else:
            items = [p.strip() for p in re.split(r'[,\u3001]', s) if p.strip()]
    else:
        items = [str(raw)]
    seen = set(); clean=[]
    for x in items:
        t = re.sub(r'\s+', ' ', x).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); clean.append(t)
    return clean[:limit]

# ---------------- summary processing ----------------
HEADER_RE = re.compile(
    r"""^\s*
        (?:대안의\s*)?
        제안\s*이유
        (?:\s*및\s*주요내용)?
        \s*[:：]?\s*$
    """, re.VERBOSE
)

def process_summary(text: str) -> Tuple[str, Optional[str]]:
    if not text:
        return "", None
    s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = s.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    section_label = None
    if i < len(lines) and HEADER_RE.match(lines[i]):
        raw_head = lines[i]
        section_label = "대안의 제안이유" if "대안" in raw_head else "제안이유 · 주요내용"
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    s = "\n\n".join(lines[i:]).strip()
    return s, section_label

# ---------------- HTML builder ----------------
def build_html(items: List[Dict[str, Any]]) -> str:
    rows = []
    for it in items:
        topic = pick(it, ["topic","topic_id","id"])
        tname = pick(it, ["topic_name","name","title"])
        if isinstance(tname, str): tname = tname.strip()
        top = it.get("top_law") or {}
        law_name = pick(top, ["lawName","name","law_name"])
        law_ratio = pick(top, ["law_ratio","ratio"])
        total_records = pick(it, ["total_records","record_count"])
        total_bill_mentions = pick(it, ["total_bill_mentions","bill_mentions_sum","total_mentions"])
        bills_full = norm_bills_full(it.get("bills", []), total_bill_mentions)
        kw = normalize_keywords(it.get("keywords"), limit=9999)
        rows.append({
            "topic": topic,
            "topic_name": tname,
            "keywords": kw,
            "top_law": law_name,
            "law_ratio": float(law_ratio) if law_ratio is not None else None,
            "total_records": total_records,
            "total_bill_mentions": total_bill_mentions,
            "bills_full": bills_full
        })

    df = pd.DataFrame(rows)

    # ---- 정렬: topic(숫자 오름) → law_ratio(내림) → total_records(내림)
    df["topic_num"] = pd.to_numeric(df["topic"], errors="coerce")
    df = df.sort_values(
        ["topic_num", "law_ratio", "total_records"],
        ascending=[True, False, False],
        na_position="last"
    )

    # ---- 헤더용 집계
    n_topics = int(len(df))
    total_records_sum = int(pd.to_numeric(df["total_records"], errors="coerce").fillna(0).sum())
    total_mentions_sum = int(pd.to_numeric(df["total_bill_mentions"], errors="coerce").fillna(0).sum())

    now_kr = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %H:%M')

    def render_kw_chips(kw_list: List[str], max_show: int = 8) -> Tuple[str, str]:
        if not kw_list:
            return "<span class='empty'>—</span>", ""
        shown = kw_list[:max_show]; hidden = kw_list[max_show:]
        chips = [f"<button class='kw-chip' title='이 키워드로 필터링' onclick='filterByKeyword(\"{html.escape(k)}\")'>{html.escape(k)}</button>" for k in shown]
        if hidden:
            chips.append(f"<span class='kw-more' title='{html.escape(', '.join(hidden))}'>+{len(hidden)}</span>")
        return "".join(chips), " ".join(kw_list)

    html_rows = []; rank = 1
    for _, r in df.iterrows():
        law_ratio = r.get("law_ratio")
        lr_width = f"{int(max(0,min(100, (law_ratio or 0)*100)))}%"

        # Top-3 chips
        chips = []
        for i, binfo in enumerate((r.get("bills_full") or [])[:3], start=1):
            label_full = (binfo.get("bill_name") or binfo.get("bill_id") or "").strip()
            label_short = label_full if len(label_full) <= 60 else label_full[:57] + "…"
            share = binfo.get("share"); width = f"{int(max(0,min(100,(share or 0)*100)))}%"
            cls = "gold" if i == 1 else ("silver" if i == 2 else "bronze")
            chips.append(f"""
            <div class="chip {cls}" title="{html.escape(label_full)}">
              <div class="chip-row">
                <span class="medal">#{i}</span>
                <span class="chip-label clamp-2">{html.escape(label_short)}</span>
                <span class="chip-pct">{pct(share)}</span>
              </div>
              <div class="chip-bar"><div style="width:{width}"></div></div>
            </div>""")
        chips_html = "".join(chips) if chips else "<span class='empty'>-</span>"

        # Keywords
        kw_list = r.get("keywords") or []
        kw_chips_col, kw_joined = render_kw_chips(kw_list, max_show=8)
        kw_chips_full, _ = render_kw_chips(kw_list, max_show=9999)

        # Top 5 cards
        cards=[]
        for j, b in enumerate((r.get("bills_full") or [])[:5], start=1):
            name = (b.get("bill_name") or b.get("bill_id") or "").strip()
            code = b.get("bill_id") or ""
            raw = (b.get("summary") or "").strip()
            summary_text, sec_label = process_summary(raw)
            has_sum = bool(summary_text)
            summary_html = html.escape(summary_text)
            share = b.get("share"); mentions = b.get("mentions") or ""
            avg_s = b.get("avg_score"); max_s = b.get("max_score")
            show_more = len(summary_text) > 320
            header_chip_html = f"<div class='chip-row top'><span class='sec-chip'>{sec_label}</span></div>" if sec_label else ""
            cards.append(f"""
            <div class="bill-card">
              <div class="bill-card-head">
                <span class="badge">Top {j}</span>
                <div class="bill-title clamp-2" title="{html.escape(name)}">{html.escape(name)}</div>
                <div class="bill-code">{html.escape(code)}</div>
              </div>
              <div class="bill-card-body">
                <div class="stats">
                  <span><span class="label">Share</span> <span class="val">{pct(share)}</span></span>
                  <span><span class="label">Mentions</span> <span class="val">{mentions}</span></span>
                  <span><span class="label">Avg</span> <span class="val">{'' if avg_s is None else f"{avg_s:.3f}"}</span></span>
                  <span><span class="label">Max</span> <span class="val">{'' if max_s is None else f"{max_s:.3f}"}</span></span>
                </div>
                {header_chip_html}
                <div class="summary clamp-6">{summary_html if has_sum else "<span class='empty'>요약 없음</span>"}</div>
                {f"<button class='more-btn' onclick='toggleSummary(this)'>더보기</button>" if (has_sum and show_more) else ""}
              </div>
            </div>""")
        cards_html = "".join(cards) if cards else "<div class='minor'>의안 데이터가 없습니다.</div>"

        row_cls = "row-gold" if rank == 1 else ("row-silver" if rank == 2 else ("row-bronze" if rank == 3 else ""))

        html_rows.append(f"""
        <tr class="{row_cls}" onclick="rowToggle(this, event)" data-keywords="{html.escape(kw_joined)}">
          <td class="rank">
            <button class="toggle" aria-label="세부 보기" title="세부 보기/닫기" onclick="toggleExpand(this); event.stopPropagation();">▾</button>
            <span class="rk">{rank}</span>
          </td>
          <td class="topic">{html.escape(str(r.get('topic','')))}</td>
          <td class="tname"><div class="tname-text clamp-2" title="{html.escape(str(r.get('topic_name','')))}">{html.escape(str(r.get('topic_name','')))}</div></td>
          <td class="kw">{kw_chips_col}</td>
          <td class="law">
            <div class="law-name clamp-1" title="{html.escape(str(r.get('top_law','')))}">{html.escape(str(r.get('top_law','')))}</div>
            <div class="bar"><div class="fill" style="width:{lr_width}"></div></div>
            <div class="law-meta">법률 점유율: {pct(law_ratio)} · 이 토픽의 의안 언급 중 해당 법률의 비율</div>
          </td>
          <td class="top3">{chips_html}</td>
          <td class="num" data-val="{r.get('total_records') or ''}">{r.get('total_records') or ""}<div class="minor">해당 토픽 문서 수</div></td>
          <td class="num" data-val="{r.get('total_bill_mentions') or ''}">{r.get('total_bill_mentions') or ""}<div class="minor">의안 언급 총합</div></td>
        </tr>
        <tr class="expand-row" style="display:none" data-open="0">
          <td colspan="8">
            <div class="expand-wrap cards">
              <div class="expand-title">토픽 키워드 & 매칭된 의안 <b>Top 5</b> 상세</div>
              <div class="kw-block">
                <div class="kw-title">Top Keywords</div>
                <div class="kw-list">{kw_chips_full if kw_chips_full else "<span class='empty'>키워드 없음</span>"}</div>
                <button class="kw-copy" onclick='copyKeywords(this)'>모두 복사</button>
              </div>
              <div class="card-grid">
                {cards_html}
              </div>
            </div>
          </td>
        </tr>""")
        rank += 1

    # -------- HTML 구조 --------
    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>뉴스 및 소셜 데이터 토픽 × 법률 의안 리더보드</title>
<style>
  :root {{
    --bg:#0b0d12; --panel:#121621; --muted:#8b93a7; --text:#e9ecf2;
    --border:#1b2030; --bar:#2b3347; --fill:#7aa2ff;
    --gold:#ffd166; --gold-dark:#e3b43f;
    --silver:#cfd8dc; --silver-dark:#a7b1b7;
    --bronze:#e6a57e; --bronze-dark:#c98760;
    --accent:#b06cff; --accent2:#6dd6ff;
    --stat-label:#ffd166;

    /* Section chip (Slate tone) */
    --sec-chip-fg:#c8d2e6;
    --sec-chip-brd:#5b6a86;
    --sec-chip-bg:rgba(120, 140, 170, .12);

    /* Hero colors */
    --hero-grad-1:#141a2a;
    --hero-grad-2:#0f1320;
    --hero-edge:#6dd6ff33;
  }}

  * {{ box-sizing: border-box; }}
  body {{ margin:0; background: var(--bg); color: var(--text);
         font-family: 'Inter','Noto Sans KR',system-ui,-apple-system,Roboto,Arial,sans-serif; }}

  /* ===== Hero (top) ===== */
  .hero {{
    position: relative;
    margin: 0 0 18px;
    padding: 28px 18px 26px;
    border-bottom: 1px solid var(--border);
    background:
      radial-gradient(1200px 420px at 10% -10%, #8fb6ff1a, transparent 60%),
      radial-gradient(1000px 500px at 90% -20%, #6dd6ff14, transparent 60%),
      linear-gradient(180deg, var(--hero-grad-1), var(--hero-grad-2));
    overflow: hidden;
  }}
  .hero::after {{
    /* subtle grid overlay */
    content:"";
    position:absolute; inset:0;
    background-image:
      linear-gradient(0deg, #ffffff08 1px, transparent 1px),
      linear-gradient(90deg, #ffffff07 1px, transparent 1px);
    background-size: 24px 24px, 24px 24px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,.8), rgba(0,0,0,.1));
    pointer-events:none;
  }}
  .wrap {{ max-width: 1280px; margin: 0 auto; }}
  .hero-inner {{ display:flex; flex-direction:column; gap:14px; position:relative; z-index:1; }}

  .hero-title {{
    font-size: 2rem; font-weight: 900; letter-spacing: .2px;
    display:flex; align-items:center; gap:10px;
  }}
  .hero-title .logo {{
    display:inline-flex; align-items:center; justify-content:center;
    min-width: 56px; height: 56px; padding: 0 16px; border-radius: 16px;
    background: radial-gradient(100% 100% at 30% 20%, #aee3ff, #7aa2ff 60%, #4a69d9 100%);
    box-shadow:
      0 0 0 1px #5f77ff55,
      0 10px 36px #6dd6ff30,
      inset 0 -8px 14px #00000030;
    color:#0b1020; font-weight:900; font-size: 22px; letter-spacing: .8px; text-transform: uppercase; line-height:1;
  }}
  @media (max-width:960px){{
    .hero-title .logo{{ min-width:48px; height:48px; padding:0 12px; font-size:20px; border-radius:12px; }}
  }}
  .hero-sub {{ color: var(--muted); font-size: .98rem; }}
  .hero-badges {{ display:flex; gap:10px; flex-wrap:wrap; }}
  .medal-pill {{
    display:inline-flex; align-items:center; gap:8px; padding:6px 10px;
    border-radius: 999px; border:1px solid var(--border); background:#121827cc;
    box-shadow: inset 0 -1px 0 #ffffff10, 0 6px 18px #00000025;
    font-weight:800; font-size:.92rem;
  }}
  .medal-pill .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
  .medal-pill.g .dot {{ background: var(--gold); box-shadow:0 0 8px #ffd16666; }}
  .medal-pill.s .dot {{ background: var(--silver); box-shadow:0 0 8px #cfd8dc66; }}
  .medal-pill.b .dot {{ background: var(--bronze); box-shadow:0 0 8px #e6a57e66; }}

  .stat-cards {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px; }}
  .stat-card {{
    border:1px solid var(--border); background:#101524cc; border-radius:14px; padding:12px 14px;
  }}
  .stat-label {{ color:var(--muted); font-size:.85rem; }}
  .stat-value {{ font-size:1.25rem; font-weight:900; letter-spacing:.2px; }}

  /* ===== Controls + Table ===== */
  .section-pad {{ padding: 0 18px 24px; }}

  .hint {{ color: var(--muted); margin-bottom: 14px; font-size: 0.95rem; }}

  .controls {{ display:flex; gap:12px; align-items:center; margin: 10px 0 18px; flex-wrap: wrap; }}
  input[type="text"] {{ padding:10px 12px; border-radius: 10px; border:1px solid var(--border);
                        background: var(--panel); color: var(--text); min-width: 320px; }}
  button {{ background: var(--accent); color: #fff; border: none; padding: 10px 14px; border-radius: 10px; cursor:pointer; }}

  table {{ width: 100%; border-collapse: collapse; background: var(--panel); border:1px solid var(--border);
           border-radius: 14px; overflow: hidden; table-layout: fixed; }}
  th, td {{ padding: 12px 14px; border-bottom:1px solid var(--border); vertical-align: middle; }}
  th {{ position: sticky; top: 0; background: #161b28; font-weight: 700; cursor: pointer; text-align: left; }}
  th .sub {{ display:block; font-weight: 400; font-size: 0.83rem; color: var(--muted); margin-top: 2px; }}
  tr:hover {{ background: #141a26; }}

  /* column widths (Topic ID hidden) */
  th:nth-child(1), td.rank {{ width: 88px; }}
  th:nth-child(2), td.topic {{ width: 76px; display:none !important; }}
  th:nth-child(3) {{ width: 22%; }}     /* Topic Name */
  th:nth-child(4) {{ width: 16%; }}     /* Keywords */
  th:nth-child(5) {{ width: 28%; }}     /* Top Law */
  th:nth-child(6) {{ width: 24%; }}     /* Top 3 Bills */
  th:nth-child(7), th:nth-child(8), td.num {{ width: 90px; }}

  td.rank {{ font-weight: 700; color: #c4ccff; white-space: nowrap; }}
  .rk {{ margin-left: 6px; }}
  .toggle {{ background:#223; border:1px solid #2a3250; color:#cfe; border-radius:8px; padding:4px 6px; font-size:0.8rem; }}

  .tname-text {{ white-space: normal; line-height: 1.3; }}
  .law-name {{ font-weight: 700; margin-bottom: 6px; }}

  .bar {{ height: 10px; background: var(--bar); border-radius: 999px; overflow: hidden; border:1px solid #22283a; }}
  .bar .fill {{ height: 100%; background: linear-gradient(90deg, var(--fill), #a6c1ff); }}
  .law-meta {{ margin-top:6px; font-size: 0.9rem; color: var(--muted); }}

  .top3 {{ min-width: 380px; }}
  .chip {{ padding: 8px 10px; border-radius: 12px; margin-bottom: 8px; background: #141a26; border:1px solid var(--border); }}
  .chip.gold {{ border-color: var(--gold-dark); background: linear-gradient(180deg, rgba(255,228,140,0.14), rgba(255,228,140,0.0)); }}
  .chip.silver {{ border-color: var(--silver-dark); background: linear-gradient(180deg, rgba(215,225,232,0.12), rgba(215,225,232,0.0)); }}
  .chip.bronze {{ border-color: var(--bronze-dark); background: linear-gradient(180deg, rgba(235,180,140,0.12), rgba(235,180,140,0.0)); }}
  .chip-row {{ display:flex; align-items:flex-start; gap:10px; margin-bottom:6px; }}
  .chip-row.top {{ margin: 4px 0 8px; }}

  .chip .medal {{ display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius: 8px; font-weight: 800; color:#000; margin-top: 1px; flex:0 0 26px; }}
  .chip.gold .medal {{ background: var(--gold); }}
  .chip.silver .medal {{ background: var(--silver); }}
  .chip.bronze .medal {{ background: var(--bronze); }}
  .chip-label {{ line-height: 1.3; }}
  .chip-pct {{ color:#dfe6ff; font-weight: 700; }}
  .chip-bar {{ height: 8px; background: var(--bar); border-radius: 999px; overflow: hidden; border:1px solid #232a3c; }}
  .chip-bar > div {{ height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); }}

  /* section chip — slate */
  .sec-chip {{
    display:inline-block; padding:6px 10px; border-radius:999px;
    border:1px solid var(--sec-chip-brd); color:var(--sec-chip-fg);
    background:var(--sec-chip-bg); font-weight:800; font-size:.92rem;
  }}

  .minor {{ color: var(--muted); font-size: 0.85rem; }}
  .empty {{ color: var(--muted); }}

  /* keyword chips */
  .kw {{ white-space: normal; line-height: 1.2; }}
  .kw-chip {{
    display:inline-block; margin: 4px 6px 4px 0; padding: 6px 10px; border-radius: 999px;
    border:1px solid #2a3250; background:#141a26; color:#dbe6ff; font-size: 0.88rem; cursor:pointer;
  }}
  .kw-chip:hover {{ background:#192137; }}
  .kw-more {{ color:#9fb3d9; font-size:0.85rem; margin-left:4px; }}

  .kw-block {{ margin-bottom: 14px; }}
  .kw-title {{ font-weight:700; margin-bottom:8px; }}
  .kw-list {{ margin-bottom: 8px; }}

  /* Row medals for top-3 topics */
  tr.row-gold td.rank .rk {{ color: var(--gold); }}
  tr.row-silver td.rank .rk {{ color: var(--silver); }}
  tr.row-bronze td.rank .rk {{ color: var(--bronze); }}

  /* Cards & Grid */
  .expand-wrap.cards {{ padding: 6px 2px 2px; }}
  .card-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; }}
  .bill-card {{ background:#141a26; border:1px solid var(--border); border-radius:16px; padding:14px; }}
  .bill-card-head {{ margin-bottom:8px; }}
  .badge {{
    display:inline-flex; align-items:center; justify-content:center;
    height:24px; padding:0 8px; border-radius:10px;
    background:#20263a; color:#cfe; font-weight:700; font-size:.85rem;
    border:1px solid #2a3250;
  }}
  .bill-title {{ font-weight:800; margin:6px 0 2px; line-height:1.25; }}
  .bill-code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); font-size: .85rem; }}

  .bill-card-body .stats {{ display:flex; flex-wrap:wrap; gap:18px; margin:6px 0 10px; align-items:baseline; }}
  .bill-card-body .stats .label {{ color: var(--stat-label); font-weight: 800; }}
  .bill-card-body .stats .val {{ font-weight: 700; color: var(--text); }}

  .summary {{ color:#dbe6ff; line-height:1.35; white-space: pre-line; }}
  .more-btn {{ margin-top:6px; padding:6px 10px; border-radius:8px; background:#20263a; color:#e6ecff; border:1px solid #2a3250; cursor:pointer; }}

  /* line clamps */
  .clamp-1 {{ display:-webkit-box; -webkit-line-clamp:1; -webkit-box-orient:vertical; overflow:hidden; }}
  .clamp-2 {{ display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }}
  .clamp-3 {{ display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }}
  .clamp-6 {{ display:-webkit-box; -webkit-line-clamp:6; -webkit-box-orient:vertical; overflow:hidden; }}
  .expanded {{ -webkit-line-clamp:unset !important; display:block !important; }}

  @media (max-width: 960px) {{
    .stat-cards {{ grid-template-columns: 1fr; }}
    .section-pad {{ padding: 0 12px 18px; }}
    .hero {{ padding: 22px 12px 20px; }}
    .card-grid {{ grid-template-columns: 1fr; }}
    .top3 {{ min-width: 320px; }}
    th:nth-child(3) {{ width: 38%; }}
    th:nth-child(4) {{ width: 0%; display:none; }}
  }}
</style>
</head>
<body>
  <div class="hero">
    <div class="wrap hero-inner">
      <div class="hero-title">
        <span class="logo">ISNLP</span>
        뉴스 및 소셜 데이터 토픽 × 법률 의안 리더보드
      </div>
      <div class="hero-sub">생성일시: {now_kr} (KST)</div>

      <div class="hero-badges">
        <span class="medal-pill g"><span class="dot"></span>#1 Gold</span>
        <span class="medal-pill s"><span class="dot"></span>#2 Silver</span>
        <span class="medal-pill b"><span class="dot"></span>#3 Bronze</span>
      </div>

      <div class="stat-cards">
        <div class="stat-card">
          <div class="stat-label">총 토픽 수</div>
          <div class="stat-value">{n_topics:,}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">문서 수 합계</div>
          <div class="stat-value">{total_records_sum:,}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">의안 언급 합계</div>
          <div class="stat-value">{total_mentions_sum:,}</div>
        </div>
      </div>
    </div>
  </div>

  <div class="wrap section-pad">
    <div class="hint">• 토픽 행을 <b>클릭</b>하면 <b>Top Keywords</b>와 <b>Top 5 의안</b> 카드가 펼쳐집니다. 키워드 칩 클릭 → 즉시 검색.</div>
    <div class="controls">
      <input id="filter" type="text" placeholder="검색: 토픽/키워드/법률/의안 이름·코드…" oninput="applyFilter()">
      <button onclick="resetFilter()">초기화</button>
    </div>

    <table id="tbl">
      <thead>
        <tr>
          <th onclick="sortTable(0,'num')">순위 <span class="sub">토픽 순위</span></th>
          <th onclick="sortTable(1,'num')">Topic <span class="sub">토픽 ID</span></th>
          <th onclick="sortTable(2,'str')">Topic Name <span class="sub">토픽 제목</span></th>
          <th onclick="sortTable(3,'str')">Keywords <span class="sub">상위 키워드</span></th>
          <th onclick="sortTable(4,'str')">Top Law · Law Ratio <span class="sub">최다 법률명 · 점유율</span></th>
          <th onclick="sortTable(5,'str')">Top 3 Bills (share) <span class="sub">의안 이름 · 점유율</span></th>
          <th onclick="sortTable(6,'num')">Records <span class="sub">문서 수</span></th>
          <th onclick="sortTable(7,'num')">Mentions <span class="sub">언급 총합</span></th>
        </tr>
      </thead>
      <tbody>
        {''.join(html_rows)}
      </tbody>
    </table>
  </div>

<script>
function sortTable(col, type) {{
  const table = document.getElementById('tbl');
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => !r.classList.contains('expand-row'));
  const pairs = [];
  for (let i=0; i<rows.length; i+=2) {{
    const row = rows[i];
    const exp = rows[i+1];
    let val = row.children[col].textContent.trim();
    if (type === 'num') {{
      val = parseFloat(row.children[col].getAttribute('data-val') || row.children[col].textContent) || 0;
    }} else {{
      val = val.toLowerCase();
    }}
    pairs.push([row, exp, val]);
  }}
  const asc = (table.getAttribute('data-sort-col') != col || table.getAttribute('data-sort-dir') === 'desc');
  pairs.sort((a,b) => (a[2] > b[2] ? 1 : a[2] < b[2] ? -1 : 0) * (asc ? 1 : -1));
  for (const [r,e] of pairs) {{ tbody.appendChild(r); tbody.appendChild(e); }}
  table.setAttribute('data-sort-col', col);
  table.setAttribute('data-sort-dir', asc ? 'asc' : 'desc');
}}

function applyFilter() {{
  const q = (document.getElementById('filter').value || '').toLowerCase();
  const tbody = document.getElementById('tbl').tBodies[0];
  const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => !r.classList.contains('expand-row'));
  for (let i=0; i<rows.length; i+=1) {{
    const row = rows[i];
    const cols = row.querySelectorAll('td');
    const tname = (cols[2]?.innerText || '').toLowerCase();
    const kw = (row.getAttribute('data-keywords') || '').toLowerCase();
    const law = (cols[4]?.innerText || '').toLowerCase();
    const top3 = (cols[5]?.innerText || '').toLowerCase();
    const show = !q || tname.includes(q) || kw.includes(q) || law.includes(q) || top3.includes(q);
    row.style.display = show ? '' : 'none';
    const exp = row.nextElementSibling;
    if (exp && exp.classList.contains('expand-row')) exp.style.display = show ? exp.style.display : 'none';
  }}
}}

function resetFilter() {{
  document.getElementById('filter').value = '';
  applyFilter();
}}

function rowToggle(tr, ev) {{
  const exp = tr.nextElementSibling;
  if (!exp || !exp.classList.contains('expand-row')) return;
  const opened = exp.style.display !== 'none';
  exp.style.display = opened ? 'none' : '';
}}

function toggleExpand(btn) {{
  const tr = btn.closest('tr');
  rowToggle(tr);
}}

function filterByKeyword(k) {{
  const inp = document.getElementById('filter');
  inp.value = k;
  applyFilter();
}}

function copyKeywords(btn) {{
  const block = btn.closest('.kw-block');
  if (!block) return;
  const list = block.querySelector('.kw-list');
  const chips = list ? Array.from(list.querySelectorAll('.kw-chip')).map(c => c.textContent.trim()) : [];
  const text = chips.join(', ');
  navigator.clipboard.writeText(text).then(() => {{
    btn.textContent = '복사됨!';
    setTimeout(() => btn.textContent = '모두 복사', 1200);
  }});
}}

function toggleSummary(btn) {{
  const card = btn.closest('.bill-card');
  const s = card.querySelector('.summary');
  if (!s) return;
  const expanded = s.classList.toggle('expanded');
  btn.textContent = expanded ? '접기' : '더보기';
}}
</script>
</body>
</html>"""
    return html_doc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path") # 입력 경로(집계 결과)
    ap.add_argument("--out", dest="out_html") # 출력 경로
    args = ap.parse_args()

    assert os.path.exists(args.in_path), f"Missing {args.in_path}"
    with open(args.in_path, encoding="utf-8") as f:
        items = json.load(f)
    if isinstance(items, dict):
        items = [items]

    html_str = build_html(items)
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"Saved: {args.out_html}")

if __name__ == "__main__":
    main()
