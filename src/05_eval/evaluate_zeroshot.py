# -*- coding: utf-8 -*-
"""
Zero-shot retrieval evaluation aligned with dual-encoder evaluation logic.

- 동일한 입출력 인터페이스:
    --test_path, --bill_groups_path, --out_dir, --use_title,
    --agg [doc|law_max|law_mean], --batch_size, --max_length, --topk, --no_int8
- 동일한 전처리/집계 로직:
    * 질의: title [SEP] content (옵션)
    * 후보 문서: bill_groups.json의 bills[].summary 만 사용 (듀얼 인코더와 동일)
    * 유사도: L2 정규화 임베딩의 점곱 (cosine)
    * 집계: doc (문서 단위), law_max / law_mean (법 단위 집계)
- 동일한 메트릭/출력 형식:
    * metrics_{agg}.json (Recall@1/5/10, MRR, MAP)
    * predictions_{agg}.jsonl (질문별 Top-K 리스트)
"""

import os, json, argparse, sys
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


# ------------------------------
# Data loading
# ------------------------------
def load_test_queries(path: str, use_title: bool = True) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for ex in data:
        title = (ex.get("title") or "").strip()
        content = (ex.get("content") or "").strip()
        qtxt = f"{title} [SEP] {content}" if use_title else content
        law = ex.get("_bucket_law")  # gold label
        if qtxt and law:
            queries.append({"text": qtxt, "gold_law": law, "raw": ex})
    if not queries:
        raise ValueError("No valid queries with _bucket_law found in test set.")
    return queries


def load_candidate_docs_from_bill_groups(path: str):
    """
    듀얼 인코더 코드와 동일하게 bills[].summary 만 사용한다.
    (제로샷 기존 코드에서는 billName + summary를 합쳤지만, 로직 일치를 위해 summary만 사용)
    """
    with open(path, encoding="utf-8") as f:
        groups = json.load(f)

    docs = []
    law_to_indices = defaultdict(list)
    for g in groups:
        law = g.get("lawName")
        if not law:
            continue
        bills = g.get("bills", [])
        for b in bills:
            txt = (b.get("summary") or "").strip()
            if not txt:
                continue
            idx = len(docs)
            docs.append({
                "text": txt,
                "lawName": law,
                "billId": b.get("billId"),
                "billName": b.get("billName"),
            })
            law_to_indices[law].append(idx)
    if not docs:
        raise ValueError("No candidate summaries found in bill_groups.")
    return docs, law_to_indices


# ------------------------------
# Model (제로샷용: 프로젝션/온도 없이 base encoder mean-pool)
# ------------------------------
class ZeroShotEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device, use_8bit: bool = True):
        super().__init__()
        self.device = device

        q_config = None
        if use_8bit:
            q_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8",
                llm_int8_threshold=6.0,
            )

        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                quantization_config=q_config,
                device_map="auto" if use_8bit else None,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if use_8bit else None,
            )
        except Exception:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                quantization_config=q_config,
                device_map="auto" if use_8bit else None,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.norm = F.normalize
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.eval()

    @torch.inference_mode()
    def embed_texts(self, texts: List[str], batch_size: int = 16, max_length: int = 160) -> torch.Tensor:
        outs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
            chunk = texts[i:i+batch_size]
            toks = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}

            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                out = self.encoder(**toks)
                if isinstance(out, torch.Tensor):
                    emb = out
                elif hasattr(out, "last_hidden_state"):
                    # 단순 mean-pooling (듀얼 인코더 평가의 평균 풀링과 대응)
                    emb = out.last_hidden_state.mean(dim=1)
                else:
                    emb = out[0]
                emb = self.norm(emb, dim=-1)
            outs.append(emb.to(self.device))
        return torch.cat(outs, dim=0) if outs else torch.empty(0, device=self.device)


# ------------------------------
# Scoring / Aggregation / Metrics (듀얼 인코더 평가 로직과 동일)
# ------------------------------
def compute_similarity_matrix(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
    return q_emb @ d_emb.T  # [Q, D], L2 정규화 임베딩이므로 cosine과 동일

def rank_by_aggregation(sim: torch.Tensor, law_to_indices: Dict[str, List[int]], agg: str):
    laws_list = sorted(law_to_indices.keys())
    Q, D = sim.shape
    L = len(laws_list)
    scores = torch.empty((Q, L), device=sim.device)

    for j, law in enumerate(laws_list):
        idxs = torch.tensor(law_to_indices[law], device=sim.device, dtype=torch.long)
        if agg == "law_max":
            scores[:, j], _ = sim[:, idxs].max(dim=1)
        elif agg == "law_mean":
            scores[:, j] = sim[:, idxs].mean(dim=1)
        else:
            raise ValueError(f"Unknown agg {agg}")
    return scores, laws_list

def recall_at_k(ranks: List[int], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

def mean_reciprocal_rank(ranks: List[int]) -> float:
    vals = [1.0 / r for r in ranks if r is not None and r > 0]
    return sum(vals) / len(ranks) if vals else 0.0

def average_precision_doc_level(pred_indices: List[int], positive_set: set) -> float:
    hit, s = 0, 0.0
    for i, idx in enumerate(pred_indices, start=1):
        if idx in positive_set:
            hit += 1
            s += hit / i
    if not positive_set:
        return 0.0
    return s / len(positive_set)

def evaluate_doc_level(sim: torch.Tensor, law_to_indices, query_gold_laws: List[str], topk: int = 10):
    Q, D = sim.shape
    results = []
    ranks_first = []
    APs = []

    doc_law = {}
    for law, idxs in law_to_indices.items():
        for idx in idxs:
            doc_law[idx] = law

    for q in range(Q):
        row = sim[q]
        order = torch.argsort(row, descending=True).tolist()  # doc indices
        gold = query_gold_laws[q]
        positives = set(law_to_indices.get(gold, []))

        # first positive rank
        first_rank = None
        for i, idx in enumerate(order, start=1):
            if idx in positives:
                first_rank = i
                break
        ranks_first.append(first_rank)

        # AP
        APs.append(average_precision_doc_level(order, positives))

        # topK display
        top_items = []
        for i, idx in enumerate(order[:topk], start=1):
            top_items.append({
                "rank": i,
                "doc_index": idx,
                "score": float(row[idx].item()),
                "lawName": doc_law[idx],
            })
        results.append(top_items)

    metrics = {
        "recall@1": recall_at_k(ranks_first, 1),
        "recall@5": recall_at_k(ranks_first, 5),
        "recall@10": recall_at_k(ranks_first, 10),
        "MRR": mean_reciprocal_rank(ranks_first),
        "MAP": sum(APs) / len(APs) if APs else 0.0,
    }
    return metrics, results, ranks_first

def evaluate_law_level(scores: torch.Tensor, laws_list: List[str], query_gold_laws: List[str], topk: int = 10):
    Q, L = scores.shape
    law2idx = {l: i for i, l in enumerate(laws_list)}

    results = []
    ranks = []
    APs = []

    for q in range(Q):
        row = scores[q]
        order = torch.argsort(row, descending=True).tolist()  # law indices
        gold = query_gold_laws[q]
        gold_idx = law2idx.get(gold, None)
        first_rank = None
        if gold_idx is not None:
            for i, j in enumerate(order, start=1):
                if j == gold_idx:
                    first_rank = i
                    break
        ranks.append(first_rank)
        APs.append(1.0 / first_rank if (first_rank is not None and first_rank > 0) else 0.0)

        top_items = []
        for i, j in enumerate(order[:topk], start=1):
            top_items.append({
                "rank": i,
                "lawName": laws_list[j],
                "score": float(row[j].item()),
            })
        results.append(top_items)

    metrics = {
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "MRR": mean_reciprocal_rank(ranks),
        "MAP": sum(APs) / len(APs) if APs else 0.0,
    }
    return metrics, results, ranks


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", required=True, help="JSON with title/content/_bucket_law")
    ap.add_argument("--bill_groups_path", required=True, help="JSON with lawName and bills[].summary")
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-8B", help="HF model name for zero-shot")
    ap.add_argument("--out_dir", required=True, help="Where to write metrics and predictions")
    ap.add_argument("--use_title", action="store_true", help="Use 'title [SEP] content' (default: content only)")
    ap.add_argument("--agg", default="doc", choices=["doc", "law_max", "law_mean"], help="Ranking granularity")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=160)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--no_int8", action="store_true", help="Disable 8bit inference")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroShotEncoder(args.model_name, device, use_8bit=not args.no_int8).to(device)

    # load data
    queries = load_test_queries(args.test_path, use_title=args.use_title)
    docs, law_to_indices = load_candidate_docs_from_bill_groups(args.bill_groups_path)

    q_texts = [q["text"] for q in queries]
    d_texts = [d["text"] for d in docs]

    # embed
    q_emb = model.embed_texts(q_texts, batch_size=args.batch_size, max_length=args.max_length)
    d_emb = model.embed_texts(d_texts, batch_size=args.batch_size, max_length=args.max_length)

    # similarity
    sim = compute_similarity_matrix(q_emb, d_emb)

    gold_laws = [q["gold_law"] for q in queries]

    if args.agg == "doc":
        metrics, results, ranks = evaluate_doc_level(sim, law_to_indices, gold_laws, topk=args.topk)
    else:
        scores, laws_list = rank_by_aggregation(sim, law_to_indices, agg=args.agg)
        metrics, results, ranks = evaluate_law_level(scores, laws_list, gold_laws, topk=args.topk)

    met_path = os.path.join(args.out_dir, f"metrics_{args.agg}.json")
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_path = os.path.join(args.out_dir, f"predictions_{args.agg}.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for q, tops in zip(queries, results):
            rec = {
                "query_title": q["raw"].get("title"),
                "query_date": q["raw"].get("date"),
                "gold_law": q["gold_law"],
                "agg": args.agg,
                "topk": tops,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[DONE] Saved:")
    print(" -", met_path)
    print(" -", pred_path)
    print("Metrics:", json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.exit(main())
