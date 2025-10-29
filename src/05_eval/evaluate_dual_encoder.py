import os, json, math, argparse, sys
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class Qwen3EmbeddingDualEncoderEval(nn.Module):
    def __init__(self, ckpt_dir: str, device: torch.device, use_8bit: bool = True):
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
                ckpt_dir,
                quantization_config=q_config,
                device_map="auto" if use_8bit else None,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if use_8bit else None,
            )
        except Exception:
            self.encoder = AutoModel.from_pretrained(
                ckpt_dir,
                quantization_config=q_config,
                device_map="auto" if use_8bit else None,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)

        extra_path = os.path.join(ckpt_dir, "extra.pt")
        if not os.path.exists(extra_path):
            raise FileNotFoundError(f"extra.pt not found in {ckpt_dir}.")

        extra = torch.load(extra_path, map_location="cpu")
        hidden = getattr(self.encoder.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.encoder.config, "hidden_dimension", 1024)

        self.proj = nn.Linear(hidden, extra.get("proj").get("weight").shape[0], bias=False)
        self.proj.load_state_dict(extra["proj"])
        self.tau = nn.Parameter(extra["tau"])

        self.norm = F.normalize
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.eval()

    @torch.inference_mode()
    def embed_texts(self, texts: List[str], batch_size: int = 16, max_length: int = 160) -> torch.Tensor:
        """Return L2-normalized embeddings [N, dim] on self.device."""
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
                    emb = out.last_hidden_state.mean(dim=1)
                else:
                    emb = out[0]
                emb = self.proj(emb)
                emb = self.norm(emb, dim=-1)
            outs.append(emb.to(self.device))
        return torch.cat(outs, dim=0) if outs else torch.empty(0, device=self.device)

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

def compute_similarity_matrix(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
    # both already L2-normalized; dot product == cosine similarity
    return q_emb @ d_emb.T  # [Q, D]

def rank_by_aggregation(sim: torch.Tensor, law_to_indices: Dict[str, List[int]], agg: str):
    laws_list = sorted(law_to_indices.keys())
    Q, D = sim.shape
    L = len(laws_list)
    scores = torch.empty((Q, L), device=sim.device)

    for j, law in enumerate(laws_list):
        idxs = torch.tensor(law_to_indices[law], device=sim.device, dtype=torch.long)
        # sim[:, idxs]: [Q, num_docs_for_law]
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

    # Precompute doc -> law
    doc_law = {}
    for law, idxs in law_to_indices.items():
        for idx in idxs:
            doc_law[idx] = law

    for q in range(Q):
        # sort docs by sim desc
        row = sim[q]
        order = torch.argsort(row, descending=True).tolist()  # list of doc idx
        gold = query_gold_laws[q]
        positives = set(law_to_indices.get(gold, []))

        # find first rank among positives
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
    APs = []  # in single positive per query, AP == precision@rank

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
        # AP degenerates to precision@rank for single positive
        APs.append(1.0 / first_rank if (first_rank is not None and first_rank > 0) else 0.0)

        # topK display
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", required=True, help="JSON with title/content/_bucket_law")
    ap.add_argument("--bill_groups_path", required=True, help="JSON with lawName and bills[].summary")
    ap.add_argument("--ckpt_dir", required=True, help="Directory containing the saved encoder + tokenizer + extra.pt (e.g., outputs/.../epoch3)")
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
    model = Qwen3EmbeddingDualEncoderEval(args.ckpt_dir, device, use_8bit=not args.no_int8).to(device)

    # load data
    queries = load_test_queries(args.test_path, use_title=args.use_title)
    docs, law_to_indices = load_candidate_docs_from_bill_groups(args.bill_groups_path)

    q_texts = [q["text"] for q in queries]
    d_texts = [d["text"] for d in docs]

    # embed
    q_emb = model.embed_texts(q_texts, batch_size=args.batch_size, max_length=args.max_length)
    d_emb = model.embed_texts(d_texts, batch_size=args.batch_size, max_length=args.max_length)

    # scores
    sim = q_emb @ d_emb.T  # [Q, D]

    gold_laws = [q["gold_law"] for q in queries]

    if args.agg == "doc":
        metrics, results, ranks = evaluate_doc_level(sim, law_to_indices, gold_laws, topk=args.topk)
    else:
        scores, laws_list = rank_by_aggregation(sim, law_to_indices, agg=args.agg)
        metrics, results, ranks = evaluate_law_level(scores, laws_list, gold_laws, topk=args.topk)

    # save outputs
    # (1) metrics
    met_path = os.path.join(args.out_dir, f"metrics_{args.agg}.json")
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # (2) per-query predictions
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
    sys.exit(main())
