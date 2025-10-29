import os, json, argparse, sys
from typing import List, Dict, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

class DualEncoderInference(nn.Module):
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
            raise FileNotFoundError(f"extra.pt not found in {ckpt_dir}")
        extra = torch.load(extra_path, map_location="cpu")

        hidden = getattr(self.encoder.config, "hidden_size", None) \
                 or getattr(self.encoder.config, "hidden_dimension", 1024)
        out_dim = extra["proj"]["weight"].shape[0]
        self.proj = nn.Linear(hidden, out_dim, bias=False)
        self.proj.load_state_dict(extra["proj"])

        self.tau = nn.Parameter(extra["tau"])
        self.norm = F.normalize
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.eval()

    @torch.inference_mode()
    def embed_texts(self, texts: List[str], batch_size: int = 32, max_length: int = 160) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 1, device=self.device)
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
            outs.append(emb)
        return torch.cat(outs, dim=0)

def load_json_any(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def make_query_text(rec: Dict[str, Any]) -> str:
    topic_name = (rec.get("topic_name") or "").strip()
    title = (rec.get("title") or "") or ""
    title = title.strip() if isinstance(title, str) else ""
    cleaned = (rec.get("cleaned_text") or "").strip()
    parts = [p for p in (topic_name, title, cleaned) if p]
    return " [SEP] ".join(parts) if parts else ""

def load_candidate_docs_from_bill_groups(path: str):
    with open(path, encoding="utf-8") as f:
        groups = json.load(f)

    docs = []
    for g in groups:
        law = g.get("lawName")
        if not law:
            continue
        for b in g.get("bills", []):
            txt = (b.get("summary") or "").strip()
            if not txt:
                continue
            docs.append({
                "text": txt,
                "lawName": law,
                "summary": txt,
                "billId": b.get("billId"),
                "billName": b.get("billName"),
            })
    if not docs:
        raise ValueError("No candidate bill summaries found.")
    return docs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, default="data/final_combined", help="뉴스/SNS JSON(list or single object)")
    ap.add_argument("--bill_groups_path",  required=True,  default="data/final_combined", help="법률별 bills[].summary 포함 JSON")
    ap.add_argument("--ckpt_dir", required=True, help="인퍼런스에 사용할 dual-encoder 체크포인트 디렉토리(extra.pt 포함)")
    ap.add_argument("--out_dir", required=True, help="출력 디렉토리 (생성됨)")
    ap.add_argument("--out_file", default="enriched.json", help="출력 파일명")
    ap.add_argument("--out_key_name", default="bills", help="붙일 키 이름 (예: bills / biils)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=160)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--no_int8", action="store_true", help="8bit 비활성화")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderInference(args.ckpt_dir, device, use_8bit=not args.no_int8).to(device)

    inputs = load_json_any(args.input_path)
    docs = load_candidate_docs_from_bill_groups(args.bill_groups_path)

    query_texts = [make_query_text(r) for r in inputs]
    valid_idx = [i for i, q in enumerate(query_texts) if q]

    d_texts = [d["text"] for d in docs]
    d_emb = model.embed_texts(d_texts, batch_size=args.batch_size, max_length=args.max_length)

    enriched_pairs = []

    if valid_idx:
        q_emb = model.embed_texts([query_texts[i] for i in valid_idx],
                                  batch_size=args.batch_size, max_length=args.max_length)
        sim = q_emb @ d_emb.T 

        for local_i, global_i in enumerate(valid_idx):
            row = sim[local_i]
            order = torch.argsort(row, descending=True)[:args.topk].tolist()
            tops = [{
                "billId":   docs[j]["billId"],
                "billName": docs[j]["billName"],
                "lawName":  docs[j]["lawName"],
                "score":    float(row[j].item()),
                "summary":  docs[j]["summary"],
            } for j in order]

            rec = dict(inputs[global_i])
            rec[args.out_key_name] = tops
            enriched_pairs.append((global_i, rec))

    empty_idx = set(range(len(inputs))) - set(valid_idx)
    for i in empty_idx:
        rec = dict(inputs[i])
        rec[args.out_key_name] = []
        enriched_pairs.append((i, rec))

    enriched_sorted = [r for _, r in sorted(enriched_pairs, key=lambda x: x[0])]
    out_path = os.path.join(args.out_dir, args.out_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enriched_sorted, f, ensure_ascii=False, indent=2)

    print("[DONE] Saved:", out_path)

if __name__ == "__main__":
    sys.exit(main())
