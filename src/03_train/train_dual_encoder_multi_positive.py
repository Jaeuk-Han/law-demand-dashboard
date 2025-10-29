import os, json, math, random, argparse, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoModel, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit


class DualDatasetBills1toMany(Dataset):
    def __init__(self, news_path, bill_groups_path, seed=42, use_title=True):
        random.seed(seed)
        self.use_title = use_title

        with open(news_path, encoding="utf-8") as f:
            self.news = json.loads(f.read())

        with open(bill_groups_path, encoding="utf-8") as f:
            groups = json.loads(f.read())

        self.law2summaries = {}
        for g in groups:
            law = g.get("lawName")
            if not law:
                continue
            summaries = [
                (b.get("summary") or "").strip()
                for b in g.get("bills", [])
                if (b.get("summary") or "").strip()
            ]
            if summaries:
                self.law2summaries[law] = summaries

        self.items = []
        for ex in self.news:
            law = ex.get("_bucket_law")
            if law in self.law2summaries:
                title = ex.get("title", "")
                content = ex.get("content", "")
                q = f"{title} [SEP] {content}" if self.use_title else content
                self.items.append((q, law))

        assert len(self.items) > 0, "매칭되는 뉴스–법률 쌍이 없습니다."

        self.laws = sorted(set(l for _, l in self.items))
        self.law2id = {l: i for i, l in enumerate(self.laws)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class MultiPositiveCollator:
    def __init__(self, law2summaries, law2id, k_pos=4):
        self.law2summaries = law2summaries
        self.law2id = law2id
        self.k_pos = k_pos

    def __call__(self, batch):
        queries, query_law_ids = [], []
        docs, doc_law_ids = [], []

        for q, law in batch:
            queries.append(q)
            query_law_ids.append(self.law2id[law])

            poss = self.law2summaries[law]
            if len(poss) <= self.k_pos:
                picks = list(range(len(poss)))
            else:
                picks = random.sample(range(len(poss)), k=self.k_pos)

            for idx in picks:
                docs.append(poss[idx])
                doc_law_ids.append(self.law2id[law])

        return {
            "queries": queries,
            "docs": docs,
            "query_law_ids": query_law_ids,
            "doc_law_ids": doc_law_ids,
        }


class Qwen3EmbeddingDualEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B", proj_dim=1024, lora_r=8):
        super().__init__()
        print(f"[Init] Loading {model_name} (8bit + LoRA r={lora_r})")

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            llm_int8_threshold=6.0,
        )

        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            print("[Warn] FlashAttention2 비활성. 기본 어텐션 사용.")
            self.encoder = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

        self.encoder.gradient_checkpointing_enable()
        self.encoder.enable_input_require_grads()

        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        self.norm = F.normalize
        self.tau = nn.Parameter(torch.tensor(0.05))

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encode(self, texts, device):
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt",
        ).to(device)

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
        return emb

    @torch.no_grad()
    def embed_texts(self, texts, device):
        self.eval()
        return self.encode(texts, device)

    def forward_multi_positive(self, batch, device):
        queries = batch["queries"]
        docs = batch["docs"]
        qlaw = torch.tensor(batch["query_law_ids"], device=device, dtype=torch.long)
        dlaw = torch.tensor(batch["doc_law_ids"], device=device, dtype=torch.long)

        q = self.encode(queries, device)

        uniq_docs, inv_map, seen = [], [], {}
        for dtxt in docs:
            if dtxt in seen:
                inv_map.append(seen[dtxt])
            else:
                idx = len(uniq_docs)
                seen[dtxt] = idx
                uniq_docs.append(dtxt)
                inv_map.append(idx)

        if len(uniq_docs) == len(docs):
            d = self.encode(docs, device)
        else:
            d_uniq = self.encode(uniq_docs, device)
            inv = torch.tensor(inv_map, device=device, dtype=torch.long)
            d = d_uniq.index_select(0, inv)

        logits = (q @ d.T) / self.tau.clamp(min=1e-3)
        pos_mask = (qlaw[:, None] == dlaw[None, :])
        assert pos_mask.any(dim=1).all().item(), "어떤 쿼리에도 양성이 없습니다."

        lse_all = torch.logsumexp(logits, dim=1)
        neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
        logits_pos_only = torch.where(pos_mask, logits, neg_inf)
        lse_pos = torch.logsumexp(logits_pos_only, dim=1)
        loss = -(lse_pos - lse_all).mean()

        with torch.no_grad():
            sims_pos = logits[pos_mask].mean().item()
            probs = torch.softmax(logits, dim=1)
            top1 = probs.topk(k=1, dim=1).indices
            top5 = probs.topk(k=5, dim=1).indices
            r_at1 = pos_mask.gather(1, top1).any(dim=1).float().mean().item()
            r_at5 = pos_mask.gather(1, top5).any(dim=1).float().mean().item()

        metrics = {"pos_sim_mean": sims_pos, "r@1": r_at1, "r@5": r_at5}
        return loss, metrics


def train_loop(args):
    import wandb

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    try:
        wandb.login(key=args.wandb_key)
    except Exception as e:
        print(f"[W&B] login skipped or failed: {e}")
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    try:
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
    except TypeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3EmbeddingDualEncoder(
        model_name=args.model_name,
        proj_dim=args.proj_dim,
        lora_r=args.lora_r,
    ).to(device)

    train_ds = DualDatasetBills1toMany(args.train_path, args.bill_groups_path, use_title=not args.no_title)
    collate_fn = MultiPositiveCollator(train_ds.law2summaries, train_ds.law2id, k_pos=args.k_pos)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    optim = AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_updates_per_epoch = math.ceil(len(train_dl) / max(1, args.grad_accum_steps))
    total_updates = num_updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * 0.1)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_updates)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs(args.out_dir, exist_ok=True)
    global_step, running_loss = 0, 0.0
    model.train()

    for ep in range(args.epochs):
        epoch_losses = []
        pbar = tqdm(train_dl, desc=f"Epoch {ep+1}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss, metrics = model.forward_multi_positive(batch, device)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            epoch_losses.append(loss.item() * args.grad_accum_steps)
            running_loss += loss.item()

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                sched.step()

                if global_step % args.log_every == 0:
                    window_avg = running_loss / args.log_every
                    wandb.log({
                        "train/step": global_step,
                        "train/loss_step": window_avg,
                        "train/lr": sched.get_last_lr()[0],
                        "train/tau": float(model.tau.detach().cpu()),
                        "train/pos_sim_mean": metrics.get("pos_sim_mean", float("nan")),
                        "train/r@1": metrics.get("r@1", float("nan")),
                        "train/r@5": metrics.get("r@5", float("nan")),
                    }, step=global_step)
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.out_dir, f"step{global_step:06d}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.encoder.save_pretrained(ckpt_dir)
                    model.tokenizer.save_pretrained(ckpt_dir)
                    torch.save(
                        {"proj": model.proj.state_dict(), "tau": model.tau.detach().cpu()},
                        os.path.join(ckpt_dir, "extra.pt"),
                    )
                    print(f"[Checkpoint] Saved at step {global_step} → {ckpt_dir}")

            pbar.set_postfix({"loss": f"{(sum(epoch_losses)/len(epoch_losses)):.4f}"})

        epoch_avg = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        wandb.log({
            "train/step": global_step,
            "train/epoch": ep + 1,
            "train/loss_epoch": epoch_avg,
            "train/tau": float(model.tau.detach().cpu()),
        }, step=global_step)

        save_dir = os.path.join(args.out_dir, f"epoch{ep+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.encoder.save_pretrained(save_dir)
        model.tokenizer.save_pretrained(save_dir)
        torch.save(
            {"proj": model.proj.state_dict(), "tau": model.tau.detach().cpu()},
            os.path.join(save_dir, "extra.pt"),
        )

    wandb.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", default="data/news/train.json")
    ap.add_argument("--bill_groups_path", default="data/bill/bill_groups.json")
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--k_pos", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--no_title", action="store_true")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", default="outputs/qwen3_embedding_dual_bills_mpos")
    ap.add_argument("--wandb_project", default="law-demand-analysis_2025")
    ap.add_argument("--wandb_key", default="YOUR_WANDB_KEY")
    ap.add_argument("--run_name", default="dual_encoder_qwen3_bills_mpos")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=1000, help="몇 step마다 체크포인트 저장할지 설정 (기본 1000)")
    args = ap.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_loop(args)
