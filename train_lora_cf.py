import argparse, os, json, math
from typing import List, Dict
from collections import Counter

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.utils import format_prompt, LABELS

LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def norm_gender(g):
    if g is None:
        return None
    if isinstance(g, str):
        gg = g.strip().lower()
        if gg in ["male", "m", "man", "boy"]:
            return "male"
        if gg in ["female", "f", "woman", "girl"]:
            return "female"
        return gg
    return g

class PairDataset(Dataset):
    """
    Each item is a male/female counterfactual pair with same pair_id.
    """
    def __init__(self, rows):
        by_pair: Dict[str, List[dict]] = {}
        for r in rows:
            pid = r.get("pair_id")
            if pid is None:
                continue
            by_pair.setdefault(pid, []).append(r)

        self.pairs = []
        self.skipped = 0

        for pid, items in by_pair.items():
            if len(items) != 2:
                self.skipped += 1
                continue

            a, b = items[0], items[1]
            ga, gb = norm_gender(a.get("gender")), norm_gender(b.get("gender"))

            if ga == "male" and gb == "female":
                male, female = a, b
            elif gb == "male" and ga == "female":
                male, female = b, a
            else:
                self.skipped += 1
                continue

            lab = male.get("label")
            if lab not in LABEL2IDX:
                self.skipped += 1
                continue

            self.pairs.append((male, female))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        male, female = self.pairs[idx]
        return {
            "male_prompt": format_prompt(male["job_desc"], male["resume"]),
            "female_prompt": format_prompt(female["job_desc"], female["resume"]),
            "label": male["label"],
        }

def build_label_ids(tok):
    """
    Tokenize labels with leading space so they behave like next-token classification.
    Keep full token list (might be multi-token).
    """
    out = []
    for lab in LABELS:
        ids = tok(" " + lab, add_special_tokens=False).input_ids
        if len(ids) == 0:
            raise RuntimeError(f"Label '{lab}' tokenized to empty ids.")
        out.append(ids)
    return out  #List[List[int]]

def pad_2d(seqs: List[List[int]], pad_id: int):
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    return ids, attn

def label_logps_full(model, tok, prompts: List[str], label_ids_list: List[List[int]], max_len: int):
    """
    Return logps: Tensor [N, K]
    Each logp is sum of token log-probs of full label sequence given prompt.
    Does K forward passes (K=3), batched across prompts.
    """
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id
    if pad_id is None:
        #safety
        pad_id = tok.eos_token_id
        if pad_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id and no eos_token_id.")

    prompt_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]

    K = len(label_ids_list)
    N = len(prompts)
    logps = torch.zeros((N, K), device=device, dtype=torch.float32)

    for k, lab_ids in enumerate(label_ids_list):
        lab_len = len(lab_ids)

        clipped = []
        prompt_lens = []
        for ids in prompt_ids:
            if len(ids) + lab_len > max_len:
                ids = ids[: max(0, max_len - lab_len)]
            clipped.append(ids)
            prompt_lens.append(len(ids))

        seqs = [clipped[i] + lab_ids for i in range(N)]
        input_ids, attn = pad_2d(seqs, pad_id)
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [N, T, V]

        lp = torch.zeros((N,), device=device, dtype=torch.float32)
        for i in range(N):
            start = prompt_lens[i]
            #label token at position pos = start + j, uses logits at pos-1
            for j, tok_id in enumerate(lab_ids):
                pos = start + j
                dist = torch.log_softmax(logits[i, pos - 1, :], dim=-1)
                lp[i] = lp[i] + dist[tok_id]
        logps[:, k] = lp

    return logps  # [N, K]

def sym_kl(logp_p, logp_q):
    p = torch.exp(logp_p)
    q = torch.exp(logp_q)
    kl_pq = torch.sum(p * (logp_p - logp_q), dim=-1)
    kl_qp = torch.sum(q * (logp_q - logp_p), dim=-1)
    return 0.5 * (kl_pq + kl_qp)

def weighted_ce_from_logdist(logdist: torch.Tensor, y: torch.Tensor, class_w: torch.Tensor, label_smoothing: float = 0.0):
    """
    logdist: [B,K] log softmax over K
    y: [B] target class index in [0..K-1]
    class_w: [K]
    label_smoothing: in [0,1)
    """
    B, K = logdist.shape
    if label_smoothing <= 0:
        #standard NLL with weight
        nll = -logdist[torch.arange(B, device=logdist.device), y]
        w = class_w.index_select(0, y)
        return (nll * w).mean()

    #smoothed target distribution
    with torch.no_grad():
        tgt = torch.full((B, K), fill_value=label_smoothing / (K - 1), device=logdist.device, dtype=logdist.dtype)
        tgt.scatter_(1, y.view(-1, 1), 1.0 - label_smoothing)

    #per-sample weight by true class
    w = class_w.index_select(0, y).view(-1, 1)  #[B,1]
    ce = -(tgt * logdist).sum(dim=-1, keepdim=True)  #[B,1]
    return (ce * w).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--base_ckpt", required=True, help="LoRA adapter dir from src.train_lora output (runs/xxx)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_pairs", type=int, default=2)
    ap.add_argument("--lambda_cf", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_accum", type=int, default=1)

    #new knobs to fight collapse
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    #CF schedule: warm up CE then add CF
    ap.add_argument("--cf_warmup_steps", type=int, default=0,
                    help="first N optimizer steps use lambda_cf=0, then turn on CF")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_cfg(args.config)
    train_file = cfg["data"]["train_file"]
    max_len = int(cfg["model"]["max_length"])
    base_model = cfg["model"]["base_model"]

    rows = load_jsonl(train_file)
    ds = PairDataset(rows)
    if len(ds) == 0:
        raise RuntimeError(f"PairDataset is empty. Check genders/pair_id in {train_file}. Skipped={ds.skipped}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #label dist + class weights (inverse freq)
    cnt = Counter([r.get("label") for r in rows if r.get("label") in LABEL2IDX])
    total = sum(cnt.values())
    w = []
    for lab in LABELS:
        w.append(total / max(cnt.get(lab, 1), 1))
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    class_w = class_w / class_w.mean()
    print("train label counter:", dict(cnt), "class_weight(normed):", class_w.detach().cpu().tolist())

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    model = PeftModel.from_pretrained(base, args.base_ckpt, is_trainable=True).to(device)
    model.train()

    label_ids_list = build_label_ids(tok)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found.")

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    loader = DataLoader(ds, batch_size=args.batch_pairs, shuffle=True)

    #optional warmup scheduler
    total_optim_steps = math.ceil((len(loader) * args.epochs) / max(args.grad_accum, 1))
    warmup_steps = int(args.warmup_ratio * total_optim_steps)
    if warmup_steps > 0:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_optim_steps)
    else:
        scheduler = None

    step_count = 0  #dataloader steps
    optim_step = 0  #optimizer steps

    for ep in range(args.epochs):
        total_loss = total_ce = total_cf = 0.0
        optim.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader, start=1):
            step_count += 1

            male_prompts = batch["male_prompt"]
            female_prompts = batch["female_prompt"]
            labels = batch["label"]
            B = len(labels)

            male_logps = label_logps_full(model, tok, male_prompts, label_ids_list, max_len=max_len)
            female_logps = label_logps_full(model, tok, female_prompts, label_ids_list, max_len=max_len)

            male_logdist = torch.log_softmax(male_logps, dim=-1)
            female_logdist = torch.log_softmax(female_logps, dim=-1)

            y = torch.tensor([LABEL2IDX[l] for l in labels], device=device, dtype=torch.long)

            ce_m = weighted_ce_from_logdist(male_logdist, y, class_w, label_smoothing=args.label_smoothing)
            ce_f = weighted_ce_from_logdist(female_logdist, y, class_w, label_smoothing=args.label_smoothing)
            ce = ce_m + ce_f

            cf = sym_kl(male_logdist, female_logdist).mean()

            #CF warmup schedule
            lam = args.lambda_cf
            if args.cf_warmup_steps > 0 and optim_step < args.cf_warmup_steps:
                lam = 0.0

            loss = ce + lam * cf
            loss = loss / max(args.grad_accum, 1)
            loss.backward()

            if step_count % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                if scheduler is not None:
                    scheduler.step()
                optim.zero_grad(set_to_none=True)
                optim_step += 1

            total_loss += float(loss.detach().cpu()) * max(args.grad_accum, 1)
            total_ce += float(ce.detach().cpu())
            total_cf += float(cf.detach().cpu())

            if step % 50 == 0:
                avg_loss = total_loss / step
                avg_ce = total_ce / step
                avg_cf = total_cf / step
                print(f"epoch {ep+1} step {step}: loss={avg_loss:.4f} (ce={avg_ce:.4f} cf={avg_cf:.4f}) lam={lam}")

        #flush leftover grads if not divisible by grad_accum
        if step_count % max(args.grad_accum, 1) != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            if scheduler is not None:
                scheduler.step()
            optim.zero_grad(set_to_none=True)
            optim_step += 1

        avg_loss = total_loss / max(len(loader), 1)
        avg_ce = total_ce / max(len(loader), 1)
        avg_cf = total_cf / max(len(loader), 1)
        print(f"epoch {ep+1} done: avg_loss={avg_loss:.4f} (ce={avg_ce:.4f} cf={avg_cf:.4f})")

    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)

    with open(os.path.join(args.out_dir, "debias_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": base_model,
            "base_ckpt": args.base_ckpt,
            "out_dir": args.out_dir,
            "epochs": args.epochs,
            "lr": args.lr,
            "lambda_cf": args.lambda_cf,
            "batch_pairs": args.batch_pairs,
            "grad_accum": args.grad_accum,
            "max_length": max_len,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "label_smoothing": args.label_smoothing,
            "cf_warmup_steps": args.cf_warmup_steps,
            "train_label_counter": dict(cnt),
            "class_weight_normed": [float(x) for x in class_w.detach().cpu()],
        }, f, indent=2, ensure_ascii=False)

    print("Saved:", args.out_dir)

if __name__ == "__main__":
    main()
