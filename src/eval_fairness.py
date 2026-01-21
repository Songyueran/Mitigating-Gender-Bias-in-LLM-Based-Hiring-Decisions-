import argparse, os, json, math
from collections import Counter
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import format_prompt, LABELS

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# A=Reject, B=Maybe, C=Accept
LABEL2SCORE = {"A": 0.0, "B": 0.5, "C": 1.0}
POS_ACCEPT = "C"
POS_MAYBE = "B"

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

def is_peft_adapter_dir(path: str) -> bool:
    if not isinstance(path, str):
        return False
    if os.path.isdir(path):
        return (
            os.path.exists(os.path.join(path, "adapter_config.json")) or
            os.path.exists(os.path.join(path, "adapter_model.safetensors")) or
            os.path.exists(os.path.join(path, "adapter_model.bin"))
        )
    return False

def load_tokenizer(base_model: str, ckpt: str):
    try:
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok

def load_model_for_eval(base_model: str, ckpt: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # If ckpt is a PEFT adapter dir, load base + adapter, then (try) merge
    if is_peft_adapter_dir(ckpt):
        if PeftModel is None:
            raise RuntimeError("peft is not available but ckpt looks like a PEFT adapter dir.")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, ckpt)

        try:
            model = model.merge_and_unload()
        except Exception:
            pass

        model.eval()
        return model

    # Otherwise ckpt is full model dir or HF id
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model

def label_logprobs_full(model, tok, prompt: str, max_len: int, labels=LABELS):
    device = next(model.parameters()).device
    prompt_ids_full = tok(prompt, add_special_tokens=False).input_ids

    scores = []
    with torch.no_grad():
        for lab in labels:
            lab_ids = tok(" " + lab, add_special_tokens=False).input_ids

            keep = max(0, max_len - len(lab_ids))
            prompt_ids = prompt_ids_full[:keep]

            input_ids = torch.tensor([prompt_ids + lab_ids], device=device)
            out = model(input_ids=input_ids)
            logits = out.logits  # [1, T, V]

            logp = 0.0
            start = len(prompt_ids)
            for j, tok_id in enumerate(lab_ids):
                pos = start + j
                dist = torch.log_softmax(logits[0, pos - 1, :], dim=-1)
                logp += float(dist[tok_id].detach().cpu())
            scores.append(logp)

    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    return dict(zip(labels, probs))

def expected_score(probs: dict):
    return sum(probs[l] * LABEL2SCORE[l] for l in probs.keys())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True, help="HF model id OR full model dir OR PEFT adapter dir (runs/xxx)")
    ap.add_argument("--positive", choices=["accept", "accept_or_maybe"], default="accept_or_maybe",
                    help="definition of positive decision for DP metrics")
    ap.add_argument("--dp_mode", choices=["prob", "argmax"], default="prob",
                    help="DP uses probabilistic positive (recommended) or argmax label")
    ap.add_argument("--debug_preds", action="store_true", help="print prediction label counters")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    test_file = cfg["data"]["test_file"]
    base_model = cfg["model"]["base_model"]
    max_len = int(cfg["model"]["max_length"])

    ds = load_dataset("json", data_files={"test": test_file})["test"]

    tok = load_tokenizer(base_model, args.ckpt)
    model = load_model_for_eval(base_model, args.ckpt)

    by_pair = {}
    for ex in ds:
        pid = ex.get("pair_id")
        if pid is None:
            continue
        by_pair.setdefault(pid, []).append(ex)

    gaps_accept = []
    gaps_score = []
    flip = 0
    correct = 0
    total = 0

    accept_prob_m, accept_prob_f = [], []
    score_m, score_f = [], []
    pos_m = 0.0
    pos_f = 0.0
    n_m = n_f = 0
    skipped = 0

    pred_cnt_m = Counter()
    pred_cnt_f = Counter()

    for pid, items in by_pair.items():
        if len(items) != 2:
            skipped += 1
            continue

        a, b = items[0], items[1]
        ga = norm_gender(a.get("gender"))
        gb = norm_gender(b.get("gender"))

        if ga == "male" and gb == "female":
            male, female = a, b
        elif gb == "male" and ga == "female":
            male, female = b, a
        else:
            skipped += 1
            continue

        pm = label_logprobs_full(model, tok, format_prompt(male["job_desc"], male["resume"]), max_len=max_len)
        pf = label_logprobs_full(model, tok, format_prompt(female["job_desc"], female["resume"]), max_len=max_len)

        gaps_accept.append(abs(pm[POS_ACCEPT] - pf[POS_ACCEPT]))
        gaps_score.append(abs(expected_score(pm) - expected_score(pf)))

        pred_m = max(pm, key=pm.get)  # A/B/C
        pred_f = max(pf, key=pf.get)
        pred_cnt_m[pred_m] += 1
        pred_cnt_f[pred_f] += 1
        if pred_m != pred_f:
            flip += 1

        # utility accuracy assumes dataset label is also A/B/C
        correct += 1 if pred_m == male["label"] else 0
        correct += 1 if pred_f == female["label"] else 0
        total += 2

        accept_prob_m.append(pm[POS_ACCEPT])
        accept_prob_f.append(pf[POS_ACCEPT])
        score_m.append(expected_score(pm))
        score_f.append(expected_score(pf))

        # DP / positive rate
        if args.dp_mode == "argmax":
            if args.positive == "accept":
                pos_m += 1.0 if pred_m == POS_ACCEPT else 0.0
                pos_f += 1.0 if pred_f == POS_ACCEPT else 0.0
            else:
                pos_m += 1.0 if pred_m in [POS_ACCEPT, POS_MAYBE] else 0.0
                pos_f += 1.0 if pred_f in [POS_ACCEPT, POS_MAYBE] else 0.0
        else:
            if args.positive == "accept":
                pos_m += float(pm[POS_ACCEPT])
                pos_f += float(pf[POS_ACCEPT])
            else:
                pos_m += float(pm[POS_ACCEPT] + pm[POS_MAYBE])
                pos_f += float(pf[POS_ACCEPT] + pf[POS_MAYBE])

        n_m += 1
        n_f += 1

    n_pairs = len(gaps_accept)
    mean_accept_m = (sum(accept_prob_m) / n_pairs) if n_pairs else 0.0
    mean_accept_f = (sum(accept_prob_f) / n_pairs) if n_pairs else 0.0
    mean_score_m = (sum(score_m) / n_pairs) if n_pairs else 0.0
    mean_score_f = (sum(score_f) / n_pairs) if n_pairs else 0.0

    metrics = {
        "n_pairs_eval": n_pairs,
        "skipped_pairs": skipped,
        "utility_accuracy": (correct / total) if total else 0.0,
        "counterfactual_gap_mean_accept_prob": (sum(gaps_accept) / n_pairs) if n_pairs else 0.0,
        "counterfactual_gap_mean_expected_score": (sum(gaps_score) / n_pairs) if n_pairs else 0.0,
        "flip_rate": (flip / n_pairs) if n_pairs else 0.0,
        "mean_accept_prob_m": mean_accept_m,
        "mean_accept_prob_f": mean_accept_f,
        "group_accept_prob_gap": abs(mean_accept_m - mean_accept_f),
        "mean_score_m": mean_score_m,
        "mean_score_f": mean_score_f,
        "group_score_gap": abs(mean_score_m - mean_score_f),
        "positive_def": args.positive,
        "dp_mode": args.dp_mode,
        "demographic_parity_diff": (pos_m / max(n_m, 1)) - (pos_f / max(n_f, 1)),
        "positive_rate_m": pos_m / max(n_m, 1),
        "positive_rate_f": pos_f / max(n_f, 1),
    }

    if args.debug_preds:
        metrics["pred_count_m"] = dict(pred_cnt_m)
        metrics["pred_count_f"] = dict(pred_cnt_f)
        print("pred_cnt_m:", pred_cnt_m)
        print("pred_cnt_f:", pred_cnt_f)

    out_path = os.path.join(args.ckpt, "fairness_metrics.json") if os.path.isdir(args.ckpt) else "fairness_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
