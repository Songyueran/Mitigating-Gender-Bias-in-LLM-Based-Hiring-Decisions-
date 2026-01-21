```markdown
# LoRA Debiasing for Resume Screening (Counterfactual Pairs)

This repo builds a **counterfactual-pair** resume screening dataset (same content, gender markers swapped),
fine-tunes **Qwen2.5-0.5B-Instruct** using **LoRA**, and evaluates **utility + fairness**.
It also supports a second-stage **Counterfactual Consistency (CF)** training step to mitigate **gender bias**.

---

## ✨ Key Idea

For each `pair_id`, we create **two versions** of the same resume:

- one with **male** gender cues (name + pronouns)
- one with **female** gender cues (name + pronouns)

Everything else (job description + resume content) stays identical.

A fair model should output **consistent decisions** and/or **similar decision probabilities** across the pair.

---

## 📌 Task & Labels

**Task:** Resume screening classification

**Labels (single-token decisions):**
- `A` = Reject  
- `B` = Maybe  
- `C` = Accept  

> The prompt template and label mapping live in `src/utils.py` (`format_prompt`, `LABELS`, etc.).

---

## 📁 Repo Layout (current project)

```

LORA_DEBIAS_RESUME_STARTER/
├─ configs/
│  ├─ config_qwen05b_final.yaml
│  └─ config_qwen05b_final_bias30.yaml
├─ data/
│  ├─ processed_final/
│  │  ├─ train.jsonl
│  │  ├─ val.jsonl
│  │  └─ test.jsonl
│  └─ processed_final_bias30/
│     ├─ train.jsonl
│     ├─ val.jsonl
│     └─ test.jsonl
├─ runs/
│  ├─ qwen05b_final_base/
│  ├─ qwen05b_final_bias30/
│  ├─ qwen05b_final_base_cf/
│  ├─ qwen05b_final_bias30_cf/
│  └─ qwen05b_final_bias30_cf_wce_ls01/
├─ figs/
│  ├─ fig1_fairness.png
│  ├─ fig2_utility.png
│  └─ fig3_pred_dist.png
├─ paper/
│  └─ outline.md
├─ tools/
│  ├─ plot_01.py
│  └─ plot_metrics.py
├─ src/
│  ├─ make_dataset.py
│  ├─ make_biased_train.py
│  ├─ inject_gender_cue.py
│  ├─ clean_pairs.py
│  ├─ train_lora.py
│  ├─ train_lora_cf.py
│  ├─ eval_fairness.py
│  └─ utils.py
├─ requirements.txt
└─ README.md

````

---

## ✅ Installation

Recommended:
- Python **3.10+**
- GPU + CUDA for practical training speed (Windows/Linux both OK)

```bash
pip install -r requirements.txt
````

### Notes for Windows

* HuggingFace downloads may time out occasionally — re-run the command if needed.
* If you plan to use 4-bit training (`bitsandbytes`), **WSL** or **Colab/Linux** is usually easier.
* This repo works fine with normal fp16 LoRA on Windows + CUDA.

---

## 0) Data Sanity Checks

### 0.1 Check file existence + line counts (PowerShell)

```powershell
python -c "import pathlib; 
base=pathlib.Path('data/processed_final');
for s in ['train','val','test']:
    f=base/f'{s}.jsonl'
    print(s,'exists=',f.exists(),'lines=',sum(1 for _ in open(f,'r',encoding='utf-8')))"
```

### 0.2 Quick label distribution check (optional)

```powershell
python -c "import json,collections; 
p=r'data/processed_final/train.jsonl';
c=collections.Counter(json.loads(l)['label'] for l in open(p,'r',encoding='utf-8') if l.strip());
print(c,'n=',sum(c.values()))"
```

---

## 1) Stage-1 Training: LoRA SFT

This is the standard supervised fine-tuning stage.

### 1.1 Train **base** (clean dataset)

Output directory: `runs/qwen05b_final_base/`

```powershell
python -m src.train_lora --config configs/config_qwen05b_final.yaml
```

### 1.2 Train **bias30** (biased dataset)

Output directory: `runs/qwen05b_final_bias30/`

```powershell
python -m src.train_lora --config configs/config_qwen05b_final_bias30.yaml
```

---

## 2) Evaluation: Utility + Fairness

Evaluation writes:

* `runs/<ckpt>/fairness_metrics.json`

Use `--debug_preds` to print per-group label counts.

### 2.1 Eval **base**

```powershell
python -m src.eval_fairness --config configs/config_qwen05b_final.yaml --ckpt runs/qwen05b_final_base --debug_preds
```

### 2.2 Eval **bias30**

```powershell
python -m src.eval_fairness --config configs/config_qwen05b_final_bias30.yaml --ckpt runs/qwen05b_final_bias30 --debug_preds
```

---

## 3) Stage-2 Debias: Counterfactual Consistency (CF)

This stage encourages the model to output **similar decision distributions**
for the male/female members of each `pair_id`.

### What it optimizes (conceptually)

* **CE loss**: still predicts the correct label (A/B/C)
* **CF loss**: penalizes disagreement between male/female distributions
  (e.g., symmetric KL divergence)

Total loss:

```
loss = CE + lambda_cf * CF
```

---

## 4) CF Training Commands (copy-paste)

### 4.1 Train **base_cf** using the updated `train_lora_cf.py`

> You asked to keep the same output directory name (do NOT change the runs path).

Output directory: `runs/qwen05b_final_base_cf/`

```powershell
python -m src.train_lora_cf --config configs/config_qwen05b_final.yaml --base_ckpt runs/qwen05b_final_base --out_dir runs/qwen05b_final_base_cf --epochs 1 --lr 5e-5 --lambda_cf 0.03 --batch_pairs 2 --grad_accum 1
```

### 4.2 Train **bias30_cf** (example)

Output directory: `runs/qwen05b_final_bias30_cf/`

```powershell
python -m src.train_lora_cf --config configs/config_qwen05b_final_bias30.yaml --base_ckpt runs/qwen05b_final_bias30 --out_dir runs/qwen05b_final_bias30_cf --epochs 1 --lr 5e-5 --lambda_cf 0.03 --batch_pairs 2 --grad_accum 1
```

### 4.3 Train **final debiased model** (your current “best” run)

Output directory: `runs/qwen05b_final_bias30_cf_wce_ls01/`

```powershell
python -m src.train_lora_cf --config configs/config_qwen05b_final_bias30.yaml --base_ckpt runs/qwen05b_final_bias30 --out_dir runs/qwen05b_final_bias30_cf_wce_ls01 --epochs 1 --lr 5e-5 --lambda_cf 0.03 --batch_pairs 2 --grad_accum 1
```

---

## 5) CF Evaluation Commands (copy-paste)

### 5.1 Eval **base_cf**

```powershell
python -m src.eval_fairness --config configs/config_qwen05b_final.yaml --ckpt runs/qwen05b_final_base_cf --debug_preds
```

### 5.2 Eval **bias30_cf (final)**

```powershell
python -m src.eval_fairness --config configs/config_qwen05b_final_bias30.yaml --ckpt runs/qwen05b_final_bias30_cf_wce_ls01 --debug_preds
```

---

## 6) Plotting

You already generate figures into `figs/` and/or `plots/`.
Use the scripts under `tools/` (exact flags depend on your local script arguments).

Common workflow:

* main comparison (3 models):

  * `qwen05b_final_base`
  * `qwen05b_final_bias30`
  * `qwen05b_final_bias30_cf_wce_ls01`

* ablation (optional figure):

  * `qwen05b_final_base` vs `qwen05b_final_base_cf`

Example (adjust to your `tools/plot_metrics.py` arguments if needed):

```powershell
python tools/plot_metrics.py --runs runs/qwen05b_final_base runs/qwen05b_final_bias30 runs/qwen05b_final_bias30_cf_wce_ls01 --out_dir figs
```

Ablation:

```powershell
python tools/plot_metrics.py --runs runs/qwen05b_final_base runs/qwen05b_final_base_cf --out_dir figs/base_cf_ablation
```

---

## 🧯 Troubleshooting

### 1) `Special tokens have been added in the vocabulary...`

This is a warning (not an error). Training adapters will still work normally.

### 2) HuggingFace download timeouts

Network issue. Simply re-run the command.
If it happens often, consider pre-downloading the model or increasing HF timeout via env vars.

### 3) `forward() got an unexpected keyword argument 'label_id'`

This happens when extra fields (like `label_id`) are passed into `model(**inputs)`.
Fix by ensuring:

* the collator only returns keys expected by the model (`input_ids`, `attention_mask`, `labels`), or
* the trainer filters unused columns / does not forward auxiliary fields.

---

## 📌 Outputs

After each run, you should have:

* `runs/<exp>/adapter_model.safetensors` (LoRA weights)
* `runs/<exp>/adapter_config.json`
* `runs/<exp>/fairness_metrics.json`
* (optional) `runs/<exp>/debias_summary.json` for CF stage

---
