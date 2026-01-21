# LoRA Debiasing (Resume Screening) — Starter Kit

This repo generates a **counterfactual-pair** resume dataset (male vs female markers),
fine-tunes a small LLM with **LoRA**, and evaluates **utility + fairness**.

## Quickstart (recommended: Colab / Linux GPU)

```bash
pip install -r requirements.txt
python src/make_dataset.py --out_dir data/processed --n_pairs 400 --seed 42
python src/train_lora.py --config configs/config.yaml
python src/eval_fairness.py --config configs/config.yaml --ckpt runs/exp1
python src/plots.py --run_dir runs/exp1
```

## Key idea
For each resume, we create **two versions** that differ only in gender-coded markers (name + pronouns).
A fair model should keep the decision consistent across the pair.

## Outputs
- `data/processed/*.jsonl` : train/val/test
- `runs/exp1/` : LoRA adapter + metrics + plots

## Notes
- On Windows, `bitsandbytes` may not install; use Colab or WSL for 4-bit training.
- You can switch the base model in `configs/config.yaml`.


## Debias stage (counterfactual consistency)

```bash
python src/train_lora_cf.py --config configs/config.yaml --base_ckpt runs/exp1 --out_dir runs/exp1_cf --epochs 1 --lr 1e-4 --batch_pairs 2 --lambda_cf 0.3
python src/eval_fairness.py --config configs/config.yaml --ckpt runs/exp1_cf
python src/plots.py --run_dir runs/exp1_cf
```
