# Mitigating Gender Bias in LLM-Based Hiring Decisions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LoRA](https://img.shields.io/badge/Method-LoRA-green.svg)](https://arxiv.org/abs/2106.09685)
[![Fairness](https://img.shields.io/badge/Focus-Fairness%20%26%20Utility-orange.svg)]()

This repository focuses on addressing gender bias in automated resume screening using Large Language Models (LLMs). By leveraging **LoRA (Low-Rank Adaptation)** and **Counterfactual Consistency Training**, we aim to reduce disparate impact while maintaining high model performance (utility).

---

## ✨ Core Concept: Counterfactual Pairs

To measure and mitigate bias, we utilize a **Counterfactual-Pair** dataset strategy:
- **Pair Construction:** For every `pair_id`, we generate two versions of the same resume—one with **male** gender markers (name/pronouns) and one with **female** markers.
- **Controlled Variables:** All professional content (skills, experience, education) remains identical across the pair.
- **Fairness Goal:** A fair model should yield **consistent decisions** and similar probability distributions regardless of the gender cues.

---

## 🛠️ Installation & Setup

### Requirements
- Python 3.10+
- NVIDIA GPU + CUDA for training acceleration

```bash
# Clone the repository
git clone [https://github.com/Songyueran/Mitigating-Gender-Bias-in-LLM-Based-Hiring-Decisions-.git](https://github.com/Songyueran/Mitigating-Gender-Bias-in-LLM-Based-Hiring-Decisions-.git)
cd Mitigating-Gender-Bias-in-LLM-Based-Hiring-Decisions-

# Install dependencies
pip install -r requirements.txt
📈 Experimental WorkflowThe training is divided into two distinct stages:Stage 1: Supervised Fine-Tuning (SFT)Standard LoRA fine-tuning on base and biased environments to establish baselines.PowerShell# Train on Base (Clean) dataset
python -m src.train_lora --config configs/config_qwen05b_final.yaml

# Train on Bias30 (30% biased injection) dataset
python -m src.train_lora --config configs/config_qwen05b_final_bias30.yaml
Stage 2: Debiasing via Counterfactual Consistency (CF)We introduce a Counterfactual Loss that penalizes the model when its predictions diverge for the male/female versions of the same resume.$$Loss = CE + \lambda_{cf} \cdot CF$$PowerShell# Perform CF training on the biased model
python -m src.train_lora_cf --config configs/config_qwen05b_final_bias30.yaml `
    --base_ckpt runs/qwen05b_final_bias30 `
    --out_dir runs/qwen05b_final_bias30_cf_wce_ls01 `
    --lambda_cf 0.03
📊 Evaluation MetricsWe evaluate models across two primary dimensions:Utility: Classification Accuracy and F1-score for hiring labels (A=Reject, B=Maybe, C=Accept).Fairness: - Demographic Parity: Difference in acceptance rates between genders.Counterfactual Consistency: The percentage of pairs where the model made the exact same decision for both versions.PowerShell# Run fairness evaluation
python -m src.eval_fairness --config configs/config_qwen05b_final_bias30.yaml --ckpt runs/qwen05b_final_bias30_cf_wce_ls01 --debug_preds
📂 Project StructurePlaintext├── configs/          # YAML configuration files for different runs
├── data/             # Processed counterfactual resume datasets
├── src/              # Core source code
│   ├── train_lora.py    # Stage 1 training script
│   ├── train_lora_cf.py # Stage 2 debiasing script
│   └── eval_fairness.py # Fairness & utility evaluation tools
├── tools/            # Plotting and data analysis utilities
└── runs/             # Model checkpoints (excluded in .gitignore)
