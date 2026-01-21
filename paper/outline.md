# Paper Outline (6–8 pages)

## Abstract
- 150–200 words: task, method (LoRA + counterfactual consistency), results (fairness improves with minor utility drop)

## 1. Introduction
- LLMs in hiring & risks of bias
- Research question: can parameter-efficient fine-tuning reduce gender sensitivity?

## 2. Related Work
- Counterfactual fairness (definition + why relevant)
- Bias evaluation datasets (CrowS-Pairs / StereoSet) (optional)
- Parameter-efficient fine-tuning (LoRA)

## 3. Dataset
- How resumes + job descriptions are generated
- Counterfactual pairing design, split by pair_id

## 4. Method
- Prompt format, label space
- LoRA setup
- Debias objective: utility loss + lambda * counterfactual consistency

## 5. Experiments
- Baselines, hyperparameters, compute
- Metrics: Accuracy/F1; Counterfactual Gap; Demographic Parity

## 6. Results
- Main table (utility + fairness)
- Tradeoff plot (lambda sweep)
- Error analysis examples

## 7. Limitations & Ethics
- Synthetic data limits, only binary gender markers, etc.

## 8. Conclusion
