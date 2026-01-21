import os, json
import numpy as np
import matplotlib.pyplot as plt

EXPS = [
    ("base", r"runs/qwen05b_final_base/fairness_metrics.json"),
    ("base_cf", r"runs/qwen05b_final_base_cf/fairness_metrics.json"),
    ("bias30", r"runs/qwen05b_final_bias30/fairness_metrics.json"),
    ("bias30_cf_best", r"runs/qwen05b_final_bias30_cf_wce_ls01/fairness_metrics.json"),
]

OUT_DIR = r"figs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

metrics = {}
for name, path in EXPS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    metrics[name] = load_metrics(path)

names = [n for n, _ in EXPS]
x = np.arange(len(names))

fair_keys = [
    ("CF gap (expected score)", "counterfactual_gap_mean_expected_score"),
    ("Flip rate", "flip_rate"),
    ("Group accept gap", "group_accept_prob_gap"),
]

plt.figure()
width = 0.25
for i, (title, key) in enumerate(fair_keys):
    vals = [metrics[n].get(key, 0.0) for n in names]
    plt.bar(x + (i - (len(fair_keys)-1)/2)*width, vals, width=width, label=title)

plt.xticks(x, names, rotation=15)
plt.ylabel("Value (lower is better)")
plt.title("Fairness Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_fairness.png"), dpi=200)

# ====== 3) 图2：Utility ======
plt.figure()
acc = [metrics[n].get("utility_accuracy", 0.0) for n in names]
plt.bar(x, acc)
plt.xticks(x, names, rotation=15)
plt.ylabel("Utility Accuracy (higher is better)")
plt.title("Utility")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_utility.png"), dpi=200)

#预测分布 m/f 分开堆叠
labels = ["A", "B", "C"]
def get_pred_counts(exp_name, gender_key):
    d = metrics[exp_name].get(gender_key, {})
    return [d.get(l, 0) for l in labels]

plt.figure()
w = 0.35
m_counts = np.array([get_pred_counts(n, "pred_count_m") for n in names])
f_counts = np.array([get_pred_counts(n, "pred_count_f") for n in names])

# male stacked
bottom = np.zeros(len(names))
for j, lab in enumerate(labels):
    plt.bar(x - w/2, m_counts[:, j], width=w, bottom=bottom, label=f"Male {lab}" if j==0 else None)
    bottom += m_counts[:, j]

# female stacked
bottom = np.zeros(len(names))
for j, lab in enumerate(labels):
    plt.bar(x + w/2, f_counts[:, j], width=w, bottom=bottom, label=f"Female {lab}" if j==0 else None)
    bottom += f_counts[:, j]

plt.xticks(x, names, rotation=15)
plt.ylabel("Count")
plt.title("Prediction Distribution (Stacked)  |  Left=Male  Right=Female")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_pred_dist.png"), dpi=200)

print("Saved figures to:", OUT_DIR)
print(" - fig1_fairness.png")
print(" - fig2_utility.png")
print(" - fig3_pred_dist.png (optional)")
