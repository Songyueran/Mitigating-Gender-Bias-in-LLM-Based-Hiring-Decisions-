import os, json
import matplotlib.pyplot as plt

RUN_DIRS = [
    "runs/qwen05b_final_base",
    "runs/qwen05b_final_bias30",
    "runs/qwen05b_final_bias30_cf_wce_ls01",
]

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

def load_metrics(run_dir: str):
    path = os.path.join(run_dir, "fairness_metrics.json")
    if not os.path.exists(path):
        print(f"[skip] missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["_run_dir"] = run_dir
    m["_name"] = os.path.basename(run_dir)
    return m

def save_bar(models, values, title, ylabel, out_path):
    plt.figure()
    plt.bar(models, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_bar_abs(models, values, title, ylabel, out_path):
    abs_vals = [abs(v) for v in values]
    plt.figure()
    plt.bar(models, abs_vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    metrics_list = []
    for d in RUN_DIRS:
        m = load_metrics(d)
        if m is not None:
            metrics_list.append(m)

    if not metrics_list:
        raise SystemExit("No fairness_metrics.json found in given RUN_DIRS.")

    #保持 RUN_DIRS 顺序
    name2idx = {os.path.basename(d): i for i, d in enumerate(RUN_DIRS)}
    metrics_list.sort(key=lambda x: name2idx.get(x["_name"], 999))

    names = [m["_name"] for m in metrics_list]

    key_specs = [
        ("utility_accuracy", "Utility Accuracy (higher better)", "accuracy"),
        ("counterfactual_gap_mean_accept_prob", "CF Gap: mean |P(C)-P(C)| (lower better)", "gap"),
        ("counterfactual_gap_mean_expected_score", "CF Gap: mean |E[score]_m - E[score]_f| (lower better)", "gap"),
        ("flip_rate", "Flip Rate (lower better)", "rate"),
        ("group_accept_prob_gap", "Group Accept Prob Gap |mean_m-mean_f| (lower better)", "gap"),
        ("group_score_gap", "Group Score Gap |mean_m-mean_f| (lower better)", "gap"),
        ("demographic_parity_diff", "Demographic Parity Diff (close to 0 better)", "diff"),
    ]

    print("\n=== Summary (from fairness_metrics.json) ===")
    header = ["model"] + [k for k, _, _ in key_specs]
    print("\t".join(header))
    for m in metrics_list:
        row = [m["_name"]]
        for k, _, _ in key_specs:
            v = m.get(k, None)
            if isinstance(v, float):
                row.append(f"{v:.6f}")
            else:
                row.append(str(v))
        print("\t".join(row))

    #逐指标画图
    for k, title, kind in key_specs:
        vals = [float(m.get(k, 0.0)) for m in metrics_list]
        out_path = os.path.join(OUT_DIR, f"{k}.png")
        save_bar(names, vals, title, k, out_path)
        print(f"[saved] {out_path}")

        # DP diff 绝对值图
        if k == "demographic_parity_diff":
            out_path2 = os.path.join(OUT_DIR, f"{k}_abs.png")
            save_bar_abs(names, vals, title + " (ABS)", f"|{k}|", out_path2)
            print(f"[saved] {out_path2}")

    # “综合视图”：fairness 指标放一起
    fairness_keys = [
        "counterfactual_gap_mean_accept_prob",
        "counterfactual_gap_mean_expected_score",
        "flip_rate",
        "group_accept_prob_gap",
        "group_score_gap",
        "demographic_parity_diff",
    ]

    plt.figure()
    for m in metrics_list:
        ys = [float(m.get(k, 0.0)) for k in fairness_keys]
        plt.plot(fairness_keys, ys, marker="o", label=m["_name"])
    plt.title("Fairness Metrics Overview (line plot)")
    plt.ylabel("metric value")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fairness_overview.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")

    print("\nDone. Open the PNGs in ./plots/")

if __name__ == "__main__":
    main()
