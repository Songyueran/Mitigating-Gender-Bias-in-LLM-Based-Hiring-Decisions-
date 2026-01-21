import argparse, json, os, random
from collections import Counter

LABELS = ["Reject", "Maybe", "Accept"]

def shift_down(label: str) -> str:
    if label == "Accept": return "Maybe"
    if label == "Maybe": return "Reject"
    return "Reject"

def shift_up(label: str) -> str:
    if label == "Reject": return "Maybe"
    if label == "Maybe": return "Accept"
    return "Accept"

def norm_gender(g):
    if g is None: return None
    gg = str(g).strip().lower()
    if gg in ["male","m","man","boy"]: return "male"
    if gg in ["female","f","woman","girl"]: return "female"
    return gg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--bias_rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["female_down", "male_up", "both"], default="both")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = []
    with open(args.in_train, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    before = Counter(r["label"] for r in rows)

    changed = 0
    for r in rows:
        g = norm_gender(r.get("gender"))
        if rng.random() > args.bias_rate:
            continue

        if args.mode == "female_down" and g == "female":
            new_label = shift_down(r["label"])
        elif args.mode == "male_up" and g == "male":
            new_label = shift_up(r["label"])
        elif args.mode == "both":
            if g == "female":
                new_label = shift_down(r["label"])
            elif g == "male":
                new_label = shift_up(r["label"])
            else:
                continue
        else:
            continue

        if new_label != r["label"]:
            r["label"] = new_label
            changed += 1

    after = Counter(r["label"] for r in rows)

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("Bias mode:", args.mode)
    print("Bias rate:", args.bias_rate, "seed:", args.seed)
    print("Changed records:", changed, "/", len(rows))
    print("Label dist before:", dict(before))
    print("Label dist after :", dict(after))
    print("Wrote:", args.out_train)

if __name__ == "__main__":
    main()
