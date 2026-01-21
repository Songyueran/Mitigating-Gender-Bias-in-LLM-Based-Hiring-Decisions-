import os, json, random, argparse
from pathlib import Path

MALE = ("John", "he/him")
FEMALE = ("Mary", "she/her")

def inject_line(obj, gender: str):
    name, pro = MALE if gender == "M" else FEMALE
    prefix = f"Name: {name}. Pronouns: {pro}.\n"
    obj["gender"] = gender
    obj["resume"] = prefix + (obj.get("resume") or "")
    return obj

def run(in_dir: str, out_dir: str, seed: int = 42):
    random.seed(seed)
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        inp = in_dir / split
        outp = out_dir / split
        if not inp.exists():
            raise FileNotFoundError(f"Missing {inp}")

        with open(inp, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
            for line in f_in:
                obj = json.loads(line)
                gender = random.choice(["M", "F"])
                obj = inject_line(obj, gender)
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved gender-injected dataset to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.in_dir, args.out_dir, seed=args.seed)
