import json, argparse
from collections import defaultdict
# 统一tag
def norm_gender(g):
    if g is None: return None
    if isinstance(g,str):
        s=g.strip().lower()
        if s in ["m","male","man","boy"]: return "male"
        if s in ["f","female","woman","girl"]: return "female"
        return s
    return g

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--keep_pairs", type=int, default=40)
    args=ap.parse_args()
    #load进内存
    rows=[]
    with open(args.infile,"r",encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    #pair_id group
    by=defaultdict(list)
    for r in rows:
        r["gender"]=norm_gender(r.get("gender"))
        by[r.get("pair_id")].append(r)

    kept=[]
    # 只取pair
    for pid,items in by.items():
        if len(items)!=2: 
            continue
        g=set([items[0].get("gender"), items[1].get("gender")])
        if g=={"male","female"}:
            kept.extend(items)
        if len(kept)//2 >= args.keep_pairs:
            break

    with open(args.outfile,"w",encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

    print("kept_pairs=", len(kept)//2, "rows=", len(kept))

if __name__=="__main__":
    main()
