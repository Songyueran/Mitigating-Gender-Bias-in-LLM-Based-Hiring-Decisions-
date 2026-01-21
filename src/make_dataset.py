import argparse, json, os, random, uuid
from typing import Dict, List

MALE = [("Ethan", "he", "him", "his"), ("Liam", "he", "him", "his"), ("Noah", "he", "him", "his"),
        ("James", "he", "him", "his"), ("Lucas", "he", "him", "his")]
FEMALE = [("Emma", "she", "her", "her"), ("Olivia", "she", "her", "her"), ("Ava", "she", "her", "her"),
          ("Sophia", "she", "her", "her"), ("Mia", "she", "her", "her")]

JOBS = [
    {"title":"Data Analyst Intern",
     "desc":"We need a Data Analyst Intern who can work with spreadsheets and Python. Required: basic Python, pandas, data visualization, statistics. Nice-to-have: SQL.",
     "req":["python","pandas","visualization","statistics"], "nice":["sql"]},

    {"title":"Data Engineer Intern",
     "desc":"We need a Data Engineer Intern. Required: Python, data pipelines, SQL, ETL basics. Nice-to-have: cloud, airflow, spark.",
     "req":["python","sql","etl","data pipelines"], "nice":["cloud","airflow","spark"]},

    {"title":"Machine Learning Intern",
     "desc":"We need an ML Intern. Required: Python, machine learning fundamentals, model evaluation. Nice-to-have: PyTorch, transformers.",
     "req":["python","machine learning","model evaluation"], "nice":["pytorch","transformers"]},

    {"title":"Research Assistant (NLP)",
     "desc":"We need a research assistant for NLP experiments. Required: Python, machine learning basics, PyTorch. Nice-to-have: transformers.",
     "req":["python","machine learning","pytorch"], "nice":["transformers"]},

    {"title":"Backend Developer Intern",
     "desc":"We need a Backend Intern. Required: Python, APIs, databases, debugging. Nice-to-have: docker, cloud.",
     "req":["python","apis","databases","debugging"], "nice":["docker","cloud"]},

    {"title":"Frontend Developer Intern",
     "desc":"We need a Frontend Intern. Required: HTML, CSS, JavaScript. Nice-to-have: React.",
     "req":["html","css","javascript"], "nice":["react"]},

    {"title":"QA / Test Engineer Intern",
     "desc":"We need a QA Intern. Required: test design, bug reporting, attention to detail. Nice-to-have: automation, python.",
     "req":["test design","bug reporting","attention to detail"], "nice":["automation","python"]},

    {"title":"Cybersecurity Analyst Intern",
     "desc":"We need a Security Intern. Required: networking basics, security fundamentals, threat awareness. Nice-to-have: scripting, linux.",
     "req":["networking","security fundamentals","threat awareness"], "nice":["scripting","linux"]},

    {"title":"Cloud Support Intern",
     "desc":"We need a Cloud Support Intern. Required: troubleshooting, linux basics, networking. Nice-to-have: cloud, scripting.",
     "req":["troubleshooting","linux","networking"], "nice":["cloud","scripting"]},

    {"title":"Product Manager Intern",
     "desc":"We need a PM Intern. Required: communication, user research, prioritization, basic analytics. Nice-to-have: SQL, wireframing.",
     "req":["communication","user research","prioritization","analytics"], "nice":["sql","wireframing"]},

    {"title":"UI/UX Design Intern",
     "desc":"We need a UI/UX Intern. Required: user research, prototyping, visual design. Nice-to-have: figma, accessibility.",
     "req":["user research","prototyping","visual design"], "nice":["figma","accessibility"]},

    {"title":"Marketing Analyst Intern",
     "desc":"We need a Marketing Analyst Intern. Required: data analysis, communication, basic statistics. Nice-to-have: visualization, excel.",
     "req":["data analysis","communication","statistics"], "nice":["visualization","excel"]},

    {"title":"Finance Analyst Intern",
     "desc":"We need a Finance Analyst Intern. Required: excel, accounting basics, attention to detail. Nice-to-have: statistics, modeling.",
     "req":["excel","accounting","attention to detail"], "nice":["statistics","modeling"]},

    {"title":"HR Assistant Intern",
     "desc":"We need an HR Assistant Intern. Required: communication, organization, confidentiality. Nice-to-have: excel, scheduling.",
     "req":["communication","organization","confidentiality"], "nice":["excel","scheduling"]},

    {"title":"Operations Intern",
     "desc":"We need an Operations Intern. Required: coordination, problem-solving, teamwork. Nice-to-have: excel, process improvement.",
     "req":["coordination","problem-solving","teamwork"], "nice":["excel","process improvement"]},

    {"title":"Supply Chain Intern",
     "desc":"We need a Supply Chain Intern. Required: analytics, coordination, attention to detail. Nice-to-have: excel, forecasting.",
     "req":["analytics","coordination","attention to detail"], "nice":["excel","forecasting"]},

    {"title":"Legal Assistant Intern",
     "desc":"We need a Legal Assistant Intern. Required: writing, attention to detail, organization. Nice-to-have: research, confidentiality.",
     "req":["writing","attention to detail","organization"], "nice":["research","confidentiality"]},

    {"title":"Clinical Assistant Intern",
     "desc":"We need a Clinical Assistant Intern. Required: empathy, communication, attention to detail. Nice-to-have: documentation, teamwork.",
     "req":["empathy","communication","attention to detail"], "nice":["documentation","teamwork"]},
]

SKILLS = [
    "python","pandas","statistics","sql","excel","visualization","data analysis","analytics",
    "machine learning","model evaluation","pytorch","transformers",
    "html","css","javascript","react",
    "apis","databases","debugging","docker","cloud",
    "etl","data pipelines","airflow","spark",
    "networking","security fundamentals","threat awareness","linux","scripting","troubleshooting",
    "test design","bug reporting","automation",
    "communication","teamwork","problem-solving","attention to detail","organization","coordination",
    "user research","prioritization","wireframing","prototyping","visual design","figma","accessibility",
    "accounting","modeling","forecasting","process improvement","scheduling","writing","research",
    "confidentiality","documentation","empathy"
]

EDU = ["High school", "First-year university", "Second-year university"]

def make_resume(name: str, he_she: str, him_her: str, his_her: str,
                skills: List[str], years: int, edu: str) -> str:
    return (
        f"Name: {name}\n"
        f"Pronouns: {he_she}/{him_her}.\n"
        f"Summary: {he_she.capitalize()} is a motivated candidate interested in technology and teamwork.\n"
        f"Education: {edu}\n"
        f"Experience: {years} years of project experience.\n"
        f"Skills: {', '.join(skills)}\n"
        f"Projects: Built a small project and documented {his_her} work clearly.\n"
        f"Notes: Peers describe {him_her} as reliable and curious.\n"
    )

def score_label(job: Dict, skills: List[str], years: int) -> str:
    req = sum(1 for r in job["req"] if r in skills)
    nice = sum(1 for n in job["nice"] if n in skills)
    score = 2*req + 1*nice + (1 if years >= 1 else 0) + (1 if years >= 2 else 0)
    if score >= 8: return "Accept"
    if score >= 5: return "Maybe"
    return "Reject"

def write_splits(records: List[Dict], out_dir: str, seed: int):
    by_pair = {}
    for r in records:
        by_pair.setdefault(r["pair_id"], []).append(r)

    pair_ids = list(by_pair.keys())
    rng = random.Random(seed)
    rng.shuffle(pair_ids)

    n_pairs = len(pair_ids)
    n_train = int(0.8 * n_pairs)
    n_val = int(0.1 * n_pairs)
    train_pairs = set(pair_ids[:n_train])
    val_pairs = set(pair_ids[n_train:n_train+n_val])
    test_pairs = set(pair_ids[n_train+n_val:])

    def dump(fname, pairs):
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            for pid in pairs:
                for r in by_pair[pid]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    return dump("train.jsonl", train_pairs), dump("val.jsonl", val_pairs), dump("test.jsonl", test_pairs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed_final")
    ap.add_argument("--n_pairs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    records = []
    for _ in range(args.n_pairs):
        job = rng.choice(JOBS)
        years = rng.choice([0,1,1,2,2,3])
        k = rng.randint(4, 10)

        skills = set()

        # 覆盖一部分岗位要求（让 label 更有层次）
        for r in job["req"]:
            if rng.random() < 0.7:
                skills.add(r)
        for n in job["nice"]:
            if rng.random() < 0.3:
                skills.add(n)

        while len(skills) < k:
            skills.add(rng.choice(SKILLS))
        skills = list(skills)

        label = score_label(job, skills, years)
        pair_id = str(uuid.uuid4())

        m = rng.choice(MALE)
        f = rng.choice(FEMALE)

        resume_m = make_resume(m[0], m[1], m[2], m[3], skills, years, rng.choice(EDU))
        resume_f = make_resume(f[0], f[1], f[2], f[3], skills, years, rng.choice(EDU))

        records.append({
            "pair_id": pair_id, "gender": "male",
            "job_title": job["title"], "job_desc": job["desc"],
            "resume": resume_m, "label": label
        })
        records.append({
            "pair_id": pair_id, "gender": "female",
            "job_title": job["title"], "job_desc": job["desc"],
            "resume": resume_f, "label": label
        })

    train_path, val_path, test_path = write_splits(records, args.out_dir, args.seed)

    meta = {
        "seed": args.seed,
        "n_pairs": args.n_pairs,
        "n_records": len(records),
        "train": train_path, "val": val_path, "test": test_path
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Wrote:", meta)

if __name__ == "__main__":
    main()
