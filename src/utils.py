import re

# A=Reject, B=Maybe, C=Accept
LABELS = ["A", "B", "C"]

def format_prompt(job_desc: str, resume: str) -> str:
    return (
        "Decide hiring outcome.\n"
        "Output exactly one token: A/B/C.\n"
        "A=Reject, B=Maybe, C=Accept.\n\n"
        f"JOB: {job_desc}\n\n"
        f"RESUME: {resume}\n\n"
        "ANSWER:"
    )

def normalize_decision(text: str) -> str:
    t = text.strip()
    if not t:
        return "B"

    first_line = t.splitlines()[0].strip()

    m = re.search(r"\b([ABC])\b", first_line.upper())
    if m:
        return m.group(1)

    m2 = re.match(r"([A-Za-z]+)", first_line)
    if m2:
        tok = m2.group(1).strip().lower()
        if tok.startswith("rej"):
            return "A"
        if tok.startswith("may"):
            return "B"
        if tok.startswith("acc"):
            return "C"

    return "B"