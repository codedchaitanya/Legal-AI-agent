"""
Download adalat-ai/indian-legal-exam-benchmark (clat_ug, clat_pg, djs_dhjs),
classify each question into one of the 12 training domains, and write per-domain
JSONL files in the same {"text": "...", "domain": "...", "source": "clat_mcq"} format
that colab_train.py already expects.

Run:
    python training/clat-data/download_and_split.py
    python training/clat-data/download_and_split.py --hf_token hf_xxx
"""

import argparse, json, os, re
from collections import Counter
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

# ── Domain system prompts (same voice as existing training data) ──────────────
SYSTEM_PROMPTS = {
    "constitutional":         "You are an expert Indian constitutional law assistant with deep knowledge of the Constitution of India, fundamental rights, directive principles, constitutional amendments, and landmark Supreme Court judgments on constitutional matters.",
    "criminal_violent":       "You are an expert Indian criminal law assistant specialising in violent offences under the IPC and Bharatiya Nyaya Sanhita. You have deep knowledge of offences against persons including murder, culpable homicide, assault, kidnapping, and dacoity.",
    "criminal_property":      "You are an expert Indian criminal law assistant specialising in property offences under the IPC and Bharatiya Nyaya Sanhita including theft, robbery, extortion, cheating, fraud, and criminal misappropriation.",
    "civil_general":          "You are an expert Indian civil law assistant with deep knowledge of the Code of Civil Procedure 1908, Indian Evidence Act, Limitation Act, and general civil litigation procedure.",
    "corporate_commercial":   "You are an expert Indian corporate and commercial law assistant with deep knowledge of the Companies Act, Contract Act, Negotiable Instruments Act, Insolvency and Bankruptcy Code, and SEBI regulations.",
    "family_matrimonial":     "You are an expert Indian family law assistant with deep knowledge of Hindu personal law, Muslim personal law, Special Marriage Act, Hindu Succession Act, and guardianship and adoption laws.",
    "land_property":          "You are an expert Indian property law assistant with deep knowledge of the Transfer of Property Act, Registration Act, easements, mortgages, and land acquisition laws.",
    "labour_employment":      "You are an expert Indian labour law assistant with deep knowledge of the Industrial Disputes Act, Factories Act, Payment of Wages Act, Employees Provident Fund Act, and trade union laws.",
    "tax_fiscal":             "You are an expert Indian tax law assistant with deep knowledge of the Income Tax Act, GST legislation, Customs Act, and revenue laws.",
    "cyber_digital":          "You are an expert Indian cyber law assistant with deep knowledge of the IT Act 2000 and 2008 amendments, DPDP Act 2023, cyber crime investigation, electronic evidence under Section 65B, and digital fraud prosecution.",
    "kidnapping_trafficking":  "You are an expert Indian criminal law assistant specialising in kidnapping, abduction, and human trafficking offences under the IPC, Bharatiya Nyaya Sanhita, and POCSO Act.",
    "sexual_offences":        "You are an expert Indian criminal law assistant specialising in sexual offences under the IPC, Bharatiya Nyaya Sanhita, and POCSO Act.",
}

# ── Keyword → domain mapping (ordered: more specific first) ──────────────────
# Each domain gets a list of keywords; matched against lowercased question text.
# First domain whose keywords score highest wins.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "sexual_offences":        ["pocso", "sexual assault", "rape", "molestation", "obscenity", "sexual offence"],
    "kidnapping_trafficking":  ["kidnapping", "abduction", "trafficking", "human trafficking"],
    "cyber_digital":          ["cyber", "it act", "information technology act", "data protection", "dpdp", "electronic evidence", "section 65b", "computer crime"],
    "tax_fiscal":             ["income tax", "gst", "goods and services tax", "customs duty", "excise", "taxation", "revenue", "tax act"],
    "labour_employment":      ["industrial dispute", "workman", "trade union", "factories act", "payment of wages", "epf", "esic", "labour", "employment", "termination of service"],
    "land_property":          ["transfer of property", "easement", "mortgage", "lease", "land acquisition", "registration act", "tpa", "property law"],
    "family_matrimonial":     ["hindu marriage", "muslim personal", "divorce", "maintenance", "custody", "adoption", "succession act", "guardianship", "matrimonial"],
    "corporate_commercial":   ["company law", "companies act", "negotiable instrument", "insolvency", "sebi", "ibc", "arbitration", "contract act", "sale of goods", "banking regulation"],
    "constitutional":         ["fundamental right", "directive principle", "constitutional amendment", "writ petition", "article 32", "article 226", "parliament", "legislature", "president of india", "constitutional"],
    "criminal_property":      ["theft", "extortion", "cheating", "criminal breach of trust", "misappropriation", "receiving stolen", "robbery", "dacoity"],
    "criminal_violent":       ["ipc", "bns", "murder", "culpable homicide", "grievous hurt", "assault", "criminal force", "abetment", "criminal law", "penal code"],
    "civil_general":          ["civil procedure", "cpc", "code of civil procedure", "decree", "injunction", "limitation", "evidence act", "onus of proof", "res judicata"],
}

DEFAULT_DOMAIN = "civil_general"


def classify(question_text: str) -> str:
    text = question_text.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else DEFAULT_DOMAIN


def format_sample(item: dict, domain: str) -> dict:
    opts = item["options"]                      # list [opt_A, opt_B, opt_C, opt_D]
    answer = str(item["answer"]).strip().upper()
    sys_prompt = SYSTEM_PROMPTS[domain]
    user_block = (
        f"{item['question_text'].strip()}\n"
        f"A. {opts[0]}\n"
        f"B. {opts[1]}\n"
        f"C. {opts[2]}\n"
        f"D. {opts[3]}\n"
        f"Answer with only A, B, C, or D."
    )
    text = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_block}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )
    return {
        "text":   text,
        "domain": domain,
        "source": "clat_mcq",
        "exam":   item.get("source_paper", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--out_dir",  default="training/clat-data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download all three subsets ────────────────────────────────────────────
    print("Downloading clat_ug …")
    ug   = load_dataset("adalat-ai/indian-legal-exam-benchmark", "clat_ug",   token=args.hf_token)
    print("Downloading clat_pg …")
    pg   = load_dataset("adalat-ai/indian-legal-exam-benchmark", "clat_pg",   token=args.hf_token)
    print("Downloading djs_dhjs …")
    djs  = load_dataset("adalat-ai/indian-legal-exam-benchmark", "djs_dhjs",  token=args.hf_token)

    # Merge all splits across all subsets
    all_splits = []
    for ds in (ug, pg, djs):
        for split in ds:
            all_splits.append(ds[split])
    combined = concatenate_datasets(all_splits)
    print(f"\nTotal questions: {len(combined)}\n")

    # ── Classify + bucket ─────────────────────────────────────────────────────
    buckets: dict[str, list[dict]] = {d: [] for d in SYSTEM_PROMPTS}
    domain_counter: Counter = Counter()

    for item in combined:
        domain = classify(item["question_text"])
        sample = format_sample(item, domain)
        buckets[domain].append(sample)
        domain_counter[domain] += 1

    # ── Write per-domain JSONL ─────────────────────────────────────────────────
    print("── Domain split ─────────────────────────────────────")
    total_written = 0
    for domain in sorted(buckets, key=lambda d: -len(buckets[d])):
        samples = buckets[domain]
        if not samples:
            continue
        out_path = out_dir / f"{domain}_clat.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {len(samples):>5}  {domain:<30}  → {out_path}")
        total_written += len(samples)

    print(f"\n  Total written: {total_written}")
    print(f"  Output dir  : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
