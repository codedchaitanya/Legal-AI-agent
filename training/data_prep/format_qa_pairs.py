#!/usr/bin/env python3
"""
Format scraped judgments into instruction-tuning QA pairs for QLoRA training.

Takes raw judgment JSONL (from kanoon_scraper.py) and creates training data in
the chat format expected by Qwen2.5-Instruct:

    {"text": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"}

Also supports creating pairs from BNS section JSON (from bns_scraper.py).

Usage:
    # From scraped judgments
    python training/data_prep/format_qa_pairs.py \\
        --input data/raw/criminal_violent_judgments.jsonl \\
        --domain criminal_violent \\
        --output data/training/criminal_violent_train.jsonl

    # From BNS sections (creates section explanation pairs)
    python training/data_prep/format_qa_pairs.py \\
        --input data/legal_db/bns_sections.json \\
        --format sections \\
        --output data/training/bns_section_qa.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Qwen2.5-Instruct chat template
CHAT_TEMPLATE = """<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""


def _load_system_prompt(domain: str) -> str:
    """Load domain-specific system prompt from training config."""
    config_path = Path(__file__).parent.parent / "configs" / f"{domain}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config.get("system_prompt", "").strip()
    return "You are an expert Indian legal assistant. Answer precisely, citing relevant sections."


# ---------------------------------------------------------------------------
# Question templates for judgments
# ---------------------------------------------------------------------------

JUDGMENT_QUESTION_TEMPLATES = [
    "What are the key legal issues in this case and which sections of law apply?",
    "Summarize the judgment and identify the relevant legal provisions cited.",
    "What was the court's reasoning in this case? Which sections were applied?",
    "Explain the legal principles established in this judgment.",
    "What sections of law were discussed and how were they interpreted?",
    "What are the applicable legal sections and what punishment/remedy was prescribed?",
    "Analyse the facts of this case and identify the relevant legal framework.",
    "What precedents and statutory provisions did the court rely on?",
]

SECTION_QUESTION_TEMPLATES = [
    "Explain Section {sec_num} of {act_name}. What does it cover and what is the punishment?",
    "What are the essential elements of the offence under Section {sec_num} {act_short}?",
    "When does Section {sec_num} of {act_name} apply? Give practical examples.",
    "What is the scope of Section {sec_num} {act_short}? What is the {old_equiv} equivalent?",
    "Describe the ingredients and punishment under Section {sec_num} {act_short}.",
]


def format_judgment_pair(judgment: dict, system_prompt: str) -> dict | None:
    """Create a single QA pair from a judgment (legacy, uses 1 random template)."""
    pairs = format_judgment_pairs_multi(judgment, system_prompt, num_pairs=1)
    return pairs[0] if pairs else None


def format_judgment_pairs_multi(judgment: dict, system_prompt: str, num_pairs: int = 1) -> list[dict]:
    """Create multiple diverse QA pairs from a single judgment."""
    text = judgment.get("text", "")
    if not text or len(text) < 300:
        return []

    if len(text) > 6000:
        text = text[:6000] + "\n\n[Judgment text truncated for training.]"

    title = judgment.get("title", "this case")
    doc_id = judgment.get("doc_id", "")
    domain = judgment.get("domain", "")

    templates = random.sample(JUDGMENT_QUESTION_TEMPLATES, min(num_pairs, len(JUDGMENT_QUESTION_TEMPLATES)))

    pairs = []
    for i, template in enumerate(templates):
        question = f"Case: {title}\n\nJudgment excerpt:\n{text[:3000]}\n\n{template}"

        if i == 0:
            analysis_start = len(text) // 2
            answer_text = text[analysis_start:]
        elif i == 1:
            analysis_start = len(text) // 3
            answer_text = text[analysis_start:analysis_start + 2500]
        else:
            analysis_start = max(0, len(text) - 2500)
            answer_text = text[analysis_start:]

        if len(answer_text) > 2000:
            answer_text = answer_text[:2000]

        formatted = CHAT_TEMPLATE.format(
            system_prompt=system_prompt,
            question=question,
            answer=answer_text.strip(),
        )

        pairs.append({
            "text": formatted,
            "domain": domain,
            "source": "judgment",
            "doc_id": doc_id,
            "pair_idx": i,
        })

    return pairs


def format_section_pair(section: dict, system_prompt: str) -> dict | None:
    """Create a QA pair from a legal section."""
    text = section.get("text", "")
    sec_num = section.get("section_number", "")
    if not text or not sec_num:
        return None

    act_name = section.get("act", "")
    act_short = section.get("act_short", "")
    old_equiv = section.get("old_equivalent", "N/A")

    template = random.choice(SECTION_QUESTION_TEMPLATES)
    question = template.format(sec_num=sec_num, act_name=act_name, act_short=act_short, old_equiv=old_equiv)

    answer = f"Section {sec_num} of {act_name} ({act_short}):\n\n{text}"
    if old_equiv:
        answer += f"\n\nNote: This corresponds to {old_equiv} under the old law."

    formatted = CHAT_TEMPLATE.format(
        system_prompt=system_prompt,
        question=question,
        answer=answer.strip(),
    )

    return {"text": formatted, "domain": section.get("domain", ""), "source": "section", "section_number": sec_num}


def process_judgments_file(input_path: Path, domain: str, output_path: Path, pairs_per_judgment: int = 1) -> int:
    """Process a judgment JSONL file into training pairs.

    Args:
        pairs_per_judgment: Number of distinct QA pairs to generate per judgment.
            Use >1 to amplify training data with diverse question angles.
    """
    system_prompt = _load_system_prompt(domain)
    pairs = []
    seen_doc_ids: set[str] = set()

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            judgment = json.loads(line)
            doc_id = judgment.get("doc_id", "")
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            new_pairs = format_judgment_pairs_multi(judgment, system_prompt, num_pairs=pairs_per_judgment)
            pairs.extend(new_pairs)

    random.shuffle(pairs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Created %d training pairs from %d unique judgments → %s", len(pairs), len(seen_doc_ids), output_path)
    return len(pairs)


def process_sections_file(input_path: Path, output_path: Path) -> int:
    """Process a sections JSON file into training pairs."""
    with open(input_path, encoding="utf-8") as f:
        sections = json.load(f)

    pairs = []
    for section in sections:
        domain = section.get("domain", "civil_general")
        system_prompt = _load_system_prompt(domain)
        pair = format_section_pair(section, system_prompt)
        if pair:
            pairs.append(pair)

    random.shuffle(pairs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Created %d section pairs → %s", len(pairs), output_path)
    return len(pairs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Format scraped data into QLoRA training pairs")
    parser.add_argument("--input", required=True, help="Input file (JSONL for judgments, JSON for sections)")
    parser.add_argument("--format", choices=["judgments", "sections"], default="judgments")
    parser.add_argument("--domain", type=str, default="civil_general", help="Domain (for judgment files)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--pairs-per-judgment", type=int, default=1, help="QA pairs per judgment (>1 for data augmentation)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.format == "sections":
        process_sections_file(Path(args.input), Path(args.output))
    else:
        process_judgments_file(Path(args.input), args.domain, Path(args.output), pairs_per_judgment=args.pairs_per_judgment)


if __name__ == "__main__":
    main()
